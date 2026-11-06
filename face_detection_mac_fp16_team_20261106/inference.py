"""TFLite FP16 portrait inference. Run: python inference.py --model all --input images"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import os
import time
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from decoders import decode_yunet, decode_retinaface, mtcnn_numpy

PACKAGE_ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((PACKAGE_ROOT / 'model_manifest.json').read_text(encoding='utf-8'))
MODEL_NAMES = list(MANIFEST['models'])
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


# 1. TFLite runtime

class LiteNetwork:
    """Signature names and fixed image shape are validated before inference."""

    def __init__(self, path, input_name, threads=2, initial_shape=None):
        self.interpreter = tf.lite.Interpreter(
            model_content=Path(path).read_bytes(), num_threads=threads
        )
        if initial_shape is not None:
            index = self.interpreter.get_input_details()[0]['index']
            self.interpreter.resize_tensor_input(index, initial_shape, strict=False)
        self.interpreter.allocate_tensors()
        self.shape = tuple(self.interpreter.get_input_details()[0]['shape'])
        self.input_name = input_name
        self.runner = self.interpreter.get_signature_runner()

    def __call__(self, pixels):
        return self.runner(**{self.input_name: np.asarray(pixels, dtype=np.float32)})

# 2. Load and letterbox

def load_image(path):
    """Return RGB pixels in the displayed orientation, including phone EXIF rotation."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')

def letterbox(image, size=(480, 640)):
    """Resize RGB without stretching; pad to portrait 480 wide x 640 high."""
    ratio = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.BILINEAR)
    left, top = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas = Image.new('RGB', size, (0, 0, 0))
    canvas.paste(resized, (left, top))
    return (canvas, (resized.width / image.width, resized.height / image.height, left, top))

# 3. Detector API

class Detector:
    def __init__(self, model='yunet', size=(480, 640), threads=2, confidence=0.5):
        if model not in MODEL_NAMES:
            raise ValueError(f'Unknown model: {model}. Available: {MODEL_NAMES}')
        if list(size) not in MANIFEST['sizes']:
            raise ValueError('This team package supports portrait 480x640 only (width x height)')
        if not 0 <= confidence <= 1:
            raise ValueError('confidence must be between 0 and 1')
        if threads < 1:
            raise ValueError('threads must be positive')
        self.model = model
        self.size = tuple(size)
        self.confidence = confidence
        self.last_ms = 0.0
        folder = PACKAGE_ROOT / 'models' / model
        if model == 'mtcnn':
            self.stages = {}
            for stage, side in [('pnet', 32), ('rnet', 24), ('onet', 48)]:
                self.stages[stage] = LiteNetwork(
                    folder / f'{stage}.tflite', 'image', threads,
                    initial_shape=[1, side, side, 3]
                )
        else:
            folder = folder / f'{size[0]}x{size[1]}'
            self.network = LiteNetwork(folder / 'model.tflite', 'images', threads)
            expected = (1, size[1], size[0], 3)
            if self.network.shape != expected:
                raise ValueError(f'Model input {self.network.shape} != {expected}')
            info = json.loads((folder / 'conversion.json').read_text(encoding='utf-8'))
            self.output_names = info['files'][0]['output_names']

    def cascade_stage(self, stage, batch):
        chunks = []
        for start in range(0, len(batch), 512):
            chunks.append(self.stages[stage](batch[start:start + 512]))
        result = {}
        for key in chunks[0]:
            result[key] = np.concatenate([chunk[key] for chunk in chunks])
        return result

    def __call__(self, image):
        """Return Nx5 [x1,y1,x2,y2,score], Nx5x2 landmarks in original image pixels.

        Dense graphs accept RGB float32 /255, NHWC. Their graph contains the
        original training colour/normalization conversion; do not repeat it.
        Letterbox is black, aspect-preserving, with the evaluation half-pixel
        inverse landmark mapping. Coordinates are not clipped for evaluation.
        """
        started = time.perf_counter()
        canvas, (sx, sy, left, top) = letterbox(image.convert('RGB'), self.size)
        pixels = np.asarray(canvas, dtype=np.float32)
        if self.model == 'mtcnn':
            boxes, points = mtcnn_numpy(pixels, self.cascade_stage)
        else:
            values = self.network(pixels[None] / np.float32(255))
            outputs = {}
            for index, name in enumerate(self.output_names):
                outputs[name] = values[f'output_{index:02d}']
            if self.model == 'yunet':
                boxes, points = decode_yunet(outputs, self.size)
            else:
                outputs = {'box': outputs['boxes'], 'class': outputs['class_logits'],
                           'landmark': outputs['landmarks']}
                boxes, points = decode_retinaface(outputs, self.size)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
        points = np.asarray(points, dtype=float).reshape(-1, 5, 2)
        boxes[:, :4] = (boxes[:, :4] - [left, top, left, top]) / [sx, sy, sx, sy]
        points = (points - [left, top] + 0.5) / [sx, sy] - 0.5
        valid = np.isfinite(boxes).all(1) & np.isfinite(points).all((1, 2))
        valid &= (boxes[:, 2:4] > boxes[:, :2]).all(1)
        valid &= boxes[:, 4] >= self.confidence
        boxes, points = boxes[valid], points[valid]
        order = np.argsort(-boxes[:, 4], kind='stable')
        self.last_ms = (time.perf_counter() - started) * 1000
        return boxes[order], points[order]

# 4. Draw results

def draw_detections(image, boxes, landmarks):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, image.width // 70))
    width = max(2, image.width // 400)
    radius = max(2, image.width // 250)
    for box, points in zip(boxes, landmarks):
        draw.rectangle(tuple(box[:4]), outline='#00dd55', width=width)
        draw.text((max(0, box[0]), max(0, box[1] - 20)), f'{box[4]:.3f}',
                  fill='yellow', font=font, stroke_width=1, stroke_fill='black')
        for x, y in points:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='orange')
    draw.text((8, 8), 'Green: face | Orange: 5 landmarks | Yellow: confidence',
              fill='white', font=font, stroke_width=1, stroke_fill='black')
    return canvas

def write_gallery(output, records):
    cards = []
    for row in records:
        title = html.escape(f"{row['model']} | {row['image']} | {row['faces']} faces")
        path = quote(row['annotated_image'], safe='/')
        cards.append(f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{path}"></figure>')
    content = ('<!doctype html><html><meta charset="utf-8"><title>All inference results</title>'
               '<style>body{font:16px sans-serif;margin:24px;background:#fafafa}'
               'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}'
               'figure{margin:0;padding:12px;background:white}img{width:100%;height:auto}'
               'figcaption{margin-bottom:10px;overflow-wrap:anywhere}</style>'
               '<h1>All inference results</h1><main>' + ''.join(cards) + '</main></html>')
    (output / 'gallery.html').write_text(content, encoding='utf-8')

# 5. Folder inference and command line

def run_folder(input_path, output_path, models, size=(480, 640), threads=2, confidence=0.5):
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError('Input and output must be different folders')
    if source.is_file():
        paths = [source]
        base = source.parent
    elif source.is_dir():
        base = source
        paths = sorted(p for p in source.rglob('*') if p.is_file()
                       and p.suffix.lower() in EXTENSIONS and not p.is_relative_to(output))
    else:
        raise FileNotFoundError(source)
    if not paths:
        raise ValueError('No supported images found. Put JPG/PNG images in the input folder.')
    if (output / 'summary.csv').exists():
        raise FileExistsError(f'Use a new output folder to preserve the earlier run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    for model in models:
        detector = Detector(model, size, threads, confidence)
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(base)
            try:
                image = load_image(path)
            except Exception as error:
                errors.append({'image': relative.as_posix(), 'model': model, 'error': str(error)})
                print(f'SKIP {relative}: {error}', flush=True)
                continue
            boxes, points = detector(image)
            target = output / model / relative.parent / (relative.name + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            draw_detections(image, boxes, points).save(target)
            result = {'model': model, 'image': relative.as_posix(), 'size_wh': list(image.size),
                      'input_wh': list(size), 'coordinate_space': 'EXIF-oriented original image pixels',
                      'boxes_xyxy_score': boxes.tolist(), 'landmarks_xy': points.tolist(),
                      'inference_ms': detector.last_ms, 'checkpoint': MANIFEST['models'][model]}
            target.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            records.append({'model': model, 'image': relative.as_posix(), 'faces': len(boxes),
                            'inference_ms': detector.last_ms,
                            'annotated_image': target.relative_to(output).as_posix()})
            print(f'{model} [{index}/{len(paths)}] {relative}: {len(boxes)} faces, '
                  f'{detector.last_ms:.1f} ms', flush=True)
        del detector
    columns = ['model', 'image', 'faces', 'inference_ms', 'annotated_image']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    (output / 'errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
    write_gallery(output, records)
    if not records:
        raise RuntimeError('All input images failed to load; see errors.json')
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all'] + MODEL_NAMES, default='yunet')
    parser.add_argument('--input', default='images')
    parser.add_argument('--output', default=None)
    parser.add_argument('--size', choices=['480x640'], default='480x640')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--check', action='store_true', help='Check model hashes and run each model on a blank image')
    args = parser.parse_args()
    if args.check:
        check_models()
        return
    models = [args.model]
    if args.model == 'all':
        models = MODEL_NAMES
    output = args.output
    if output is None:
        output = 'outputs/run_' + time.strftime('%Y%m%d_%H%M%S')
    size = tuple(int(value) for value in args.size.split('x'))
    run_folder(args.input, output, models, size, args.threads, args.confidence)
    print(f'Open: {Path(output).resolve() / "gallery.html"}')

def check_models():
    for line in (PACKAGE_ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        if name.startswith('models/'):
            actual = hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Model hash mismatch: {name}')
    for name in MODEL_NAMES:
        model = Detector(name)
        boxes, points = model(Image.new('RGB', (480,640)))
        assert boxes.shape == (len(boxes),5)
        assert points.shape == (len(boxes),5,2)
        print(name, '480x640 PASS')


if __name__ == '__main__':
    main()
