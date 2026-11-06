"""Prediction decoding: shared NMS, YuNet, RetinaFace MNV1 0.25, MTCNN cascade."""
import math
import numpy as np

RETINA_CFG = {'steps': [8, 16, 32], 'min_sizes': [[16, 32], [64, 128], [256, 512]]}


# Shared NMS

def nms(boxes, threshold):
    """Inclusive-coordinate IoU, matching the evaluation protocol (+1 pixel)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    order = np.argsort(-boxes[:, 4], kind='stable')
    area = np.prod(boxes[:, 2:4] - boxes[:, :2] + 1, axis=1)
    kept = []
    while len(order):
        current = order[0]
        kept.append(current)
        rest = order[1:]
        low = np.maximum(boxes[current, :2], boxes[rest, :2])
        high = np.minimum(boxes[current, 2:4], boxes[rest, 2:4])
        intersection = np.prod(np.maximum(high - low + 1, 0), axis=1)
        union = area[current] + area[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlap <= threshold]
    return np.asarray(kept, dtype=np.int64)

# YuNet

def sigmoid_np(x):
    x = np.asarray(x, np.float32)
    return np.exp(-np.logaddexp(np.float32(0), -x)).astype(np.float32)

def decode_yunet(output, size, score_floor=0.001, top_k=5000, nms_iou=0.3):
    cls = np.concatenate([output[f'cls_{s}_logits'] for s in [8, 16, 32]], axis=1)[0, :, 0]
    obj = np.concatenate([output[f'obj_{s}_logits'] for s in [8, 16, 32]], axis=1)[0, :, 0]
    raw = np.concatenate([output[f'bbox_{s}'] for s in [8, 16, 32]], axis=1)[0]
    lm = np.concatenate([output[f'kps_{s}'] for s in [8, 16, 32]], axis=1)[0]
    grids = []
    strides = []
    w, h = size
    for stride in [8, 16, 32]:
        yy, xx = np.meshgrid(np.arange(h // stride), np.arange(w // stride), indexing='ij')
        grids.append(np.stack([xx, yy], axis=-1).reshape(-1, 2))
        strides.append(np.full((xx.size, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(strides).astype(np.float32)
    score = np.sqrt(sigmoid_np(cls) * sigmoid_np(obj))
    center = (raw[:, :2] + grid) * stride
    wh = np.exp(np.clip(raw[:, 2:], -15, 15)) * stride
    boxes = np.concatenate([center - wh / 2, center + wh / 2], axis=1)
    points = (lm.reshape(-1, 5, 2) + grid[:, None]) * stride[:, None]
    ids = np.flatnonzero(score > score_floor)
    ids = ids[np.argsort(-score[ids], kind='stable')[:top_k]]
    pred = np.column_stack([boxes[ids], score[ids]])
    points = points[ids]
    keep = nms(pred, nms_iou)
    return (pred[keep], points[keep])

# RetinaFace MNV1 0.25

def decode_retinaface(output, size, score_floor=0.001, top_k=5000, nms_iou=0.3):
    w, h = size
    anchors = []
    for step, sizes in zip(RETINA_CFG['steps'], RETINA_CFG['min_sizes']):
        for y in range(math.ceil(h / step)):
            for x in range(math.ceil(w / step)):
                for side in sizes:
                    anchors.append([(x + 0.5) * step / w, (y + 0.5) * step / h, side / w, side / h])
    prior = np.asarray(anchors, np.float32)
    loc = output['box'][0]
    logit = output['class'][0]
    lm = output['landmark'][0]
    exp = np.exp(logit - logit.max(axis=1, keepdims=True))
    score = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    ids = np.flatnonzero(score > score_floor)
    ids = ids[np.argsort(-score[ids], kind='stable')[:top_k]]
    p = prior[ids]
    raw = loc[ids]
    center = p[:, :2] + raw[:, :2] * np.float32(0.1) * p[:, 2:]
    wh = p[:, 2:] * np.exp(raw[:, 2:] * np.float32(0.2))
    scale = np.asarray([w, h, w, h], np.float32)
    boxes = np.concatenate([center - wh / 2, center + wh / 2], axis=1) * scale
    points = (p[:, None, :2] + lm[ids].reshape(-1, 5, 2) * np.float32(0.1) * p[:, None, 2:]) * np.asarray(size, np.float32)
    pred = np.column_stack([boxes, score[ids]])
    keep = nms(pred, nms_iou)
    return (pred[keep], points[keep])

# MTCNN cascade

def area_resize_np(image, size_hw):
    oh, ow = size_hw
    ih, iw = image.shape[:2]
    y0 = np.floor(np.arange(oh) * ih / oh).astype(int)
    y1 = np.ceil((np.arange(oh) + 1) * ih / oh).astype(int)
    x0 = np.floor(np.arange(ow) * iw / ow).astype(int)
    x1 = np.ceil((np.arange(ow) + 1) * iw / ow).astype(int)
    summed = np.pad(np.asarray(image, np.float64), ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
    total = summed[y1[:, None], x1] - summed[y0[:, None], x1] - summed[y1[:, None], x0] + summed[y0[:, None], x0]
    return (total / ((y1 - y0)[:, None] * (x1 - x0))[..., None]).astype(np.float32)

def cascade_nms(boxes, threshold, minimum=False):
    if not len(boxes):
        return np.empty(0, np.int64)
    if minimum:
        order = np.argsort(boxes[:, 4])
        margin = np.float32(1.0)
    else:
        order = np.argsort(-boxes[:, 4], kind='stable')[::-1]
        margin = np.float32(0.0)
    area = (boxes[:, 2] - boxes[:, 0] + margin) * (boxes[:, 3] - boxes[:, 1] + margin)
    keep = []
    while len(order):
        i = order[-1]
        keep.append(i)
        rest = order[:-1]
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:4], boxes[rest, 2:4])
        wh = np.maximum(np.float32(0), hi - lo + margin)
        inter = wh[:, 0] * wh[:, 1]
        if minimum:
            denominator = np.minimum(area[i], area[rest])
        else:
            denominator = area[i] + area[rest] - inter
        overlap = np.divide(inter, denominator, out=np.zeros_like(inter), where=denominator > 0)
        order = rest[overlap <= threshold]
    return np.asarray(keep, np.int64)

def square_boxes(boxes):
    boxes = boxes.copy()
    wh = boxes[:, 2:4] - boxes[:, :2]
    side = wh.max(axis=1)
    boxes[:, :2] += wh * np.float32(0.5) - side[:, None] * np.float32(0.5)
    boxes[:, 2:4] = boxes[:, :2] + side[:, None]
    return boxes

def regress_boxes(boxes, regression, inclusive=True):
    boxes = boxes.copy()
    margin = np.float32(0.0)
    if inclusive:
        margin = np.float32(1.0)
    wh = boxes[:, 2:4] - boxes[:, :2] + margin
    boxes[:, :4] += regression * np.column_stack([wh, wh])
    return boxes

def cascade_crops(image, boxes, side):
    h, w = image.shape[:2]
    integer = np.trunc(boxes[:, :4]).astype(int)
    integer[:, 0] = np.maximum(integer[:, 0], 1)
    integer[:, 1] = np.maximum(integer[:, 1], 1)
    integer[:, 2] = np.minimum(integer[:, 2], w)
    integer[:, 3] = np.minimum(integer[:, 3], h)
    ids = np.flatnonzero((integer[:, 2] >= integer[:, 0]) & (integer[:, 3] >= integer[:, 1]))
    crops = [area_resize_np(image[y1 - 1:y2, x1 - 1:x2], (side, side)) for x1, y1, x2, y2 in integer[ids]]
    if crops:
        data = np.stack(crops)
    else:
        data = np.empty((0, side, side, 3), np.float32)
    return (ids, (data - np.float32(127.5)) * np.float32(0.0078125))

def mtcnn_numpy(image, heads):
    image = np.asarray(image, np.float32)
    h, w = image.shape[:2]
    scale = 0.6
    small = min(h, w) * scale
    proposals = []
    while small >= 12:
        resized = area_resize_np(image, (int(h * scale + 1), int(w * scale + 1)))
        out = heads('pnet', (resized[None] - np.float32(127.5)) * np.float32(0.0078125))
        y, x = np.where(out['prob'][0, 1] >= 0.6)
        xy = np.stack([x, y], axis=1).astype(np.float32)
        boxes = np.column_stack([np.floor((2 * xy + 1) / np.float32(scale)), np.floor((2 * xy + 12) / np.float32(scale)), out['prob'][0, 1, y, x], out['box'][0].transpose(1, 2, 0)[y, x]])
        proposals.append(boxes[cascade_nms(boxes, 0.5)])
        scale *= 0.709
        small *= 0.709
    if proposals:
        boxes = np.concatenate(proposals)
    else:
        boxes = np.empty((0, 9), np.float32)
    if not len(boxes):
        return (np.empty((0, 5)), np.empty((0, 5, 2)))
    boxes = boxes[cascade_nms(boxes, 0.7)]
    boxes = square_boxes(regress_boxes(boxes[:, :5], boxes[:, 5:9], inclusive=False))
    ids, crops = cascade_crops(image, boxes, 24)
    boxes = boxes[ids]
    if not len(boxes):
        return (np.empty((0, 5)), np.empty((0, 5, 2)))
    out = heads('rnet', crops)
    score = out['prob'][:, 1]
    passed = score > 0.7
    boxes = np.column_stack([boxes[passed, :4], score[passed]])
    reg = out['box'][passed]
    keep = cascade_nms(boxes, 0.7)
    boxes = square_boxes(regress_boxes(boxes[keep], reg[keep]))
    ids, crops = cascade_crops(image, boxes, 48)
    boxes = boxes[ids]
    if not len(boxes):
        return (np.empty((0, 5)), np.empty((0, 5, 2)))
    out = heads('onet', crops)
    score = out['prob'][:, 1]
    passed = score > 0.7
    boxes = np.column_stack([boxes[passed, :4], score[passed]])
    raw = out['landmark'][passed]
    reg = out['box'][passed]
    wh = boxes[:, 2:4] - boxes[:, :2] + 1
    points = np.stack([wh[:, 0, None] * raw[:, :5] + boxes[:, 0, None] - 1, wh[:, 1, None] * raw[:, 5:] + boxes[:, 1, None] - 1], axis=-1)
    boxes = regress_boxes(boxes, reg)
    keep = cascade_nms(boxes, 0.7, minimum=True)
    return (boxes[keep], points[keep])
