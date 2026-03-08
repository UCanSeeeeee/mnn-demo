#!/usr/bin/env python3
"""
Ultra-Light Face Detector using MNN (slim-320 model)
Self-contained script - only requires: MNN, opencv-python, numpy
"""
import os
import sys
import time
import argparse
from math import ceil

import numpy as np
import cv2
import MNN

# ==================== Box Utilities (inline) ====================

def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate([
        locations[..., :2] - locations[..., 2:] / 2,
        locations[..., :2] + locations[..., 2:] / 2
    ], len(locations.shape) - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        overlap_lt = np.maximum(rest_boxes[:, :2], current_box[:2])
        overlap_rb = np.minimum(rest_boxes[:, 2:], current_box[2:])
        hw = np.clip(overlap_rb - overlap_lt, 0.0, None)
        overlap_area = hw[:, 0] * hw[:, 1]
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        area_cur = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        iou = overlap_area / (area_rest + area_cur - overlap_area + 1e-5)
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]

# ==================== Prior Box Generation ====================

IMAGE_MEAN = np.array([127, 127, 127])
IMAGE_STD = 128.0
CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2
MIN_BOXES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STRIDES = [8, 16, 32, 64]
IOU_THRESHOLD = 0.3


def generate_priors(image_size):
    feature_map_w = [ceil(image_size[0] / s) for s in STRIDES]
    feature_map_h = [ceil(image_size[1] / s) for s in STRIDES]
    priors = []
    for idx in range(len(STRIDES)):
        scale_w = image_size[0] / STRIDES[idx]
        scale_h = image_size[1] / STRIDES[idx]
        for j in range(feature_map_h[idx]):
            for i in range(feature_map_w[idx]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h
                for min_box in MIN_BOXES[idx]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    priors = np.array(priors, dtype=np.float32)
    np.clip(priors, 0.0, 1.0, out=priors)
    return priors

# ==================== Detection ====================

def predict(width, height, confidences, boxes, prob_threshold):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=IOU_THRESHOLD, top_k=-1)
        picked_box_probs.append(box_probs)
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.ones(len(picked_box_probs)), picked_box_probs[:, 4]


def main():
    parser = argparse.ArgumentParser(description='MNN Face Detector Test')
    parser.add_argument('--model', default='model/slim-320.mnn', help='model path')
    parser.add_argument('--imgs', default='imgs', help='input images dir')
    parser.add_argument('--output', default='results', help='output dir')
    parser.add_argument('--threshold', default=0.7, type=float, help='confidence threshold')
    parser.add_argument('--input_size', default='320,240', help='model input WxH')
    args = parser.parse_args()

    input_size = [int(v) for v in args.input_size.split(',')]
    priors = generate_priors(input_size)
    print(f"Priors generated: {len(priors)}")

    os.makedirs(args.output, exist_ok=True)

    img_files = sorted([f for f in os.listdir(args.imgs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not img_files:
        print(f"No images found in {args.imgs}")
        return

    for fname in img_files:
        img_path = os.path.join(args.imgs, fname)
        image_ori = cv2.imread(img_path)
        if image_ori is None:
            print(f"Failed to read {img_path}, skipping")
            continue

        h, w = image_ori.shape[:2]

        # MNN inference
        interpreter = MNN.Interpreter(args.model)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)

        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(input_size))
        image = (image.astype(np.float32) - IMAGE_MEAN) / IMAGE_STD
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        tmp_input = MNN.Tensor(
            (1, 3, input_size[1], input_size[0]),
            MNN.Halide_Type_Float,
            np.ascontiguousarray(image, dtype=np.float32).flatten().tolist(),
            MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp_input)

        t0 = time.time()
        interpreter.runSession(session)
        t1 = time.time()

        scores = interpreter.getSessionOutput(session, "scores").getData()
        boxes_raw = interpreter.getSessionOutput(session, "boxes").getData()

        boxes_np = np.expand_dims(np.reshape(boxes_raw, (-1, 4)), axis=0)
        scores_np = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

        # Decode
        boxes_decoded = convert_locations_to_boxes(boxes_np, priors, CENTER_VARIANCE, SIZE_VARIANCE)
        boxes_decoded = center_form_to_corner_form(boxes_decoded)
        det_boxes, labels, probs = predict(w, h, scores_np, boxes_decoded, args.threshold)

        # Draw results
        num_faces = 0
        if len(det_boxes) > 0:
            num_faces = len(det_boxes)
            for i in range(num_faces):
                box = det_boxes[i]
                score = probs[i]
                cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image_ori, f"{score:.2f}", (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(args.output, fname)
        cv2.imwrite(out_path, image_ori)
        print(f"[{fname}] inference: {(t1-t0)*1000:.1f}ms | faces: {num_faces} | saved: {out_path}")

    print(f"\nDone! All results saved to '{args.output}/' directory.")


if __name__ == '__main__':
    main()
