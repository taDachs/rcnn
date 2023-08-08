#!/usr/bin/env python3

# ---
from typing import List, Tuple, Sequence

import tensorflow as tf
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


# ---
IMG_SHAPE = (64, 64, 3)


# ---
def generate_squares(n: int, size: int, img: np.ndarray):
    img = np.copy(img)
    img_w, img_h, _ = img.shape

    x = np.random.randint(0, img_w, size=n)
    y = np.random.randint(0, img_h, size=n)
    poses = np.stack((x, y), axis=-1)
    sizes = np.full((n, 2), size)

    squares = np.concatenate((poses, sizes), axis=-1)

    for x, y, w, h in squares:
        img[x : x + w, y : y + h] = (1.0, 0, 0)

    bboxes = np.concatenate((poses, poses + sizes), axis=-1) / (img_w, img_h, img_w, img_h)

    return img, bboxes


# ---
img = np.zeros(IMG_SHAPE)
squares, bboxes = generate_squares(10, 9, img)


# ---
def generate_anchor_boxes(
    grid_size: Tuple[int, int],
    base_size: int = 1,
    aspect_ratios=(0.5, 1.0, 2.0),
    sclaes=(1, 2, 3),
) -> np.ndarray:
    """
    Generate anchor boxes for object detection with R-CNN for each cell in a grid.

    Args:
        grid_size (tuple): Number of cells in the grid (rows, columns).
        base_size (int): The base size of the anchor box (usually the smaller dimension).
        aspect_ratios (list): List of aspect ratios for generating different box shapes.
        scales (list): List of scales to multiply the base size by.

    Returns:
        anchor_boxes (list): List of anchor boxes in the format (x_center, y_center, width, height).
    """
    anchor_boxes = []
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            for scale in scales:
                for ratio in aspect_ratios:
                    width = base_size * scale / grid_size[0]
                    height = base_size * scale * ratio / grid_size[1]
                    x_center = (x + 0.5) / grid_size[0]
                    y_center = (y + 0.5) / grid_size[1]

                    anchor_boxes.append((x_center, y_center, width, height))

    return np.array(anchor_boxes)


# ---
def xyxy_to_ccwh(bboxes):
    x1, y1, x2, y2 = bboxes.T

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return np.stack((cx, cy, w, h), axis=-1)


def ccwh_to_xyxy(bboxes):
    x, y, w, h = bboxes.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return np.stack((x1, y1, x2, y2), axis=-1)


def rel_to_grid(bboxes, grid_size):
    w, h = grid_size
    bboxes = bboxes * (w, h, w, h)
    return bboxes


def grid_to_rel(bboxes, grid_size):
    w, h = grid_size
    bboxes = bboxes / (w, h, w, h)
    return bboxes


# ---
# Define parameters
grid_size = (8, 8)  # Example grid size (rows, columns)
base_size = 1
aspect_ratios = [0.5, 1.0, 2.0]
scales = [1, 2, 3]

# Generate anchor boxes
anchor_boxes = generate_anchor_boxes(grid_size, base_size, aspect_ratios, scales)
print(anchor_boxes)
anchor_boxes = ccwh_to_xyxy(anchor_boxes)
print(rel_to_grid(anchor_boxes, grid_size))

# ---
img = squares
# Convert the anchor boxes to TensorFlow format
anchor_boxes_tf = np.array(anchor_boxes, dtype=np.float32)
anchor_boxes_tf = np.expand_dims(anchor_boxes_tf, axis=0)  # Add batch dimension

# Create a TensorFlow tensor for the image
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

# Draw anchor boxes on the image using TensorFlow
drawn_img = tf.image.draw_bounding_boxes(
    img_tensor, anchor_boxes_tf[0, 20 * 9 : 20 * 9 + 9][None], ((0.0, 1.0, 0.0),)
)
# drawn_img = tf.image.draw_bounding_boxes(img_tensor, bboxes[None], ((0.0, 1.0, 0.0),))

# Convert the tensor to a NumPy array for visualization
drawn_img = drawn_img.numpy()

# ---
# Display the image with anchor boxes
plt.imshow(drawn_img[0])
plt.axis("off")
plt.show()


# ---
def calculate_iou_array(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) between arrays of bounding boxes.

    Args:
        boxes1 (np.ndarray): Array of bounding box coordinates (N x 4), where N is the number of boxes.
        boxes2 (np.ndarray): Array of bounding box coordinates (M x 4), where M is the number of boxes.

    Returns:
        iou_matrix (np.ndarray): Matrix of IoU values (N x M).
    """
    x1_min = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
    y1_min = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
    x2_max = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
    y2_max = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])

    intersection_area = np.maximum(0, x2_max - x1_min) * np.maximum(0, y2_max - y1_min)

    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = box1_area[:, np.newaxis] + box2_area - intersection_area

    iou_matrix = intersection_area / union_area
    iou_matrix[union_area <= 0] = 0.0  # Set IoU to 0 where union area is non-positive

    return iou_matrix


# Example arrays of bounding boxes in the format (x1, y1, x2, y2)
boxes1 = np.array([[50, 50, 150, 150], [100, 100, 200, 200], [200, 200, 300, 300]])

boxes2 = np.array(
    [
        [50, 50, 150, 150],
        [75, 75, 175, 175],
        [125, 125, 225, 225],
        [250, 250, 350, 350],
    ]
)

# Calculate IoU matrix
iou_matrix = calculate_iou_array(boxes1, boxes2)
print("IoU Matrix:\n", iou_matrix)


# ---
def calculate_offsets(anc: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x_a, y_a, w_a, h_a = anc.T
    x, y, w, h = gt.T

    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = np.log(w / w_a)
    t_h = np.log(h / h_a)

    return np.stack((t_x, t_y, t_w, t_h), axis=-1)


def apply_offsets(anc: np.ndarray, offset: np.ndarray) -> np.ndarray:
    x_a, y_a, w_a, h_a = anc.T
    t_x, t_y, t_w, t_h = offset.T

    x = t_x * w_a + x_a
    y = t_y * h_a + y_a
    w = np.exp(t_w) * w_a
    h = np.exp(t_h) * h_a

    return np.stack((x, y, w, h), axis=-1)


# ---


def label_img(
    bboxes: np.ndarray,
    grid_size: Tuple[int, int],
    pos_thresh: float = 0.7,
    neg_thresh=0.5,
    anc_size: int = 1,
    anc_ratios=(0.5, 1.0, 2.0),
    sclaes=(1, 2, 3),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ancs = generate_anchor_boxes(grid_size, base_size, aspect_ratios, scales)
    ancs = ccwh_to_xyxy(ancs)
    A = ancs.shape[0]
    B = bboxes.shape[0]
    anc_mapping = np.zeros(A, dtype=int)  # (A, ) maps each anchor to a gt bbox
    iou = calculate_iou_array(ancs, bboxes)  # (A, B)

    # map each gt bbox to a anchor
    max_iou_for_gt_idx = np.argmax(iou, axis=0)  # (B, )
    anc_mapping[max_iou_for_gt_idx] = np.arange(B)
    pre_mapped_mask = np.zeros(A, dtype=bool)
    pre_mapped_mask[max_iou_for_gt_idx] = True

    # map anc boxes with iou > pos_thesh to gt bbox
    iou_thresh_mask = np.max(iou, axis=1) > pos_thresh  # (A, )
    iou_thresh_mask &= ~pre_mapped_mask
    max_iou_for_anc_idx = np.argmax(iou, axis=1)  # (A, )
    anc_mapping[iou_thresh_mask] = max_iou_for_anc_idx[iou_thresh_mask]

    pos_mask = pre_mapped_mask | iou_thresh_mask
    neg_mask = np.max(iou, axis=1) < neg_thresh
    neg_mask &= ~pos_mask

    ancs = xyxy_to_ccwh(ancs)
    mapped_bboxes = bboxes[anc_mapping]
    mapped_bboxes = xyxy_to_ccwh(mapped_bboxes)
    offsets = calculate_offsets(ancs, mapped_bboxes)

    return anc_mapping, offsets, pos_mask, neg_mask


# ---
img = np.zeros(IMG_SHAPE)
squares, bboxes = generate_squares(10, 9, img)
anc_mapping, offsets, pos_mask, neg_mask = label_img(bboxes, grid_size)
ancs = generate_anchor_boxes(grid_size)

mapped_bboxes = apply_offsets(ancs, offsets)
ancs = ccwh_to_xyxy(ancs)
mapped_bboxes = ccwh_to_xyxy(mapped_bboxes)

colors = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.5, 1.0, 5.0),
    (1.0, 5.0, 1.0),
    (0.5, 1.0, 1.0),
    (0.1, 0.4, 0.2),
)

drawn_bboxes = squares[None, ...]
for idx in np.where(pos_mask)[0]:
    drawn_bboxes = tf.image.draw_bounding_boxes(
        drawn_bboxes, mapped_bboxes[idx][None, None], (colors[anc_mapping[idx]],)
    )
    drawn_bboxes = tf.image.draw_bounding_boxes(
        drawn_bboxes, ancs[idx][None, None], (colors[anc_mapping[idx]],)
    )

# ---
plt.imshow(drawn_bboxes[0])
plt.show()
