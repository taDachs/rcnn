import math

import numpy as np
import tensorflow as tf


def xyxy_to_ccwh(bboxes):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return np.stack((cx, cy, w, h), axis=-1)


def ccwh_to_xyxy(bboxes):
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
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


def visualize_labels(
    img,
    ancs,
    offsets,
    pred,
    anc_mapping=None,
    thresh: float = 0.5,
    show_ancs=True,
    show_offsets=True,
    show_heatmap=False,
    n_ancs=None,
    grid_size=None,
    nms=False,
):
    pred_mask = pred > thresh
    img = np.copy(img)

    mapped_bboxes = apply_offsets(ancs, offsets)
    ancs = ccwh_to_xyxy(ancs)
    mapped_bboxes = ccwh_to_xyxy(mapped_bboxes)

    num_colors = np.maximum(
        np.sum(pred_mask), np.max(anc_mapping) if anc_mapping is not None else 0
    )
    colors = np.array(
        [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ]
    )
    interpolated = []
    for i in range(num_colors + 1):
        i = i / num_colors * (len(colors) - 1)
        a = colors[math.floor(i)]
        b = colors[math.ceil(i)]

        alpha = i - math.floor(i)

        color = alpha * a + (1 - alpha) * b
        interpolated.append(color)

    drawn_bboxes = img[None, ...]

    if nms:
        idx = tf.image.non_max_suppression(mapped_bboxes, pred, 30, score_threshold=thresh)
        mapped_bboxes = mapped_bboxes[idx]
        for i, box in enumerate(mapped_bboxes):
            color = interpolated[i]
            if show_offsets:
                drawn_bboxes = tf.image.draw_bounding_boxes(drawn_bboxes, box[None, None], (color,))
    else:
        for i, idx in enumerate(np.where(pred_mask)[0]):
            color = interpolated[anc_mapping[idx]] if anc_mapping is not None else interpolated[i]
            if show_offsets:
                drawn_bboxes = tf.image.draw_bounding_boxes(
                    drawn_bboxes, mapped_bboxes[idx][None, None], (color,)
                )

            if show_ancs:
                drawn_bboxes = tf.image.draw_bounding_boxes(
                    drawn_bboxes, ancs[idx][None, None], (color,)
                )

    drawn_bboxes = drawn_bboxes

    if show_heatmap:
        pred_img = pred.reshape((*grid_size, n_ancs))
        pred_img = np.max(pred_img, axis=-1)
        pred_img = tf.image.resize(pred_img[..., None], img.shape[:2])[:, :, 0]
        alpha = 0.5
        drawn_bboxes = np.array(drawn_bboxes)
        drawn_bboxes[..., 2] = drawn_bboxes[..., 2] * (1 - alpha) + pred_img * alpha

    return drawn_bboxes[0]
