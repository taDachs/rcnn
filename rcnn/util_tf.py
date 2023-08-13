import math

import tensorflow as tf  # type: ignore


def xyxy_to_ccwh(bboxes):
    x1 = bboxes[..., 0]
    y1 = bboxes[..., 1]
    x2 = bboxes[..., 2]
    y2 = bboxes[..., 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return tf.stack((cx, cy, w, h), axis=-1)


def ccwh_to_xyxy(bboxes):
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return tf.stack((x1, y1, x2, y2), axis=-1)


def rel_to_grid(bboxes, grid_size):
    w, h = grid_size
    bboxes = bboxes * (w, h, w, h)
    return bboxes


def grid_to_rel(bboxes, grid_size):
    w, h = grid_size
    bboxes = bboxes / (w, h, w, h)
    return bboxes


def calculate_iou_array(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Intersection over Union (IoU) between arrays of bounding boxes.

    Args:
        boxes1 (tf.Tensor): Array of bounding box coordinates (N x 4), where N is the number of boxes.
        boxes2 (tf.Tensor): Array of bounding box coordinates (M x 4), where M is the number of boxes.

    Returns:
        iou_matrix (tf.Tensor): Matrix of IoU values (N x M).
    """
    x1_min = tf.maximum(boxes1[:, 0][:, None], boxes2[:, 0])
    y1_min = tf.maximum(boxes1[:, 1][:, None], boxes2[:, 1])
    x2_max = tf.minimum(boxes1[:, 2][:, None], boxes2[:, 2])
    y2_max = tf.minimum(boxes1[:, 3][:, None], boxes2[:, 3])

    intersection_area = tf.maximum(0.0, x2_max - x1_min) * tf.maximum(0.0, y2_max - y1_min)

    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = box1_area[:, None] + box2_area - intersection_area

    iou_matrix = intersection_area / union_area
    # Assuming iou_matrix and union_area are TensorFlow tensors
    mask = union_area <= 0.0
    indices = tf.where(mask)
    updates = tf.zeros_like(indices, dtype=tf.float32)[:, 0]
    iou_matrix = tf.tensor_scatter_nd_update(iou_matrix, indices, updates)

    return iou_matrix


# ---
def calculate_offsets(anc: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
    x_a, y_a, w_a, h_a = tf.unstack(tf.transpose(anc))
    x, y, w, h = tf.unstack(tf.transpose(gt))

    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = tf.math.log(w / w_a)
    t_h = tf.math.log(h / h_a)

    return tf.stack((t_x, t_y, t_w, t_h), axis=-1)


def apply_offsets(anc: tf.Tensor, offset: tf.Tensor) -> tf.Tensor:
    x_a, y_a, w_a, h_a = tf.unstack(tf.transpose(anc))
    t_x, t_y, t_w, t_h = tf.unstack(tf.transpose(offset))
    x = t_x * w_a + x_a
    y = t_y * h_a + y_a
    w = tf.math.exp(t_w) * w_a
    h = tf.math.exp(t_h) * h_a

    return tf.stack((x, y, w, h), axis=-1)


def visualize_labels(
    img,
    ancs,
    gt_offsets=None,
    pred_offsets=None,
    gt_label=None,
    pred_label=None,
    show_ancs=False,
    show_heatmap=False,
    stride=None,
    n_ancs=None,
    thresh=0.5,
    nms=False,
):
    gt_color = (0.0, 1.0, 0.0)
    pred_color = (1.0, 0.0, 0.0)
    anc_color = (0.0, 0.0, 1.0)

    xy_ancs = ccwh_to_xyxy(ancs)

    drawn_bboxes = img[None, ...]

    if gt_offsets is not None and gt_label is not None:
        gt_bboxes = apply_offsets(ancs, gt_offsets)
        gt_bboxes = ccwh_to_xyxy(gt_bboxes)
        drawn_bboxes = tf.image.draw_bounding_boxes(
            drawn_bboxes, gt_bboxes[gt_label > thresh][None, ...], (gt_color,)
        )
    if show_ancs and gt_label is not None:
        drawn_bboxes = tf.image.draw_bounding_boxes(
            drawn_bboxes, xy_ancs[gt_label > thresh][None, ...], (anc_color,)
        )

    if pred_offsets is not None and pred_label is not None:
        pred_bboxes = apply_offsets(ancs, pred_offsets)
        pred_bboxes = ccwh_to_xyxy(pred_bboxes)
        drawn_bboxes = tf.image.draw_bounding_boxes(
            drawn_bboxes, pred_bboxes[pred_label > thresh][None, ...], (pred_color,)
        )
    if show_ancs and pred_label is not None:
        drawn_bboxes = tf.image.draw_bounding_boxes(
            drawn_bboxes, xy_ancs[pred_label > thresh][None, ...], (anc_color,)
        )

    if show_heatmap:
        grid_size = (tf.shape(img)[0] // stride, tf.shape(img)[1] // stride)
        if pred_label is not None:
            pred_label_img = tf.reshape(pred_label, (*grid_size, n_ancs))
            pred_label_img = tf.reduce_max(pred_label_img, axis=-1)
            pred_label_img = tf.image.resize(pred_label_img[..., None], img.shape[:2])[:, :, 0]
            alpha = 0.5
            drawn_bboxes = tf.concat(
                [
                    drawn_bboxes[..., :2],
                    drawn_bboxes[..., 2:3] * (1.0 - alpha)
                    + pred_label_img[..., tf.newaxis] * alpha,
                    drawn_bboxes[..., 3:],
                ],
                axis=-1,
            )
        if gt_label is not None:
            gt_label_img = tf.reshape(gt_label, (*grid_size, n_ancs))
            gt_label_img = tf.reduce_max(gt_label_img, axis=-1)
            gt_label_img = tf.image.resize(gt_label_img[..., None], img.shape[:2])[:, :, 0]
            alpha = 0.5
            drawn_bboxes = tf.concat(
                [
                    drawn_bboxes[..., :2],
                    drawn_bboxes[..., 2:3] * (1.0 - alpha) + gt_label_img[..., tf.newaxis] * alpha,
                    drawn_bboxes[..., 3:],
                ],
                axis=-1,
            )

    return drawn_bboxes[0]


def tf_random_choice(t: tf.Tensor, n: int):
    idx = tf.range(0, tf.shape(t)[0])
    idx = tf.random.shuffle(idx)[:n]
    return tf.gather(t, idx)
