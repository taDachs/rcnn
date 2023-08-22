import tensorflow as tf  # type: ignore

def compute_iou(bboxes1: tf.Tensor, bboxes2: tf.Tensor) -> tf.Tensor:
    """
    Compute the Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
    - bboxes1 (tf.Tensor): A tensor of shape (N, 4) representing bounding boxes where
      each box is represented as [y_min, x_min, y_max, x_max].
    - bboxes2 (tf.Tensor): A tensor of shape (M, 4) representing bounding boxes.

    Returns:
    - tf.Tensor: A tensor of shape (N, M) where the element at (i, j) is the IoU
      between bboxes1[i] and bboxes2[j].
    """

    # Expand dimensions to make sure we can compute pairwise IoU
    bboxes1 = tf.expand_dims(bboxes1, 1)  # Shape: [N, 1, 4]
    bboxes2 = tf.expand_dims(bboxes2, 0)  # Shape: [1, M, 4]

    # Compute coordinates for the intersections
    y_min_int = tf.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0])
    x_min_int = tf.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1])
    y_max_int = tf.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2])
    x_max_int = tf.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3])

    # Compute areas of intersection
    intersection_area = tf.maximum(0.0, y_max_int - y_min_int) * tf.maximum(
        0.0, x_max_int - x_min_int
    )

    # Compute areas of the bounding boxes
    bboxes1_area = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (bboxes1[:, :, 3] - bboxes1[:, :, 1])
    bboxes2_area = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (bboxes2[:, :, 3] - bboxes2[:, :, 1])

    # Compute areas of union
    union_area = bboxes1_area + bboxes2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def tf_random_choice(t: tf.Tensor, n: int):
    idx = tf.range(0, tf.shape(t)[0])
    idx = tf.random.shuffle(idx)[:n]
    return tf.gather(t, idx)

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


def calculate_offsets(anc: tf.Tensor, gt: tf.Tensor, mean=None, std=None) -> tf.Tensor:
    x_a, y_a, w_a, h_a = tf.unstack(tf.transpose(anc))
    x, y, w, h = tf.unstack(tf.transpose(gt))

    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = tf.math.log(w / w_a)
    t_h = tf.math.log(h / h_a)

    offset = tf.stack((t_x, t_y, t_w, t_h), axis=-1)

    if (mean is not None) and (std is not None):
        offset = (offset - mean) / std

    return offset


def apply_offsets(anc: tf.Tensor, offset: tf.Tensor, mean=None, std=None) -> tf.Tensor:
    x_a, y_a, w_a, h_a = tf.unstack(tf.transpose(anc))
    t_x, t_y, t_w, t_h = tf.unstack(tf.transpose(offset))
    x = t_x * w_a + x_a
    y = t_y * h_a + y_a
    w = tf.math.exp(t_w) * w_a
    h = tf.math.exp(t_h) * h_a

    bboxes = tf.stack((x, y, w, h), axis=-1)

    if (mean is not None) and (std is not None):
        offset = offset * std + mean

    return bboxes
