from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn.util import calculate_iou_array, calculate_offsets, ccwh_to_xyxy, xyxy_to_ccwh


# ---
def generate_squares(n: int, size: int, img: np.ndarray):
    img = np.copy(img)
    img_w, img_h, _ = img.shape

    x = np.random.randint(0, img_w, size=n)
    y = np.random.randint(0, img_h, size=n)
    poses = np.stack((x, y), axis=-1)
    sizes = np.full((n, 2), size)

    squares = np.concatenate((poses, sizes), axis=-1)

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

    bboxes = []
    for x, y, w, h in squares:
        if ((x + w) >= img_w) or ((y + h) >= img_h):
            continue
        color = colors[np.random.randint(len(colors))]
        img[x : x + w, y : y + h] = color
        bboxes.append((x, y, x + w, y + h))

    bboxes = np.array(bboxes) / (img_w, img_h, img_w, img_h)

    return img, bboxes


# ---
def make_batch(img, bboxes, grid_size, ancs, batch_size):
    anc_mapping, offsets, pos_mask, neg_mask = label_img(bboxes, grid_size, ancs)
    xy_ancs = ccwh_to_xyxy(ancs)
    out_of_bounds_mask = np.any(xy_ancs < (0, 0, 0, 0), axis=-1) | np.any(
        xy_ancs > (1.0, 1.0, 1.0, 1.0), axis=-1
    )
    pos_mask &= ~out_of_bounds_mask
    neg_mask &= ~out_of_bounds_mask

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]

    num_pos = np.minimum(batch_size / 2, np.sum(pos_mask)).astype(int)
    num_neg = np.minimum(batch_size - num_pos, np.sum(neg_mask)).astype(int)

    pos_samples = np.random.choice(pos_idx, num_pos, replace=False)
    neg_samples = np.random.choice(neg_idx, num_neg, replace=False)

    samples = np.concatenate((pos_samples, neg_samples))
    cls_true = np.concatenate((np.ones_like(pos_samples), np.zeros_like(neg_samples)))
    reg_true = offsets[pos_samples]

    pos_samples = tf.convert_to_tensor(pos_samples)
    neg_samples = tf.convert_to_tensor(neg_samples)
    samples = tf.convert_to_tensor(samples)
    cls_true = tf.convert_to_tensor(cls_true)
    reg_true = tf.convert_to_tensor(reg_true)

    return img, pos_samples, neg_samples, samples, cls_true, reg_true


# ---
(ds_train_mnist, ds_test_mnist), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train_mnist = ds_train_mnist.repeat()
ds_train_it = ds_train_mnist.as_numpy_iterator()


def generate_mnist(n: int, size: int, img: np.ndarray):
    img = np.copy(img)
    img_w, img_h, _ = img.shape

    x = np.random.randint(0, img_w, size=n)
    y = np.random.randint(0, img_h, size=n)
    poses = np.stack((x, y), axis=-1)
    sizes = np.full((n, 2), size)

    squares = np.concatenate((poses, sizes), axis=-1)

    bboxes = []
    for x, y, w, h in squares:
        if ((x + w) >= img_w) or ((y + h) >= img_h):
            continue
        mnist, label = next(ds_train_it)
        mnist = tf.image.resize(mnist, (w, h))
        rect = img[x : x + w, y : y + h]
        rect[mnist[..., 0] != 0] = mnist[mnist[..., 0] != 0]
        img[x : x + w, y : y + h] = rect
        bboxes.append((x, y, x + w, y + h))

    bboxes = np.array(bboxes) / (img_w, img_h, img_w, img_h)

    return img, bboxes


def mnist_generator(n: int, size: int, img: np.ndarray, grid_size, ancs, batch_size):
    while True:
        sample, bboxes = generate_mnist(n, size, img)
        yield make_batch(sample, bboxes, grid_size, ancs, batch_size)


def gen_to_dataset(gen, n=None):
    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32),
        output_shapes=((None, None, 3), (None,), (None,), (None,), (None,), (None, 4)),
    )

    if n is not None:
        ds = ds.take(n)

    return ds


# ---

def pascal_voc(input_shape, grid_size, ancs, batch_size):
    (ds_train, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "test"],
        shuffle_files=True,
        with_info=True,
    )

    def gen():
        for sample in ds_train.as_numpy_iterator():
            img = tf.image.resize(sample["image"] / 255.0, input_shape[:2])
            bbox = sample["objects"]["bbox"]
            yield make_batch(img, bbox, grid_size, ancs, batch_size)

    return gen_to_dataset(gen)


# ---
def generate_anchor_boxes(
    grid_size: Tuple[int, int],
    base_size: int,
    aspect_ratios,
    scales,
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
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for scale in scales:
                for ratio in aspect_ratios:
                    width = base_size * scale / grid_size[0]
                    height = base_size * scale * ratio / grid_size[1]
                    x_center = (x + 0.5) / grid_size[0]
                    y_center = (y + 0.5) / grid_size[1]

                    anchor_boxes.append((x_center, y_center, width, height))

    return np.array(anchor_boxes)


def label_img(
    bboxes: np.ndarray,
    grid_size: Tuple[int, int],
    ancs: np.ndarray,
    pos_thresh: float = 0.5,
    neg_thresh=0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ancs = ccwh_to_xyxy(ancs)
    A = ancs.shape[0]
    B = bboxes.shape[0]
    anc_mapping = np.full(A, -1, dtype=int)  # (A, ) maps each anchor to a gt bbox
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
    neg_mask = np.max(iou, axis=1) <= neg_thresh
    neg_mask &= ~pos_mask

    ancs = xyxy_to_ccwh(ancs)
    mapped_bboxes = bboxes[anc_mapping]
    mapped_bboxes = xyxy_to_ccwh(mapped_bboxes)
    offsets = calculate_offsets(ancs, mapped_bboxes)
    offsets[~pos_mask] = -1

    return anc_mapping, offsets, pos_mask, neg_mask
