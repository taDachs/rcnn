from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn.util_tf import (
    calculate_iou_array,
    calculate_offsets,
    ccwh_to_xyxy,
    xyxy_to_ccwh,
    tf_random_choice,
)


# ---
def generate_squares(n: int, size: int, img: tf.Tensor):
    img = tf.constant(img)
    img_w, img_h, _ = img.shape

    x = tf.random.uniform(n, 0, img_w, tf.int32)
    y = tf.random.uniform(n, 0, img_h, tf.int32)
    poses = tf.stack((x, y), axis=-1)
    sizes = tf.fill((n, 2), size)

    squares = tf.concat((poses, sizes), axis=-1)

    colors = tf.constant(
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
        color = colors[tf.random.uniform(1, maxval=len(colors), dtype=tf.int32)]
        img[x : x + w, y : y + h] = color
        bboxes.append((x, y, x + w, y + h))

    bboxes = tf.constant(bboxes) / (img_w, img_h, img_w, img_h)

    return img, bboxes


# ---
def make_batch(img, bboxes, grid_size, ancs, batch_size):
    anc_mapping, offsets, pos_mask, neg_mask = label_img(bboxes, grid_size, ancs)
    xy_ancs = ccwh_to_xyxy(ancs)
    out_of_bounds_mask = tf.reduce_any(xy_ancs < (0, 0, 0, 0), axis=-1) | tf.reduce_any(
        xy_ancs > (1.0, 1.0, 1.0, 1.0), axis=-1
    )
    pos_mask &= ~out_of_bounds_mask
    neg_mask &= ~out_of_bounds_mask

    pos_idx = tf.where(pos_mask)
    neg_idx = tf.where(neg_mask)

    num_pos = tf.cast(
        tf.minimum(batch_size / 2, tf.reduce_sum(tf.cast(pos_mask, tf.float32))), tf.int32
    )
    num_neg = tf.minimum(batch_size - num_pos, tf.reduce_sum(tf.cast(neg_mask, tf.int32)))

    pos_samples = tf_random_choice(pos_idx, num_pos)
    neg_samples = tf_random_choice(neg_idx, num_neg)

    samples = tf.concat((pos_samples, neg_samples), axis=0)
    cls_true = tf.concat((tf.ones_like(pos_samples), tf.zeros_like(neg_samples)), axis=0)
    reg_true = tf.gather(offsets, pos_samples, name="bar")

    pos_samples = tf.reshape(pos_samples, (-1,))
    neg_samples = tf.reshape(neg_samples, (-1,))
    samples = tf.reshape(samples, (-1,))
    cls_true = tf.reshape(cls_true, (-1,))
    reg_true = tf.reshape(reg_true, (-1, 4))

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


def generate_mnist(n: int, size: int, img: tf.Tensor) -> tuple:
    img_w, img_h, _ = img.shape
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    x = tf.random.uniform((n,), 0, img_w - size, dtype=tf.int32)
    y = tf.random.uniform((n,), 0, img_h - size, dtype=tf.int32)

    # Initialize an empty bounding boxes tensor
    bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    for i in tf.range(n):
        x_pos, y_pos = x[i], y[i]
        if (x_pos + size) < img_w and (y_pos + size) < img_h:
            mnist, _ = next(ds_train_it)  # Assuming ds_train_it yields images and labels
            mnist_resized = tf.image.resize(mnist, (size, size))

            mask = mnist_resized[..., 0] > 0
            mask_3d = tf.repeat(mask[..., tf.newaxis], 3, axis=-1)
            mnist_resized_colored = tf.where(mask_3d, mnist_resized, 0)

            slice_img = img[x_pos: x_pos + size, y_pos: y_pos + size]
            updated_slice = slice_img * (1 - tf.cast(mask_3d, tf.float32)) + mnist_resized_colored

            img = img.numpy() # Convert to numpy for direct slicing
            img[x_pos: x_pos + size, y_pos: y_pos + size] = updated_slice
            img = tf.convert_to_tensor(img) # Convert back to tensor

            bbox = tf.convert_to_tensor([x_pos, y_pos, x_pos + size, y_pos + size], dtype=tf.float32)
            bboxes = bboxes.write(i, bbox)

    bboxes_stacked = bboxes.stack()

    # Normalize the bounding boxes
    shape_tensor = tf.constant([img_w, img_h, img_w, img_h], dtype=tf.float32)
    bboxes_normalized = bboxes_stacked / shape_tensor

    return img, bboxes_normalized






def mnist_generator(n: int, size: int, img: tf.Tensor, grid_size, ancs, batch_size):
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

    def g(img, pos_samples, neg_samples, samples, cls_true, reg_true):
        return img, (pos_samples, neg_samples, samples, cls_true, reg_true)

    ds = ds.map(g)
    ds = ds.batch(1)
    return ds


# ---


def pascal_voc(input_shape, grid_size, ancs, batch_size):
    (ds_train, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "test"],
        shuffle_files=True,
        with_info=True,
    )

    def f(sample):
        img = tf.image.resize(tf.cast(sample["image"], tf.float32) / 255.0, input_shape[:2])
        bbox = sample["objects"]["bbox"]
        return img, bbox

    def g(img, bbox):
        img, pos_samples, neg_samples, samples, cls_true, reg_true = make_batch(
            img, bbox, grid_size, ancs, batch_size
        )
        return img, (pos_samples, neg_samples, samples, cls_true, reg_true)

    ds_train = ds_train.map(f)
    ds_train = ds_train.map(g)
    ds_train = ds_train.batch(1)

    # def gen():
    #     for sample in ds_train.as_numpy_iterator():
    #         img = tf.image.resize(sample["image"] / 255.0, input_shape[:2])
    #         bbox = sample["objects"]["bbox"]
    #         yield make_batch(img, bbox, grid_size, ancs, batch_size)
    #
    # return gen_to_dataset(gen)
    return ds_train


# ---
def generate_anchor_boxes(
    grid_size: Tuple[int, int],
    base_size: int,
    aspect_ratios,
    scales,
) -> tf.Tensor:
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

    return tf.constant(anchor_boxes)


def label_img(
    bboxes: tf.Tensor,
    grid_size: Tuple[int, int],
    ancs: tf.Tensor,
    pos_thresh: float = 0.5,
    neg_thresh=0.3,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    ancs = ccwh_to_xyxy(ancs)
    A = tf.shape(ancs)[0]
    B = tf.shape(bboxes)[0]
    anc_mapping = tf.zeros(A, dtype=tf.int32)  # (A, ) maps each anchor to a gt bbox
    iou = calculate_iou_array(ancs, bboxes)  # (A, B)
    # map each gt bbox to a anchor
    max_iou_for_gt_idx = tf.argmax(iou, axis=0)  # (B, )

    anc_mapping = tf.tensor_scatter_nd_update(
        anc_mapping, tf.expand_dims(max_iou_for_gt_idx, axis=-1), tf.range(B)
    )
    pre_mapped_mask = tf.zeros(A, dtype=tf.bool)

    indices = tf.reshape(max_iou_for_gt_idx, [-1, 1])
    updates = tf.ones_like(max_iou_for_gt_idx, dtype=tf.bool)
    pre_mapped_mask = tf.tensor_scatter_nd_update(pre_mapped_mask, indices, updates)

    # map anc boxes with iou > pos_thesh to gt bbox
    iou_thresh_mask = tf.reduce_max(iou, axis=1) > pos_thresh  # (A, )
    iou_thresh_mask &= ~pre_mapped_mask
    max_iou_for_anc_idx = tf.argmax(iou, axis=1)  # (A, )

    indices = tf.where(iou_thresh_mask)
    updates = tf.gather_nd(max_iou_for_anc_idx, indices, name="foo_nd")
    anc_mapping = tf.tensor_scatter_nd_update(anc_mapping, indices, tf.cast(updates, tf.int32))

    pos_mask = pre_mapped_mask | iou_thresh_mask
    neg_mask = tf.reduce_max(iou, axis=1) <= neg_thresh
    neg_mask &= ~pos_mask

    ancs = xyxy_to_ccwh(ancs)
    mapped_bboxes = tf.gather(bboxes, anc_mapping, name="foo")
    mapped_bboxes = xyxy_to_ccwh(mapped_bboxes)
    offsets = calculate_offsets(ancs, mapped_bboxes)
    indices = tf.where(~pos_mask)
    updates = -tf.ones((tf.shape(indices)[0], 4), dtype=tf.float32)
    offsets = tf.tensor_scatter_nd_update(offsets, indices, updates)

    return anc_mapping, offsets, pos_mask, neg_mask
