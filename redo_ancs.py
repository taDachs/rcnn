#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

from typing import Sequence

from rcnn.util_tf import (
    xyxy_to_ccwh,
    ccwh_to_xyxy,
    calculate_offsets,
    apply_offsets,
    tf_random_choice,
)

ANC_RATIOS = (0.5, 1.0, 2.0)
ANC_SIZES = (128, 256, 512)
IMG_SIZE = 600
BATCH_SIZE = 256

POS_THRESH = 0.5
NEG_THRESH = 0.3
NUM_CLASSES = 21
ROI_SIZE = 7
WEIGHT_DECAY = 5e-4
L2 = 0.5 * WEIGHT_DECAY


def generate_anchor_map(
    img: tf.Tensor, feat_map: tf.Tensor, sizes: Sequence[int], ratios: Sequence[float]
):
    num_ancs = len(sizes) * len(ratios)
    grid_shape = tf.shape(feat_map)[:2]

    strides = tf.shape(img)[:2] / grid_shape
    strides = tf.cast(strides, tf.float32)

    combs = tf.meshgrid(tf.cast(sizes, tf.float32), ratios)
    combs = tf.stack(combs, axis=-1)
    combs = tf.reshape(combs, (-1, 2))

    widths = combs[..., 0] / tf.sqrt(combs[..., 1])
    widths /= strides[0]
    widths = tf.broadcast_to(widths[None, ..., None], (grid_shape[0], grid_shape[1], num_ancs, 1))
    heights = combs[..., 0] * tf.sqrt(combs[..., 1])
    heights /= strides[1]
    heights = tf.broadcast_to(heights[None, ..., None], (grid_shape[0], grid_shape[1], num_ancs, 1))

    grid_coors = tf.meshgrid(tf.range(grid_shape[0]), tf.range(grid_shape[1]), indexing="ij")
    grid_coors = tf.stack(grid_coors, axis=-1)
    grid_coors = tf.cast(grid_coors, tf.float32)
    grid_coors += 0.5
    grid_coors = tf.broadcast_to(
        grid_coors[..., None, :], (grid_shape[0], grid_shape[1], num_ancs, 2)
    )

    anc_map = tf.concat((grid_coors, widths, heights), axis=-1)
    anc_map /= (grid_shape[0], grid_shape[1], grid_shape[0], grid_shape[1])
    xy1 = anc_map[..., :2] - anc_map[..., 2:] * 0.5
    xy2 = anc_map[..., :2] + anc_map[..., 2:] * 0.5
    anc_valid_mask = tf.reduce_all(xy1 >= 0, axis=-1) & tf.reduce_all(xy2 <= 1.0, axis=-1)
    anc_valid_mask = tf.cast(anc_valid_mask, tf.float32)

    return anc_map, anc_valid_mask


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


def generate_rpn_map(anc_map, anc_valid_mask, gt_bboxes, pos_thresh, neg_thresh):
    grid_shape = tf.shape(anc_valid_mask)
    anc_bboxes = tf.reshape(anc_map, (-1, 4))
    iou = compute_iou(ccwh_to_xyxy(anc_bboxes), gt_bboxes)

    # iou *= tf.reshape(anc_valid_mask, (-1,))[..., None]  # remove ancs outside bounds

    max_iou_per_anc = tf.reduce_max(iou, axis=1)
    best_box_idx_per_anc = tf.argmax(iou, axis=1)
    max_iou_per_gt_box = tf.reduce_max(iou, axis=0)
    best_box_per_gt_box_mask = tf.reduce_any(iou == max_iou_per_gt_box, axis=1)

    foreground_mask = max_iou_per_anc > pos_thresh
    foreground_mask |= best_box_per_gt_box_mask
    background_mask = max_iou_per_anc <= neg_thresh
    trainable_mask = (foreground_mask | background_mask) & tf.reshape(anc_valid_mask > 0, (-1,))
    trainable_mask = tf.cast(trainable_mask, tf.float32)

    gt_assignments = best_box_idx_per_anc

    offsets = calculate_offsets(anc_bboxes, xyxy_to_ccwh(tf.gather(gt_bboxes, gt_assignments)))

    rpn_map = tf.concat(
        (
            tf.reshape(trainable_mask, (grid_shape[0], grid_shape[1], grid_shape[2], 1)),
            tf.reshape(
                tf.cast(foreground_mask, tf.float32),
                (grid_shape[0], grid_shape[1], grid_shape[2], 1),
            ),
            tf.reshape(offsets, (grid_shape[0], grid_shape[1], grid_shape[2], 4)),
        ),
        axis=-1,
    )

    return rpn_map


def select_minibatch(rpn_map, batch_size):
    trainable_mask = rpn_map[..., 0]
    background_mask = (rpn_map[..., 1] == 0) & (trainable_mask > 0)
    foreground_mask = (rpn_map[..., 1] == 1) & (trainable_mask > 0)

    pos_idx = tf_random_choice(tf.where(foreground_mask), tf.cast(batch_size // 2, tf.int64))
    num_pos = tf.shape(pos_idx)[0]
    # num_neg = tf.maximum(num_pos, 1)
    num_neg = batch_size - num_pos
    neg_idx = tf_random_choice(tf.where(background_mask), num_neg)

    batch_cls_mask = tf.scatter_nd(
        tf.concat((pos_idx, neg_idx), axis=0),
        tf.ones(num_pos + num_neg),
        tf.shape(foreground_mask, tf.int64),
    )

    batch_reg_mask = tf.scatter_nd(
        pos_idx,
        tf.ones(num_pos),
        tf.shape(foreground_mask, tf.int64),
    )

    return batch_cls_mask, batch_reg_mask


def pascal_voc(
    batch_size: int,
    anc_ratios: Sequence[float],
    anc_scales: Sequence[int],
    img_scale_to: int = 500,
):
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "validation", "test"],
        shuffle_files=False,
        with_info=True,
    )

    ds_train = ds_train.concatenate(ds_val)

    def f(sample):
        img = tf.cast(sample["image"], tf.float32) / 255.0
        img_w = tf.cast(tf.shape(img)[0], tf.float32)
        img_h = tf.cast(tf.shape(img)[1], tf.float32)
        scaling = img_scale_to / tf.minimum(img_w, img_h)

        img = tf.image.resize(
            tf.cast(sample["image"], tf.float32) / 255.0,
            tf.cast((img_w * scaling, img_h * scaling), tf.int32),
        )
        bbox = sample["objects"]["bbox"]
        labels = sample["objects"]["label"]
        return img, bbox, labels + 1

    ds_train = ds_train.shuffle(5000)
    ds_train = ds_train.map(f)
    ds_train = ds_train.batch(1)

    return ds_train, ds_info.features["labels"].names


def draw_anc_map(img, anc_map, anc_valid_mask):
    xy1 = anc_map[..., :2] - anc_map[..., 2:] * 0.5
    xy2 = anc_map[..., :2] + anc_map[..., 2:] * 0.5
    anc_map = tf.concat((xy1, xy2), axis=-1)
    img = img[None, ...]
    img = tf.image.draw_bounding_boxes(
        img, anc_map[anc_valid_mask == 1][None, ...], ((0.0, 1.0, 0.0),)
    )
    img = tf.image.draw_bounding_boxes(
        img, anc_map[anc_valid_mask == 0][None, ...], ((1.0, 0.0, 0.0),)
    )
    return img[0]


def draw_rpn_map(img, anc_map, rpn_map, show_ancs=False):
    ancs = tf.reshape(anc_map, (-1, 4))
    offsets = tf.reshape(rpn_map[..., 2:], (-1, 4))
    bboxes = apply_offsets(ancs, offsets)
    bboxes = ccwh_to_xyxy(bboxes)
    object_mask = tf.reshape(rpn_map[..., 1] > 0, (-1,))
    valid_mask = tf.reshape(rpn_map[..., 0] > 0, (-1,))
    img = img[None, ...]
    img = tf.image.draw_bounding_boxes(
        img, bboxes[object_mask & valid_mask][None, ...], ((0.0, 1.0, 0.0),)
    )
    if show_ancs:
        img = tf.image.draw_bounding_boxes(
            img, ccwh_to_xyxy(ancs[object_mask & valid_mask])[None, ...], ((1.0, 0.0, 0.0),)
        )
    return img[0]


def visualize_model_output(ds, model, mapping):
    it = ds.as_numpy_iterator()
    for x in it:
        img = x[0][0]
        gt_bboxes = x[1][0]
        gt_labels = x[2][0]

        # ancs, anc_valid = generate_anchor_map(img, STRIDE, ANC_SIZES, ANC_RATIOS)
        # rpn_map = generate_rpn_map(ancs, anc_valid, gt_bboxes, POS_THRESH, NEG_THRESH)

        # tf.print(tf.shape(rpn_map))
        # tf.print(tf.shape(ancs))
        # tf.print(tf.shape(img))
        cls_pred, reg_pred, proposals, label_pred, detector_reg = model(img[None, ...])
        # tf.print(tf.shape(cls_pred))
        # exit()
        #     print(tf.shape(batch_cls_mask))
        #     print(tf.shape(rpn_map))
        #     print(tf.shape(cls))
        scores = tf.reduce_max(label_pred, axis=-1)
        label_pred = tf.argmax(label_pred, axis=-1) - 1
        mask = label_pred >= 0
        indices = tf.stack((tf.range(tf.shape(label_pred)[0], dtype=tf.int64), label_pred), axis=-1)
        detector_reg = tf.gather_nd(detector_reg, indices)
        print(tf.shape(proposals))
        print(tf.shape(detector_reg))

        proposals = apply_offsets(xyxy_to_ccwh(proposals), detector_reg)
        proposals = ccwh_to_xyxy(proposals)
        proposals = proposals[mask]
        label_pred = label_pred[mask]
        scores = scores[mask]
        scores = tf.ones_like(scores)

        proposals_idx = tf.image.non_max_suppression(proposals, scores, 20)
        proposals = tf.gather(proposals, proposals_idx)

        # anc_map, anc_valid_mask = generate_anchor_map(img, 16, ANC_SIZES, ANC_RATIOS)
        # rpn_img = draw_rpn_map(img, anc_map, rpn_map, True)
        predicted_img = tf.image.draw_bounding_boxes(
            img[None, ...], proposals[None, ...], ((0, 1, 0),)
        )[0]
        predicted_img = tf.image.draw_bounding_boxes(
            predicted_img[None, ...], gt_bboxes[None, ...], ((1, 0, 0),)
        )[0]
        predicted_img = predicted_img.numpy()
        w, h = predicted_img.shape[:2]
        for box, label in zip(gt_bboxes, gt_labels):
            pos = (int(box[1] * h), int(box[0] * w))
            predicted_img = cv2.putText(
                predicted_img,
                mapping[label - 1],
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (1, 0, 0),
                3,
                cv2.LINE_AA,
            )
        for box, label in zip(proposals, label_pred):
            pos = (int(box[1] * h), int(box[0] * w))
            predicted_img = cv2.putText(
                predicted_img,
                mapping[label.numpy()],
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 1),
                3,
                cv2.LINE_AA,
            )
        #     # batch_cls_mask, batch_reg_mask = select_minibatch(rpn_map, 256)
        #     # print(tf.reduce_sum(batch_reg_mask))
        #     # plt.imshow(tf.cast(tf.reduce_all(valid_anc_mask, axis=-1), tf.float32))
        #     # plt.imshow(mnist_img)
        #     # plt.imshow(anc_img)
        fig, axs = plt.subplots(1, squeeze=False)
        # axs[0].imshow(rpn_img)
        axs[0][0].imshow(predicted_img)
        # axs[1][0].imshow(tf.reduce_max(rpn_map[..., 1], axis=-1))
        # axs[1][1].imshow(tf.reduce_sum(cls[0], axis=-1) / 9, vmin=0.0, vmax=1.0)
        plt.show()


def visualize_minibatch(ds):
    it = ds.as_numpy_iterator()
    for x in it:
        img = x[0][0]
        gt_bboxes = x[1][0]
        labels = x[2][0]
        anc_map, anc_valid_mask = generate_anchor_map(img, 16, ANC_SIZES, ANC_RATIOS)
        rpn_map = generate_rpn_map(anc_map, anc_valid_mask, gt_bboxes, POS_THRESH, NEG_THRESH)
        batch_cls_mask, batch_reg_mask = select_minibatch(rpn_map, BATCH_SIZE)
        img = img[None, ...]
        img = tf.image.draw_bounding_boxes(img, gt_bboxes[None, ...], ((0.0, 1.0, 0.0),))
        img = tf.image.draw_bounding_boxes(
            img, ccwh_to_xyxy(anc_map[rpn_map[..., 1] > 0][None, ...]), ((1.0, 0.0, 0.0),)
        )
        img = img[0].numpy()
        w, h = img.shape[:2]
        for box, label in zip(gt_bboxes, labels):
            pos = (int(box[1] * h), int(box[0] * w))
            img = cv2.putText(
                img, str(label), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 3, cv2.LINE_AA
            )
        print(tf.reduce_sum(batch_reg_mask))
        print(tf.reduce_sum(rpn_map[..., 1]))
        fig, axs = plt.subplots(2)
        axs[0].imshow(img)
        axs[1].imshow(tf.reduce_max(batch_cls_mask, axis=-1))
        plt.show()


def generate_mnist(mnist_imgs, size: int, img: tf.Tensor) -> tuple:
    img_w, img_h, _ = img.shape
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    x = tf.random.uniform((tf.shape(mnist_imgs)[0],), 0, img_w - size, dtype=tf.int32)
    y = tf.random.uniform((tf.shape(mnist_imgs)[0],), 0, img_h - size, dtype=tf.int32)

    # Initialize an empty bounding boxes tensor
    bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    for i in tf.range(tf.shape(mnist_imgs)[0]):
        x_pos, y_pos = x[i], y[i]
        mnist = mnist_imgs[i]
        if (x_pos + size) < img_w and (y_pos + size) < img_h:
            mnist_resized = tf.image.resize(mnist, (size, size))

            mask = mnist_resized[..., 0] > 0
            mask_3d = tf.repeat(mask[..., tf.newaxis], 3, axis=-1)
            mnist_resized_colored = tf.where(mask_3d, mnist_resized, 0)

            slice_img = img[x_pos : x_pos + size, y_pos : y_pos + size]
            updated_slice = slice_img * (1 - tf.cast(mask_3d, tf.float32)) + mnist_resized_colored

            # img = img.numpy()  # Convert to numpy for direct slicing
            # img[x_pos : x_pos + size, y_pos : y_pos + size] = updated_slice
            # img = tf.convert_to_tensor(img)  # Convert back to tensor
            # Define a slice within the tensor img
            # Create a mask of the same shape as img
            mask = tf.pad(
                tf.ones((size, size, 3)),
                [
                    [x_pos, img.shape[0] - (x_pos + size)],
                    [y_pos, img.shape[1] - (y_pos + size)],
                    [0, 0],
                ],
            )

            # Inverse of the mask
            inverse_mask = 1 - mask

            # Multiply the image by the inverse mask to "erase" the portion we want to replace
            erased_img = img * inverse_mask

            # Multiply the updated slice by the mask to keep only the portion we want
            padded_updated_slice = (
                tf.pad(
                    updated_slice,
                    [
                        [x_pos, img.shape[0] - (x_pos + size)],
                        [y_pos, img.shape[1] - (y_pos + size)],
                        [0, 0],
                    ],
                )
                * mask
            )

            # Add the two results together
            img = erased_img + padded_updated_slice

            bbox = tf.convert_to_tensor(
                [x_pos, y_pos, x_pos + size, y_pos + size], dtype=tf.float32
            )
            bboxes = bboxes.write(i, bbox)

    bboxes_stacked = bboxes.stack()

    # Normalize the bounding boxes
    shape_tensor = tf.constant([img_w, img_h, img_w, img_h], dtype=tf.float32)
    bboxes_normalized = bboxes_stacked / shape_tensor

    return img, bboxes_normalized


def mnist_dataset(
    size: int,
    num_mnist: int,
    img: tf.Tensor,
    stride: int,
    batch_size: int,
    anc_ratios: Sequence[float],
    anc_scales: Sequence[int],
    n: int = None,
):
    (ds, _), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds = ds.batch(num_mnist)

    def f(x, y):
        mnist_img, bboxes = generate_mnist(x, size, img)
        return mnist_img, bboxes, y + 1

    # def g(img, bbox, label):
    #     anc_map, anc_valid_mask = generate_anchor_map(img, stride, anc_scales, anc_ratios)
    #     rpn_map = generate_rpn_map(anc_map, anc_valid_mask, bbox, POS_THRESH, NEG_THRESH)
    #     batch_cls_mask, batch_reg_mask = select_minibatch(rpn_map, batch_size)
    #
    #     return img, (rpn_map, batch_cls_mask, batch_reg_mask, label)

    # ds_train = ds_train.shuffle(5000)
    ds = ds.map(f)
    if n is not None:
        ds = ds.take(n)
    # ds = ds.map(g)
    ds = ds.batch(1)
    return ds


class Rpn(tf.keras.Model):
    def __init__(
        self,
        batch_size: int,
        anc_sizes: Sequence[int],
        anc_ratios: Sequence[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.anc_sizes = anc_sizes
        self.anc_ratios = anc_ratios
        self.num_ancs = len(anc_ratios) * len(anc_sizes)

        self.max_proposals_pre_nms = 12000
        self.max_proposals_post_nms = 2000

        regularizer = tf.keras.regularizers.l2(L2)
        initial_weights = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        self.bottleneck = tf.keras.layers.Conv2D(
            512,
            1,
            padding="same",
            activation="relu",
            kernel_initializer=initial_weights,
            kernel_regularizer=regularizer,
        )
        self.cls_out = tf.keras.layers.Conv2D(
            self.num_ancs,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=initial_weights,
        )

        self.reg_out = tf.keras.layers.Conv2D(
            self.num_ancs * 4, 1, padding="same", kernel_initializer=initial_weights
        )

    def call(self, x, training=False):
        feats, ancs, ancs_valid = x[0], x[1], x[2]
        feats = self.bottleneck(feats)
        cls_pred = self.cls_out(feats)
        reg_pred = self.reg_out(feats)
        reg_pred = tf.reshape(
            reg_pred, tf.concat((tf.shape(reg_pred)[:-1], [self.num_ancs, 4]), axis=0)
        )

        proposals = self.extract_proposals(cls_pred, reg_pred, ancs, ancs_valid, training)

        return cls_pred, reg_pred, proposals

    def extract_proposals(self, cls_pred, reg_pred, ancs, ancs_valid, training=False):
        if training:
            max_pre_nms = self.max_proposals_pre_nms
            max_post_nms = self.max_proposals_post_nms
        else:
            max_pre_nms = 6000
            max_post_nms = 300

        grid_w, grid_h = tf.shape(cls_pred)[1], tf.shape(cls_pred)[2]

        cls_pred = tf.reshape(cls_pred, (-1,))
        reg_pred = tf.reshape(reg_pred, (-1, 4))
        ancs = tf.reshape(ancs, (-1, 4))
        ancs_valid = tf.reshape(ancs_valid, (-1,))

        # cls_pred = cls_pred[ancs_valid > 0]
        # ancs = ancs[ancs_valid > 0]
        # reg_pred = ancs[ancs_valid > 0]

        proposals = apply_offsets(ancs, reg_pred)
        proposals = ccwh_to_xyxy(proposals)
        proposals = tf.clip_by_value(proposals, 0, 1)

        proposals_sizes = (proposals[..., 2:] - proposals[..., :2]) * (grid_w, grid_h)
        large_enough_proposals_mask = tf.reduce_all(proposals_sizes >= 1.0, axis=-1)

        cls_pred = cls_pred[large_enough_proposals_mask]
        proposals = proposals[large_enough_proposals_mask]

        sorted_indices = tf.argsort(cls_pred)[::-1]  # descending order
        proposals = tf.gather(proposals, sorted_indices)[:max_pre_nms]
        objectness = tf.gather(cls_pred, sorted_indices)[:max_pre_nms]

        idx = tf.image.non_max_suppression(proposals, objectness, max_post_nms, 0.7)

        proposals = tf.gather(proposals, idx)

        return proposals

    @staticmethod
    def cls_loss(cls_pred, gt_rpn_map, batch_cls_mask):
        loss = tf.reduce_sum(
            tf.losses.binary_crossentropy(
                gt_rpn_map[batch_cls_mask > 0][..., 1, None],
                cls_pred[batch_cls_mask > 0][..., None],
            )
        )
        loss /= tf.maximum(tf.reduce_sum(batch_cls_mask), 1)
        return loss

    @staticmethod
    def reg_loss(reg_pred, gt_rpn_map, batch_reg_mask):
        scale_factor = 2
        sigma = 3.0  # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
        sigma_squared = sigma * sigma
        reg_true = gt_rpn_map[..., 2:][batch_reg_mask > 0]
        reg_pred = reg_pred[batch_reg_mask > 0]

        x = reg_true - reg_pred
        x_abs = tf.abs(x)
        is_negative_branch = tf.stop_gradient(
            tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype=tf.float32)
        )
        R_negative_branch = 0.5 * x * x * sigma_squared
        R_positive_branch = x_abs - 0.5 / sigma_squared
        loss = (
            is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
        )
        loss = tf.reduce_sum(loss)

        loss /= tf.maximum(tf.reduce_sum(batch_reg_mask), 1)
        return loss * scale_factor


class DetectionNetwork(tf.keras.Model):
    def __init__(self, roi_size: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        regularizer = tf.keras.regularizers.l2(L2)
        class_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)
        self.roi_size = roi_size
        self.num_classes = num_classes

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer=regularizer)
        self.fc2 = tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer=regularizer)
        self.cls_out = tf.keras.layers.Dense(
            num_classes, activation="softmax", kernel_initializer=class_initializer
        )
        self.reg_out = tf.keras.layers.Dense(
            (self.num_classes - 1) * 4, kernel_initializer=regressor_initializer
        )

    def roi_pool(self, feats, proposals):
        batch_idx = tf.zeros(tf.shape(proposals)[0], tf.int32)
        rois = tf.image.crop_and_resize(
            feats,
            proposals,
            batch_idx,
            (self.roi_size * 2, self.roi_size * 2),
        )
        pool = tf.nn.max_pool(rois, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return pool

    def call(self, inputs, training=False):
        feats = inputs[0]
        proposals = inputs[1]
        rois = self.roi_pool(feats, proposals)
        flat_rois = self.flatten(rois)
        x = self.fc1(flat_rois)
        x = self.fc2(x)
        cls_out = self.cls_out(x)
        reg_out = self.reg_out(x)
        reg_out = tf.reshape(reg_out, (tf.shape(reg_out)[0], self.num_classes - 1, 4))

        return cls_out, reg_out

    @staticmethod
    def cls_loss(cls_true, cls_pred):
        cls_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(cls_true, cls_pred))
        return cls_loss

    @staticmethod
    def reg_loss(reg_pred, reg_true, labels):
        scale_factor = 2
        mask = tf.argmax(labels, axis=-1) > 0
        sigma = 1.0  # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
        sigma_squared = sigma * sigma
        reg_true = reg_true[mask]
        reg_pred = reg_pred[mask]

        classes = tf.argmax(labels[mask], axis=-1) - 1  # -1 because of background class
        indices = tf.stack((tf.range(tf.shape(classes)[0], dtype=tf.int64), classes), axis=-1)

        reg_pred = tf.gather_nd(reg_pred, indices)

        x = reg_true - reg_pred
        x_abs = tf.abs(x)
        is_negative_branch = tf.stop_gradient(
            tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype=tf.float32)
        )
        R_negative_branch = 0.5 * x * x * sigma_squared
        R_positive_branch = x_abs - 0.5 / sigma_squared
        loss = (
            is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
        )
        loss = tf.reduce_sum(loss)

        loss /= tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1)
        return loss * scale_factor


class FasterRCNN(tf.keras.Model):
    def __init__(
        self,
        batch_size: int,
        anc_sizes: Sequence[int],
        anc_ratios: Sequence[float],
        num_classes: int,
        roi_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(self, *args, **kwargs)
        self.batch_size = batch_size
        self.anc_sizes = anc_sizes
        self.anc_ratios = anc_ratios
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.object_iou_thresh = 0.5
        self.background_iou_thresh = 0.0
        self.detector_batch_size = 256

        self.backbone = self._build_backbone()
        self.rpn = Rpn(batch_size, anc_sizes, anc_ratios)
        self.detector = DetectionNetwork(roi_size, num_classes)
        self.cls_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        self.label_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    def _build_backbone(self) -> tf.keras.Model:
        # backbone = tf.keras.applications.MobileNetV2(include_top=False)
        # backbone = tf.keras.applications.ResNet50(include_top=False)
        backbone = tf.keras.applications.VGG16(include_top=False)
        # feat = backbone.get_layer("block_6_expand_relu").output
        # feat = backbone.get_layer("block_13_expand_relu").output
        # feat = backbone.get_layer("conv4_block6_out").output
        feat = backbone.get_layer("block5_conv3").output
        # feat = backbone.get_layer("conv4_block23_out").output
        return tf.keras.Model(backbone.inputs, feat)

    def _assign_labels_to_proposals(self, proposals, gt_bboxes, gt_labels):
        proposals = tf.concat((proposals, gt_bboxes), axis=0)
        iou = compute_iou(proposals, gt_bboxes)

        best_iou = tf.reduce_max(iou, axis=1)
        best_idx = tf.argmax(iou, axis=1)
        best_class_label = tf.gather(gt_labels, best_idx)
        best_class_boxes = tf.gather(gt_bboxes, best_idx)

        idxs = tf.where(best_iou >= self.background_iou_thresh)[:, 0]
        proposals = tf.gather(proposals, idxs)
        best_ious = tf.gather(best_iou, idxs)
        best_class_label = tf.gather(best_class_label, idxs)
        best_class_boxes = tf.gather(best_class_boxes, idxs)

        retain_mask = tf.cast(best_ious >= self.object_iou_thresh, tf.int64)
        best_class_label = best_class_label * retain_mask

        gt_classes = tf.one_hot(best_class_label, self.num_classes, dtype=tf.float32)
        offsets = calculate_offsets(xyxy_to_ccwh(proposals), xyxy_to_ccwh(best_class_boxes))

        return proposals, offsets, best_class_boxes, gt_classes

    def _select_detector_batch(self, proposals, gt_bboxes, gt_labels, offsets):
        background_mask = tf.argmax(gt_labels, axis=-1) == 0
        foreground_mask = tf.argmax(gt_labels, axis=-1) > 0

        num_pos = tf.minimum(
            tf.reduce_sum(tf.cast(foreground_mask, tf.int64)), self.detector_batch_size // 2
        )
        foreground_idx = tf_random_choice(tf.where(foreground_mask)[:, 0], num_pos)
        # num_neg = tf.maximum(num_pos, 1)
        num_neg = self.detector_batch_size - num_pos
        background_idx = tf_random_choice(tf.where(background_mask)[:, 0], num_neg)

        idx = tf.concat((foreground_idx, background_idx), axis=0)

        return (
            tf.gather(proposals, idx),
            tf.gather(gt_bboxes, idx),
            tf.gather(gt_labels, idx),
            tf.gather(offsets, idx),
        )

    def call(self, x, training=False):
        # img, ancs, anc_valid = x[0], x[1], x[2]
        feats = self.backbone(x)
        ancs, anc_valid = generate_anchor_map(x[0], feats[0], self.anc_sizes, self.anc_ratios)
        cls_pred, reg_pred, proposals = self.rpn((feats, ancs, anc_valid), training=False)
        label_pred, detector_reg = self.detector((feats, proposals))
        return cls_pred, reg_pred, proposals, label_pred, detector_reg

    def train_step(self, data):
        img = data[0]
        gt_bboxes, gt_labels = data[1], data[2]

        with tf.GradientTape() as tape:
            feats = self.backbone(img)
            anc_map, anc_valid = generate_anchor_map(
                img[0], feats[0], self.anc_sizes, self.anc_ratios
            )
            gt_rpn_map = generate_rpn_map(anc_map, anc_valid, gt_bboxes[0], POS_THRESH, NEG_THRESH)
            batch_cls_mask, batch_reg_mask = select_minibatch(gt_rpn_map, self.batch_size)

            cls_pred, reg_pred, proposals = self.rpn((feats, anc_map, anc_valid), training=True)
            cls_pred = cls_pred[0]
            reg_pred = reg_pred[0]
            rpn_cls_loss = Rpn.cls_loss(cls_pred, gt_rpn_map, batch_cls_mask)
            rpn_reg_loss = Rpn.reg_loss(reg_pred, gt_rpn_map, batch_reg_mask)

            proposals = tf.stop_gradient(proposals)

            proposals, offsets, gt_bboxes, gt_labels = self._assign_labels_to_proposals(
                proposals, gt_bboxes[0], gt_labels[0]
            )
            proposals, gt_bboxes, gt_labels, offsets = self._select_detector_batch(
                proposals, gt_bboxes, gt_labels, offsets
            )

            pred_labels, detector_reg = self.detector((feats, proposals))
            detector_label_loss = DetectionNetwork.cls_loss(gt_labels, pred_labels)
            detector_reg_loss = DetectionNetwork.reg_loss(detector_reg, offsets, gt_labels)

            loss = rpn_cls_loss + rpn_reg_loss + detector_label_loss + detector_reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        grads = (tf.clip_by_norm(g, 1) for g in grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        rpn_num_pos = tf.reduce_sum(gt_rpn_map[..., 1][batch_cls_mask > 0])
        rpn_num_neg = tf.reduce_sum(1 - gt_rpn_map[..., 1][batch_cls_mask > 0])
        num_background_rois = tf.reduce_sum(tf.cast(tf.argmax(gt_labels, axis=-1) == 0, tf.float32))
        num_foreground_rois = tf.reduce_sum(tf.cast(tf.argmax(gt_labels, axis=-1) > 0, tf.float32))

        cls_acc = self.cls_accuracy_metric(
            tf.cast(gt_rpn_map[batch_cls_mask > 0][..., 1, None] > 0.5, tf.float32),
            cls_pred[batch_cls_mask > 0][..., None],
        )

        label_acc = self.label_accuracy_metric(gt_labels, pred_labels)

        return {
            "loss": loss,
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "detector_label_loss": detector_label_loss,
            "detector_reg_loss": detector_reg_loss,
            "rpn_num_pos": rpn_num_pos,
            "rpn_num_neg": rpn_num_neg,
            "num_foreground_rois": num_foreground_rois,
            "num_background_rois": num_background_rois,
            "rpn_cls_acc": cls_acc,
            "detector_label_acc": label_acc,
        }


def main():
    matplotlib.use("GTK3Agg")  # Or any other X11 back-end
    model = FasterRCNN(BATCH_SIZE, ANC_SIZES, ANC_RATIOS, NUM_CLASSES, ROI_SIZE)
    ds, label_mapping = pascal_voc(BATCH_SIZE, ANC_RATIOS, ANC_SIZES, IMG_SIZE)
    # canvas = tf.zeros((600, 1000, 3))
    # ds = mnist_dataset(100, 10, canvas, BATCH_SIZE, ANC_RATIOS, ANC_SIZES, 6000)
    model.load_weights("./test_new_anchors/weights")
    visualize_model_output(ds, model, label_mapping)
    # visualize_minibatch(ds)

    callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau("loss", patience=3),
        # tf.keras.callbacks.EarlyStopping("loss", patience=5),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            "./checkpoints/chkpt", "loss", save_best_only=True, save_weights_only=True
        ),
    ]

    # model.compile(tf.keras.optimizers.Adam(1e-3))
    model.compile(tf.keras.optimizers.SGD(1e-3, momentum=0.9, weight_decay=WEIGHT_DECAY))
    # model.compile(tf.keras.optimizers.SGD(1e-3))
    model.fit(ds, epochs=10, workers=14, callbacks=callbacks)
    # model.compile(tf.keras.optimizers.Adam(1e-4))
    model.compile(tf.keras.optimizers.SGD(1e-4, momentum=0.9, weight_decay=WEIGHT_DECAY))
    model.fit(ds, epochs=5, workers=14)
    # model.compile(tf.keras.optimizers.SGD(1e-5, momentum=0.0, weight_decay=0.0005))
    # model.fit(ds, epochs=5, workers=14)
    model.save_weights("./test_new_anchors/weights")


if __name__ == "__main__":
    main()
