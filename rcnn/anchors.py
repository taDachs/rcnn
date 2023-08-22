
import tensorflow as tf
from typing import Sequence
from rcnn.util import compute_iou,ccwh_to_xyxy, xyxy_to_ccwh, calculate_offsets

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





def generate_rpn_map(anc_map, anc_valid_mask, gt_bboxes, pos_thresh, neg_thresh):
    grid_shape = tf.shape(anc_valid_mask)
    anc_bboxes = tf.reshape(anc_map, (-1, 4))
    iou = compute_iou(ccwh_to_xyxy(anc_bboxes), gt_bboxes)

    iou *= tf.reshape(anc_valid_mask, (-1,))[..., None]  # remove ancs outside bounds

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

