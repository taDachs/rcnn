import tensorflow as tf
from typing import Sequence

from rcnn.util import apply_offsets, ccwh_to_xyxy


class Rpn(tf.keras.Model):
    def __init__(
        self,
        anc_sizes: Sequence[int],
        anc_ratios: Sequence[float],
        l2: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anc_sizes = anc_sizes
        self.anc_ratios = anc_ratios
        self.num_ancs = len(anc_ratios) * len(anc_sizes)
        self.l2 = l2

        self.training_max_proposals_pre_nms = 12000
        self.training_max_proposals_post_nms = 2000
        self.inference_max_proposals_pre_nms = 6000
        self.inference_max_proposals_post_nms = 300

        regularizer = tf.keras.regularizers.l2(self.l2)
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
            max_pre_nms = self.training_max_proposals_pre_nms
            max_post_nms = self.training_max_proposals_post_nms
        else:
            max_pre_nms = self.inference_max_proposals_pre_nms
            max_post_nms = self.inference_max_proposals_post_nms

        grid_w, grid_h = tf.shape(cls_pred)[1], tf.shape(cls_pred)[2]

        cls_pred = tf.reshape(cls_pred, (-1,))
        reg_pred = tf.reshape(reg_pred, (-1, 4))
        ancs = tf.reshape(ancs, (-1, 4))
        ancs_valid = tf.reshape(ancs_valid, (-1,))

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
        scale_factor = 1
        sigma = 3.0
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
