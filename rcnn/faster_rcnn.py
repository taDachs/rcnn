from typing import Sequence
import tensorflow as tf
from rcnn.rpn import Rpn
from rcnn.detector import DetectionNetwork

from rcnn.util import (
    compute_iou,
    calculate_offsets,
    xyxy_to_ccwh,
    tf_random_choice,
    apply_offsets,
    ccwh_to_xyxy,
)
from rcnn.anchors import generate_anchor_map, generate_rpn_map


class FasterRCNN(tf.keras.Model):
    def __init__(
        self,
        anc_sizes: Sequence[int],
        anc_ratios: Sequence[float],
        num_classes: int,
        roi_size: int,
        l2: float = 0,
        rpn_foreground_iou_thresh: float = 0.5,
        rpn_background_iou_thresh: float = 0.3,
        detector_foreground_iou_thresh: float = 0.5,
        detector_background_iou_thresh: float = 0.0,
        rpn_batch_size: int = 256,
        detector_batch_size: int = 128,
        backbone_type: str = "vgg",
        *args,
        **kwargs,
    ):
        super().__init__(self, *args, **kwargs)
        self.rpn_batch_size = rpn_batch_size
        self.detector_batch_size = detector_batch_size
        self.anc_sizes = anc_sizes
        self.anc_ratios = anc_ratios
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.rpn_foreground_iou_thresh = rpn_foreground_iou_thresh
        self.rpn_background_iou_thresh = rpn_background_iou_thresh
        self.detector_foreground_iou_thresh = detector_foreground_iou_thresh
        self.detector_background_iou_thresh = detector_background_iou_thresh
        self.l2 = l2
        self.backbone_type = backbone_type

        self.backbone = self._build_backbone(self.backbone_type)
        self.rpn = Rpn(anc_sizes, anc_ratios, self.l2)
        self.detector = DetectionNetwork(roi_size, num_classes, self.l2)
        self.cls_accuracy_metric = tf.keras.metrics.BinaryAccuracy()  # type: ignore
        self.label_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()  # type: ignore

    def _build_backbone(self, backbone_type: str) -> tf.keras.Model:
        if backbone_type == "vgg":
            backbone = tf.keras.applications.VGG16(include_top=False)
            feat = backbone.get_layer("block5_conv3").output
        elif backbone_type == "resnet":
            backbone = tf.keras.applications.ResNet50(include_top=False)
            feat = backbone.get_layer("conv4_block6_out").output
        elif backbone_type == "mobilenet":
            backbone = tf.keras.applications.MobileNetV2(include_top=False)
            feat = backbone.get_layer("block_13_expand_relu").output
        return tf.keras.Model(backbone.inputs, feat)

    def _assign_labels_to_proposals(self, proposals, gt_bboxes, gt_labels):
        proposals = tf.concat((proposals, gt_bboxes), axis=0)
        iou = compute_iou(proposals, gt_bboxes)

        best_iou = tf.reduce_max(iou, axis=1)
        best_idx = tf.argmax(iou, axis=1)
        best_class_label = tf.gather(gt_labels, best_idx)
        best_class_boxes = tf.gather(gt_bboxes, best_idx)

        idxs = tf.where(best_iou >= self.detector_background_iou_thresh)[:, 0]
        proposals = tf.gather(proposals, idxs)
        best_ious = tf.gather(best_iou, idxs)
        best_class_label = tf.gather(best_class_label, idxs)
        best_class_boxes = tf.gather(best_class_boxes, idxs)

        retain_mask = tf.cast(best_ious >= self.detector_foreground_iou_thresh, tf.int64)
        best_class_label = best_class_label * retain_mask

        gt_classes = tf.one_hot(best_class_label, self.num_classes, dtype=tf.float32)
        offsets = calculate_offsets(
            xyxy_to_ccwh(proposals),
            xyxy_to_ccwh(best_class_boxes),
        )

        return proposals, offsets, best_class_boxes, gt_classes

    def _select_detector_batch(self, proposals, gt_bboxes, gt_labels, offsets):
        background_mask = tf.argmax(gt_labels, axis=-1) == 0
        foreground_mask = tf.argmax(gt_labels, axis=-1) > 0

        num_pos = tf.minimum(
            tf.reduce_sum(tf.cast(foreground_mask, tf.int64)), self.detector_batch_size // 2
        )
        foreground_idx = tf_random_choice(tf.where(foreground_mask)[:, 0], num_pos)
        num_neg = tf.maximum(num_pos, 1)
        # num_neg = self.detector_batch_size - num_pos
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

    def predict_on_image(self, x):
        feats = self.backbone(x[None, ...])
        ancs, anc_valid = generate_anchor_map(x, feats[0], self.anc_sizes, self.anc_ratios)
        cls_pred, reg_pred, proposals = self.rpn((feats, ancs, anc_valid), training=False)
        label_pred, detector_reg = self.detector((feats, proposals))

        scores = tf.reduce_max(label_pred, axis=-1)
        label_pred = tf.argmax(label_pred, axis=-1) - 1
        mask = label_pred >= 0
        indices = tf.stack((tf.range(tf.shape(label_pred)[0], dtype=tf.int64), label_pred), axis=-1)
        detector_reg = tf.gather_nd(detector_reg, indices)

        before_offset = proposals

        proposals = apply_offsets(xyxy_to_ccwh(proposals), detector_reg)
        proposals = ccwh_to_xyxy(proposals)
        proposals = proposals[mask]
        label_pred = label_pred[mask]
        scores = scores[mask]
        before_offset = before_offset[mask]

        proposals = tf.clip_by_value(proposals, 0.0, 1.0)

        proposals_idx = tf.image.non_max_suppression(
            proposals, scores, tf.shape(proposals)[0], 0.3, 0.7
        )
        proposals = tf.gather(proposals, proposals_idx)
        label_pred = tf.gather(label_pred, proposals_idx)
        # before_offset = tf.gather(before_offset, proposals_idx)

        return proposals, label_pred, before_offset

    def select_minibatch(self, rpn_map):
        trainable_mask = rpn_map[..., 0]
        background_mask = (rpn_map[..., 1] == 0) & (trainable_mask > 0)
        foreground_mask = (rpn_map[..., 1] == 1) & (trainable_mask > 0)

        pos_idx = tf_random_choice(tf.where(foreground_mask), tf.cast(self.rpn_batch_size // 2, tf.int64))
        num_pos = tf.shape(pos_idx)[0]
        num_neg = tf.maximum(num_pos, 1)
        # num_neg = self.rpn_batch_size - num_pos
        neg_idx = tf_random_choice(tf.where(background_mask), num_neg)
        num_pos = tf.shape(neg_idx)[0]

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

    def train_step(self, data):
        img = data[0]
        gt_bboxes, gt_labels = data[1], data[2]

        with tf.GradientTape() as tape:
            feats = self.backbone(img)
            anc_map, anc_valid = generate_anchor_map(
                img[0], feats[0], self.anc_sizes, self.anc_ratios
            )
            gt_rpn_map = generate_rpn_map(
                anc_map,
                anc_valid,
                gt_bboxes[0],
                self.rpn_foreground_iou_thresh,
                self.rpn_background_iou_thresh,
            )
            batch_cls_mask, batch_reg_mask = self.select_minibatch(gt_rpn_map)

            cls_pred, reg_pred, proposals = self.rpn((feats, anc_map, anc_valid), training=True)
            cls_pred = cls_pred[0]
            reg_pred = reg_pred[0]
            rpn_cls_loss = Rpn.cls_loss(cls_pred, gt_rpn_map, batch_cls_mask)
            rpn_reg_loss = Rpn.reg_loss(reg_pred, gt_rpn_map, batch_reg_mask)

            proposals = tf.stop_gradient(proposals)

            proposals, gt_offsets, gt_bboxes, gt_labels = self._assign_labels_to_proposals(
                proposals, gt_bboxes[0], gt_labels[0]
            )
            proposals, gt_bboxes, gt_labels, gt_offsets = self._select_detector_batch(
                proposals, gt_bboxes, gt_labels, gt_offsets
            )

            pred_labels, detector_reg = self.detector((feats, proposals))
            detector_label_loss = DetectionNetwork.cls_loss(gt_labels, pred_labels)
            detector_reg_loss = DetectionNetwork.reg_loss(detector_reg, gt_offsets, gt_labels)

            loss = rpn_cls_loss + rpn_reg_loss + detector_label_loss + detector_reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        # grads = (tf.clip_by_norm(g, 1) for g in grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        rpn_num_pos = tf.reduce_sum(gt_rpn_map[..., 1][batch_cls_mask > 0])
        rpn_num_neg = tf.reduce_sum(1 - gt_rpn_map[..., 1][batch_cls_mask > 0])
        num_background_rois = tf.reduce_sum(
            tf.cast(tf.argmax(gt_labels, axis=-1) == 0, tf.float32)
        )
        num_foreground_rois = tf.reduce_sum(
            tf.cast(tf.argmax(gt_labels, axis=-1) > 0, tf.float32)
        )

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
