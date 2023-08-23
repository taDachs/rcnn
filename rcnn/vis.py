import tensorflow as tf
import cv2
import matplotlib.pyplot as plt  # type: ignore
from typing import Mapping

from rcnn.anchors import generate_anchor_map

from rcnn.util import apply_offsets, ccwh_to_xyxy


def visualize_dataset(ds: tf.data.Dataset, mapping: Mapping[int, str]):
    for x in ds.as_numpy_iterator():
        img = x[0][0]
        gt_bboxes = x[1][0]
        gt_labels = x[2][0]

        img = tf.image.draw_bounding_boxes(
            img[None, ...], gt_bboxes[None, ...], ((0, 1, 0),)
        )[0]

        img = img.numpy()
        w, h = img.shape[:2]
        for box, label in zip(gt_bboxes, gt_labels):
            pos = (int(box[1] * h), int(box[0] * w + 16))
            img = cv2.putText(
                img,
                mapping[label - 1],
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 1, 0),
                2,
                cv2.LINE_AA,
            )

        plt.imshow(img)
        plt.show()


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


def vis_single_image(img, model, mapping):
    proposals, label_pred, pre_offset = model.predict_on_image(img)

    predicted_img = img[None, ...]
    predicted_img = tf.image.draw_bounding_boxes(
        predicted_img, proposals[None, ...], ((0, 1, 0),)
    )

    predicted_img = predicted_img[0].numpy()
    w, h = predicted_img.shape[:2]

    for box, label in zip(proposals, label_pred):
        pos = (int(box[1] * h), int(box[0] * w + 16))
        predicted_img = cv2.putText(
            predicted_img,
            mapping[label.numpy()],
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 1, 0),
            2,
            cv2.LINE_AA,
        )

    fig, axs = plt.subplots(1, squeeze=False)
    plt.axis("off")
    plt.imshow(predicted_img)
    plt.show()


def visualize_model_output(ds, model, mapping):
    it = ds.as_numpy_iterator()
    for x in it:
        img = x[0][0]
        gt_bboxes = x[1][0]
        gt_labels = x[2][0]

        proposals, label_pred, pre_offset = model.predict_on_image(img)

        predicted_img = img[None, ...]
        # predicted_img = tf.image.draw_bounding_boxes(
        #     predicted_img, gt_bboxes[None, ...], ((1, 0, 0),)
        # )
        predicted_img = tf.image.draw_bounding_boxes(
            predicted_img, proposals[None, ...], ((0, 1, 0),)
        )
        # predicted_img = tf.image.draw_bounding_boxes(
        #     predicted_img, pre_offset[None, ...], ((0, 0, 1),)
        # )

        predicted_img = predicted_img[0].numpy()
        w, h = predicted_img.shape[:2]
        # for box, label in zip(gt_bboxes, gt_labels):
        #     pos = (int(box[1] * h), int(box[0] * w + 16))
        #     predicted_img = cv2.putText(
        #         predicted_img,
        #         mapping[label - 1],
        #         pos,
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (1, 0, 0),
        #         2,
        #         cv2.LINE_AA,
        #     )

        for box, label in zip(proposals, label_pred):
            pos = (int(box[1] * h), int(box[0] * w + 16))
            predicted_img = cv2.putText(
                predicted_img,
                mapping[label.numpy()],
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 1, 0),
                2,
                cv2.LINE_AA,
            )

        fig, axs = plt.subplots(1, squeeze=False)
        axs[0][0].imshow(predicted_img)
        # axs[1][0].imshow(tf.reduce_max(rpn_map[..., 1], axis=-1))
        # axs[1][1].imshow(tf.reduce_sum(cls[0], axis=-1) / 9, vmin=0.0, vmax=1.0)
        plt.show()


def vis_anc_map(anc_sizes, anc_ratios):
    feat_size = (32, 32, 512)
    img = tf.zeros((32 * 16, 32 * 16, 3))
    feat_map = tf.zeros(feat_size)
    ancs, anc_valid = generate_anchor_map(img, feat_map, anc_sizes, anc_ratios)
    ancs = ancs[16, 16]
    ancs = ccwh_to_xyxy(ancs)

    img = tf.image.draw_bounding_boxes(img[None, ...], ancs[None, ...], ((1, 0, 0), ))

    plt.imshow(img[0])
    plt.show()

