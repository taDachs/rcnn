#!/usr/bin/env python3

import matplotlib.pyplot as plt  # type: ignore
import matplotlib

matplotlib.use("GTK3Agg")  # Or any other X11 back-end
import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore

from rcnn.data_tf import (
    generate_anchor_boxes,
    generate_mnist,
    generate_squares,
    label_img,
    make_batch,
    gen_to_dataset,
    mnist_generator,
    pascal_voc,
)
from rcnn.util_tf import ccwh_to_xyxy, visualize_labels
from rcnn.rpn import Rpn

# ---
# SQUARE_SIZE = 32
ANC_SIZE = 4
ANC_RATIOS = (0.5, 1.0, 2.0)
ANC_SCALES = (1, 2, 4)
NUM_ANCS = len(ANC_RATIOS) * len(ANC_SCALES)
STRIDE = 16

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 20

# ---


def vis_labels():
    # img = tf.zeros(IMG_SHAPE)
    # squares, bboxes = generate_squares(50, square_size, img)
    # img, bboxes = generate_mnist(50, square_size, img)
    # (ds_train, ds_test), ds_info = tfds.load(
    #     "voc/2007",
    #     split=["train", "test"],
    #     shuffle_files=True,
    #     with_info=True,
    # )
    ds_train = pascal_voc(STRIDE, BATCH_SIZE, ANC_SIZE, ANC_RATIOS, ANC_SCALES)

    it = ds_train.as_numpy_iterator()
    for img, y in it:
        (
            ancs,
            pos_samples,
            neg_samples,
            samples,
            cls_true,
            reg_true,
            anc_mapping,
            offsets,
            pos_mask,
            neg_mask,
        ) = y
        # print(tf.reduce_sum(tf.cast(pos_mask, tf.float32)))

        drawn_bboxes = visualize_labels(
            img[0],
            ancs[0],
            gt_offsets=offsets[0],
            gt_label=tf.cast(pos_mask[0], tf.float32),
            show_ancs=True,
            show_heatmap=True,
            stride=STRIDE,
            n_ancs=NUM_ANCS,
        )

        plt.imshow(drawn_bboxes)
        plt.axis("off")
        plt.show()


def visualize_results(model):
    # # ---
    # img = tf.zeros(IMG_SHAPE)
    # img, bboxes = generate_mnist(10, square_size, img)
    (ds_train, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "test"],
        shuffle_files=True,
        with_info=True,
    )

    x = next(ds_test.as_numpy_iterator())
    # img = tf.image.resize(x["image"] / 255.0, IMG_SHAPE[:2])
    img = x["image"] / 255.0

    cls_pred, reg_pred = model(img[None, ...])
    cls_pred = cls_pred[0].numpy()
    reg_pred = reg_pred[0].numpy()
    ancs = generate_anchor_boxes(img, STRIDE, ANC_SIZE, ANC_RATIOS, ANC_SCALES)

    drawn_bboxes = visualize_labels(
        img,
        ancs,
        reg_pred,
        cls_pred,
        # anc_mapping,
        show_ancs=False,
        show_heatmap=True,
        show_offsets=True,
        n_ancs=NUM_ANCS,
        stride=STRIDE,
        thresh=0.6,
        nms=True,
    )

    plt.imshow(drawn_bboxes)
    plt.axis("off")
    plt.show()


# ---
# ---


def train_model():
    # ---
    model = Rpn(STRIDE, NUM_ANCS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer)
    model(tf.zeros((1, 256, 256, 3)))
    model.summary(expand_nested=False)

    # steps = 1000

    # canvas = tf.zeros(IMG_SHAPE)
    # ds_train = gen_to_dataset(
    #     lambda: mnist_generator(15, square_size, canvas, grid_size, ancs, batch_size), n=steps
    # )
    ds_train = pascal_voc(STRIDE, BATCH_SIZE, ANC_SIZE, ANC_RATIOS, ANC_SCALES)

    model.fit(
        ds_train,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr if epoch != 32 else lr * 0.1
            )
        ],
        workers=12,
    )

    model.save("rcnn")

    return model


if __name__ == "__main__":
    # vis_labels()
    model = train_model()
    visualize_results(model)
