#!/usr/bin/env python3

# ---
import math
import random
from typing import Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds

from rcnn.data import (
    generate_anchor_boxes,
    generate_mnist,
    generate_squares,
    label_img,
    make_batch,
    gen_to_dataset,
    mnist_generator,
    pascal_voc
)
from rcnn.util import ccwh_to_xyxy, visualize_labels

# ---
IMG_SHAPE = (256, 256, 3)
square_size = 16
grid_size = (32, 32)
anc_size = 6
anc_ratios = (0.5, 1.0, 2.0)
anc_scales = (1, 2, 3)
num_ancs = len(anc_ratios) * len(anc_scales)


# ---
# img = np.zeros(IMG_SHAPE)
# squares, bboxes = generate_squares(50, square_size, img)
# img, bboxes = generate_mnist(50, square_size, img)
ancs = generate_anchor_boxes(grid_size, anc_size, anc_ratios, anc_scales)

(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
)

x = next(ds_train.as_numpy_iterator())
img = x["image"] / 255.0
bboxes = x["objects"]["bbox"]

anc_mapping, offsets, pos_mask, neg_mask = label_img(bboxes, grid_size, ancs)

drawn_bboxes = visualize_labels(
    img,
    ancs,
    offsets,
    pos_mask.astype(np.float32),
    anc_mapping,
    show_heatmap=True,
    n_ancs=num_ancs,
    show_ancs=True,
    show_offsets=True,
    grid_size=grid_size,
)

plt.imshow(drawn_bboxes)
plt.axis("off")
plt.show()


# ---
input_shape = IMG_SHAPE
learning_rate = 1e-3

# ---

inputs = tf.keras.Input(input_shape)
x = inputs
x = tf.keras.layers.Conv2D(16, 3, strides=1, padding="same")(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same")(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same")(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.ReLU()(x)
feat = x
cls_out = tf.keras.layers.Conv2D(num_ancs, 1, padding="same")(feat)
cls_out = tf.keras.layers.Flatten()(cls_out)

reg_out = tf.keras.layers.Conv2D(num_ancs * 4, 1, padding="same")(feat)
reg_out = tf.keras.layers.Reshape((-1, 4))(reg_out)

model = tf.keras.Model(inputs, [cls_out, reg_out])
model.compile()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.summary()

# ---
ancs = generate_anchor_boxes(grid_size, anc_size, anc_ratios, anc_scales)
xy_ancs = ccwh_to_xyxy(ancs)
out_of_bounds_mask = np.any(xy_ancs < (0, 0, 0, 0), axis=-1) | np.any(
    xy_ancs > (1.0, 1.0, 1.0, 1.0), axis=-1
)

steps = 1000
batch_size = 64

num_pos_tot = 0
num_neg_tot = 0

canvas = np.zeros(IMG_SHAPE)
# ds_train = gen_to_dataset(
#     lambda: mnist_generator(15, square_size, canvas, grid_size, ancs, batch_size), n=steps
# )
ds_train = pascal_voc(IMG_SHAPE, grid_size, ancs, batch_size)
epochs = 100

for epoch in range(epochs):
    print(f"### EPOCH {epoch} ###")
    for step, (img, pos_samples, neg_samples, samples, cls_true, reg_true) in enumerate(ds_train):
        # num_objects = random.randint(10, 20)
        # # img, bboxes = generate_squares(num_squares, square_size, canvas)
        # img, bboxes = generate_mnist(num_objects, square_size, canvas)
        # img, pos_samples, neg_samples, samples, cls_true, reg_true = make_batch(
        #     img, bboxes, grid_size, ancs, batch_size
        # )

        with tf.GradientTape() as tape:
            cls_pred, reg_pred = model(img[None, ...], training=True)
            cls_pred = tf.gather(cls_pred[0], samples)
            reg_pred = tf.gather(reg_pred[0], pos_samples)

            cls_loss = tf.losses.binary_crossentropy(cls_true, cls_pred, from_logits=True)
            cls_loss = tf.reduce_sum(cls_loss)
            cls_loss *= 1 / tf.cast(tf.shape(samples)[0], tf.float32)
            reg_loss = tf.reduce_sum(tf.losses.huber(reg_true, reg_pred, delta=0.3))
            reg_loss *= 10 / (grid_size[0] * grid_size[1] * num_ancs)
            loss_value = cls_loss + reg_loss

        acc = tf.keras.metrics.BinaryAccuracy()(cls_true, cls_pred)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 100 == 0:
            print(
                "Training loss (for one batch) at step %d: %.8f, cls_closs: %.8f, reg_loss: %.8f, accuracy: %f, learning_rate: %f"
                % (step, float(loss_value), float(cls_loss), float(reg_loss), acc, learning_rate)
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
        # if step % 3000 == 0 and step != 0:
        #     learning_rate = learning_rate * 0.1
        #     optimizer.learning_rate = learning_rate

print("########### TRAINING FINISHED ###########")

# ---
# img = np.zeros(IMG_SHAPE)
# img, bboxes = generate_mnist(10, square_size, img)
(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
)

x = next(ds_test.as_numpy_iterator())
img = tf.image.resize(x["image"] / 255.0, IMG_SHAPE[:2])

cls_pred, reg_pred = model(img[None, ...])
cls_pred = tf.sigmoid(cls_pred)[0].numpy()
reg_pred = reg_pred[0].numpy()

drawn_bboxes = visualize_labels(
    img,
    ancs,
    reg_pred,
    cls_pred,
    # anc_mapping,
    show_ancs=False,
    show_heatmap=True,
    show_offsets=False,
    n_ancs=num_ancs,
    grid_size=grid_size,
    thresh=0.6,
    nms=True,
)

plt.imshow(drawn_bboxes)
plt.axis("off")
plt.show()
