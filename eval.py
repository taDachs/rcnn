#!/usr/bin/env python3
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from rcnn.util_tf import visualize_labels
from rcnn.data_tf import generate_anchor_boxes

IMG_SHAPE = (512, 512, 3)
square_size = 32
grid_size = (32, 32)
anc_size = 4
anc_ratios = (0.5, 1.0, 2.0)
anc_scales = (1, 2, 4)
num_ancs = len(anc_ratios) * len(anc_scales)
ancs = generate_anchor_boxes(grid_size, anc_size, anc_ratios, anc_scales)

model = tf.keras.models.load_model("./rcnn")

# # ---
# img = tf.zeros(IMG_SHAPE)
# img, bboxes = generate_mnist(10, square_size, img)
(ds_train, ds_test), ds_info = tfds.load(
    "voc/2007",
    split=["train", "test"],
    shuffle_files=True,
    with_info=True,
)

it = ds_train.as_numpy_iterator()
for x in it:
    img = tf.image.resize(x["image"] / 255.0, IMG_SHAPE[:2])

    cls_pred, reg_pred = model(img[None, ...])
    reg_pred = reg_pred[0].numpy()
    cls_pred = cls_pred[0].numpy()

    thresh = 0.7


    drawn_bboxes = visualize_labels(
        img,
        ancs,
        reg_pred,
        cls_pred,
        # anc_mapping,
        show_ancs=True,
        show_heatmap=True,
        show_offsets=False,
        n_ancs=num_ancs,
        grid_size=grid_size,
        thresh=thresh,
        nms=False,
    )

    plt.imshow(drawn_bboxes)
    plt.axis("off")
    plt.show()
    cls_map = tf.reshape(tf.cast(cls_pred > thresh, tf.float32), (*grid_size, num_ancs))
    cls_map = tf.reduce_max(cls_map, axis=-1)
    plt.imshow(cls_map)
    plt.axis("off")
    plt.show()
