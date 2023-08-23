import tensorflow_datasets as tfds  # type: ignore
import tensorflow as tf

from typing import Sequence, Optional, Tuple



def generate_mnist(mnist_imgs, size: int, img: tf.Tensor) -> Tuple:
    img_w, img_h = tf.shape(img)[0], tf.shape(img)[1]
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
    size_mnist: int,
    num_mnist: int,
    img_size: Tuple[int, int],
    n: Optional[int] = None,
):
    img = tf.zeros((*img_size, 3))
    (ds, _), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds = ds.batch(num_mnist)

    def f(x, y):
        mnist_img, bboxes = generate_mnist(x, size_mnist, img)
        return mnist_img, bboxes, y + 1

    ds = ds.map(f)
    if n is not None:
        ds = ds.take(n)
    # ds = ds.map(g)
    ds = ds.batch(1)
    return ds, ds, ds_info.features["label"].names






def kitti(
    img_scale_to: int = 500,
):
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "kitti",
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
        bbox = tf.stack((1 - bbox[..., 2], bbox[..., 1], 1 - bbox[..., 0], bbox[..., 3]), axis=-1)
        labels = sample["objects"]["type"]
        return img, bbox, labels + 1

    ds_train = ds_train.shuffle(6000)
    ds_train = ds_train.map(f)
    ds_train = ds_train.batch(1)

    ds_test = ds_test.map(f)
    ds_test = ds_test.batch(1)

    return ds_train, ds_test, ds_info.features["objects"]["type"].names


def pascal_voc(
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

    ds_test = ds_test.map(f)
    ds_test = ds_test.batch(1)
    return ds_train, ds_test, ds_info.features["labels"].names
