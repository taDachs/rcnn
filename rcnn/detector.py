import tensorflow as tf



class DetectionNetwork(tf.keras.Model):
    def __init__(self, roi_size: int, num_classes: int, l2: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.l2 = l2
        regularizer = tf.keras.regularizers.l2(self.l2)
        class_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

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
        scale_factor = 1
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
