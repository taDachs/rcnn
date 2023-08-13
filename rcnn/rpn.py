import tensorflow as tf


class Rpn(tf.keras.Model):
    def __init__(
        self,
        stride: int,
        num_ancs: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.num_ancs = num_ancs
        # backbone = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False)
        # feat = backbone.get_layer("block_13_expand_relu").output
        backbone = tf.keras.applications.ResNet50(include_top=False)
        # backbone = tf.keras.applications.VGG16(input_tensor=inputs, include_top=False)
        # feat = backbone.get_layer("block_13_expand_relu").output
        feat = backbone.get_layer("conv4_block6_out").output
        # feat = backbone.get_layer("conv4_block23_out").output
        # feat = backbone.output
        feat = tf.keras.layers.Conv2D(512, 1, padding="same", activation="relu")(feat)
        cls_out = tf.keras.layers.Conv2D(num_ancs, 1, padding="same", activation="sigmoid")(feat)
        cls_out = tf.keras.layers.Flatten()(cls_out)

        reg_out = tf.keras.layers.Conv2D(num_ancs * 4, 1, padding="same")(feat)
        reg_out = tf.keras.layers.Reshape((-1, 4))(reg_out)

        self.model = tf.keras.Model(backbone.inputs, [cls_out, reg_out])

        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    def call(self, x, training=False):
        return self.model(x)

    def train_step(self, data):
        img, label = data[0], data[1]
        grid_size = tf.cast(
            (tf.shape(img[0])[0] / self.stride, tf.shape(img[0])[1] / self.stride), tf.float32
        )
        with tf.GradientTape() as tape:
            cls_pred, reg_pred = self.model(img)
            cls_pred = cls_pred[0]
            reg_pred = reg_pred[0]

            pos_samples, neg_samples, samples, cls_true, reg_true = (
                label[1][0],
                label[2][0],
                label[3][0],
                label[4][0],
                label[5][0],
            )

            cls_pred = tf.gather(cls_pred, samples)
            reg_pred = tf.gather(reg_pred, pos_samples)

            cls_loss = tf.losses.binary_crossentropy(cls_true, cls_pred, from_logits=True)
            # alpha = 0.3
            # gamma = 2.0
            # cls_loss = tf.math.log(cls_pred)
            # cls_loss = -alpha * ((1.0 - cls_pred) ** gamma) * tf.math.log(cls_pred + 0.0001) * tf.cast(cls_true, tf.float32) + \
            #            -(1.0 - alpha) * cls_pred ** gamma * tf.math.log(1.0 - cls_pred + 0.0001) * (1.0 - tf.cast(cls_true, tf.float32))
            # cls_loss = tf.reduce_mean(cls_loss)
            cls_loss *= 1 / tf.cast(tf.shape(samples)[0], tf.float32)
            reg_loss = tf.reduce_sum(tf.losses.huber(reg_true, reg_pred, delta=0.3))
            reg_loss *= 10 / (grid_size[0] * grid_size[1] * self.num_ancs)
            loss_value = cls_loss + reg_loss

        num_pos = tf.shape(pos_samples)[0]
        num_neg = tf.shape(neg_samples)[0]

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        acc = self.accuracy_metric(cls_true, cls_pred)

        return {
            "loss": loss_value,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "acc": acc,
            "num_pos": num_pos,
            "num_neg": num_neg,
        }
