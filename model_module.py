import tensorflow as tf


class TPSE(tf.keras.Model):
    def __init__(self, autoencoder, class_model, **kwargs):
        """
        The Two Phase Supervised Encoder.
        :param autoencoder: The autoencoder network (encoder and decoder networks).
        :param class_model: The Separator network.
        """
        super(TPSE, self).__init__(**kwargs)

        # the autoencoder network
        self.autoencoder = autoencoder

        # the separator network
        self.class_model = class_model

    def compile(self,
                class_optimizer,
                ae_optimizer,
                class_loss_fn,
                ae_loss_fn,
                class_metric,
                ae_metric):
        """
        Method for setting the model configuration for training.

        :param class_optimizer:  The optimizer for the supervised phase.
        :param ae_optimizer: The optimizer for the unsupervised phase.
        :param class_loss_fn: The loss function of the supervised phase.
        :param ae_loss_fn: The loss function of the unsupervised phase.
        :param class_metric: The metric for the supervised phase.
        :param ae_metric: The metric for the unsupervised phase.
        :return:
        """

        super(TPSE, self).compile()
        self.class_optimizer = class_optimizer
        self.ae_optimizer = ae_optimizer
        self.class_loss_fn = class_loss_fn
        self.ae_loss_fn = ae_loss_fn
        self.class_metric = class_metric
        self.ae_metric = ae_metric
        self.acc_metric = tf.keras.metrics.Accuracy()

    @property
    def metrics(self):
        """
        A method for returning the model's metrics.
        :return: a list of the model's metrics.
        """
        return [self.ae_metric, self.class_metric, self.acc_metric]

    def train_step(self, data):
        """
        A method for the logic for one training step.
        :param data: the input data.
        :return: the results obtained from the metrics.
        """
        # unpack the input data
        x_data, y_data = data

        # unsupervised phase
        with tf.GradientTape() as tape:
            reconstructions = self.autoencoder(x_data, training=True)
            loss_value = self.ae_loss_fn(x_data, reconstructions)

        grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)
        self.ae_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        # supervised phase
        with tf.GradientTape() as tape:
            predictions = self.class_model(x_data, training=True)
            loss_value_class = self.class_loss_fn(y_data, predictions)

        grads_class = tape.gradient(loss_value_class, self.class_model.trainable_weights)
        self.class_optimizer.apply_gradients(zip(grads_class, self.class_model.trainable_weights))

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions, axis=1))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        A method for the logic for one evaluation step.
        :param data: the input data.
        :return:
        """

        # unpack the input data
        x_data, y_data = data

        # compute the reconstructions and the predictions
        reconstructions = self.autoencoder(x_data,
                                           training=False)
        predictions = self.class_model(x_data,
                                       training=False)

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions, axis=1))

        return {m.name: m.result() for m in self.metrics}

    def call(self, data):
        """
        A method for calling the model on new inputs.
        :param data: the input data.
        :return: The output of the autoencoder and the separator network.
        """
        return self.autoencoder(data), self.class_model(data)


def build_test_model(input_shape: list,
                     latent_dims: int,
                     number_of_classes: int,
                     summary_flag: bool = True):
    """
    A function for constructing the components and the architecture of TP-SE.
    :param input_shape: the input shape in channels last format.
    :param latent_dims: the number of dimensions in the latent space.
    :param number_of_classes: the number of classes.
    :param summary_flag: a flag for displaying the summary of the constructed models.
    :return: The autoencoder network (encoder and decoder networks) along with the Separator network.
    """

    # create the input layer
    input_layer = tf.keras.Input(shape=input_shape,
                                 name="InputLayer")

    # ===================================================================================
    # Encoder network
    x = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",
                               name="CONV1")(input_layer)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",
                               name="CONV2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Flatten()(x)

    enc = tf.keras.layers.Dense(units=latent_dims,
                                activation="linear",
                                name="LatentLayer")(x)

    # ===================================================================================
    # ===================================================================================
    # Separator network
    enc1 = tf.keras.layers.Dense(units=64,
                                 name="Class1")(enc)
    enc1 = tf.keras.layers.Dropout(0.5)(enc1)
    enc1 = tf.keras.layers.ReLU()(enc1)

    classification_layer = tf.keras.layers.Dense(units=number_of_classes,
                                                 name="ClassLayer")(enc1)
    # ===================================================================================
    # ===================================================================================
    # Decoder network
    x = tf.keras.layers.Dense(units=(input_shape[0] // 4) * (input_shape[1] // 4) * 128,
                              activation="relu",
                              name="Decoder1")(enc)

    x = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 128))(x)

    x = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=(3, 3),
                                        strides=2,
                                        padding="same",
                                        name="DCONV2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=(3, 3),
                                        strides=2,
                                        padding="same",
                                        name="DCONV1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    output = tf.keras.layers.Conv2D(filters=input_shape[-1],
                                    kernel_size=(3, 3),
                                    strides=1,
                                    activation="sigmoid",
                                    padding="same",
                                    name="AEOUT")(x)
    # ===================================================================================

    # create the autoencoder network (encoder and decoder networks) and the separator network respectively
    autoencoder = tf.keras.models.Model(inputs=input_layer,
                                        outputs=output,
                                        name="Autoencoder")
    class_model = tf.keras.models.Model(inputs=input_layer,
                                        outputs=classification_layer,
                                        name="ClassificationModel")

    if summary_flag:
        # see a summary of the architectures
        autoencoder.summary()
        class_model.summary()

    # return the autoencoder and separator models
    return autoencoder, class_model


def wrapper_lr_schedule(number_of_epochs: int):
    """
    A wrapper function for the learning rate scheduler function.
    :param number_of_epochs: the total number of epochs.
    :return: the scheduling function for the learning rate.
    """
    epoch_lim = number_of_epochs // 3

    def lr_schedule(epoch: int,
                    lr: float):
        """
        A function for scheduling the learning rate.
        :param epoch: the epoch number.
        :param lr: the value of the learning rate.
        :return: the value of the scheduled learning rate.
        """
        if ((epoch + 1) % epoch_lim) == 0:
            return lr / 3
        else:
            return lr

    return lr_schedule
