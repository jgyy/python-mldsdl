"""
Variational AutoEncoders (VAE)
"""
from os import environ
from random import seed as rseed
from numpy import concatenate, expand_dims
from numpy.random import seed as nseed, choice, normal
from matplotlib.pyplot import (
    subplot,
    figure,
    show,
    axis,
    imshow,
    plot,
    legend,
    scatter,
    colorbar,
    xlabel,
    ylabel,
)
from tensorflow import (
    config,
    random,
    shape,
    exp,
    reduce_mean,
    reduce_sum,
    square,
    GradientTape,
)
from tensorflow.keras import (
    datasets,
    layers,
    backend,
    models,
    losses,
    metrics,
    optimizers,
    Model,
    Input,
)


class SamplingLayer(layers.Layer):
    """
    Reparameterization Trick z = mu + sigma * epsilon
    """

    def call(self, inputs, *args, **kwargs):
        """
        call function
        """
        print([*args], {**kwargs}, end="\r")
        z_mean, z_log_var = inputs
        batch = shape(z_mean)[0]
        dim = shape(z_mean)[1]
        epsilon = backend.random_normal(shape=(batch, dim))
        return z_mean + exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """
    VAE class
    """

    def __init__(self, encoder, decoder):
        """
        register total loss as an observable metric in the model training history
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.ce_loss_tracker = metrics.Mean(name="ce_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        """
        call
        """
        print(self, inputs, training, mask)

    @staticmethod
    def get_config():
        """
        get_config
        """
        print("get_config")

    @property
    def metrics(self):
        """
        This are all observable metrics
        """
        return [self.total_loss_tracker, self.ce_loss_tracker, self.kl_loss_tracker]

    @staticmethod
    def reconstruction_loss(data, reconstructed):
        """
        reconstruction loss function
        """
        return reduce_mean(
            reduce_sum(losses.binary_crossentropy(data, reconstructed), axis=(1, 2))
        )

    @staticmethod
    def kl_divergence_loss(z_mean, z_log_var):
        """
        kl divergence loss function
        """
        return reduce_mean(
            reduce_sum(-0.5 * (1 + z_log_var - square(z_mean) - exp(z_log_var)), axis=1)
        )

    def calc_total_loss(self, data, reconstructed, z_mean, z_log_var):
        """
        calculate total loss function
        """
        loss1 = self.reconstruction_loss(data, reconstructed)
        loss2 = self.kl_divergence_loss(z_mean, z_log_var)
        kl_weight = 3.0
        return loss1, loss2, loss1 + kl_weight * loss2

    def train_step(self, data):
        """
        Now calculate loss + calculate gradients + update weights
        Gradient tape is a recording of all gradients for the trainable
        weights that need to be updated > forward path > backward path
        keep track of loss > return the loss for history object
        """
        with GradientTape() as tape:
            z_mean, z_log_var, zdata = self.encoder(data)
            reconstruction = self.decoder(zdata)
            ce_loss, kl_loss, total_loss = self.calc_total_loss(
                data, reconstruction, z_mean, z_log_var
            )
        # backward path
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # keep track of loss
        self.total_loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # return the loss for history object
        return {
            "total_loss": self.total_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_encoder(latent_dim, encoder_inputs):
    """
    Given a batch of images the convolutional block extracts the features
    pass the inputs through the convolutional block
    a dedicated layer to learn mean in parallel
    a dedicated layer to learn variance in parallel
    now the reparametrization trick to find z as defined by mean and variance
    """
    lvl1 = models.Sequential(
        [
            layers.Conv2D(128, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
        ]
    )
    xdata = lvl1(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(xdata)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(xdata)
    zdata = SamplingLayer()([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, zdata], name="encoder")


def build_decoder(latent_inputs):
    """
    build decoder function
    """
    lvl1 = models.Sequential(
        [
            layers.Dense(
                7 * 7 * 64, activation="relu", input_shape=(latent_inputs.shape[1],)
            ),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(
                128, 3, activation="relu", strides=2, padding="same"
            ),
            layers.Conv2DTranspose(
                64, 3, activation="relu", strides=2, padding="same"
            ),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ]
    )
    return Model(latent_inputs, lvl1(latent_inputs), name="decoder")


def wrapper():
    """
    wrapper function
    """
    print("Num GPUs Available: ", len(config.list_physical_devices("GPU")))
    seed = 123456
    environ["PYTHONHASHSEED"] = str(seed)
    environ["TF_CUDNN_DETERMINISTIC"] = "1"
    rseed(seed)
    nseed(seed)
    random.set_seed(seed)
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    figure(figsize=(9, 9))
    rnd_samples = choice(60000, 9)
    for i in range(9):
        subplot(3, 3, i + 1)
        imshow(x_train[rnd_samples[i]], cmap="Greys_r")
        axis("off")
    dataset = concatenate([x_train, x_test], axis=0)
    dataset = expand_dims(dataset, -1).astype("float32") / 255

    encoder_inputs = Input(shape=(28, 28, 1))
    encoder = build_encoder(2, encoder_inputs)
    print(encoder.summary())

    latent_inputs = Input(shape=(2,))
    decoder = build_decoder(latent_inputs)
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))
    history = vae.fit(dataset, epochs=32, batch_size=128)

    figure(figsize=(10, 9))
    plot(history.history.get("total_loss"), label="total loss")
    plot(history.history.get("ce_loss"), label="reconstruction loss")
    plot(history.history.get("kl_loss"), label="KL loss")
    legend()
    figure(figsize=(10, 9))
    plot(history.history.get("kl_loss"), label="KL loss")
    legend()
    figure()
    synth = vae.decoder.predict([[1, 2]])
    axis("off")
    imshow(synth.reshape((28, 28)), cmap="Greys_r")

    images_func(vae, x_train, y_train, y_test, dataset)


def images_func(vae, x_train, y_train, y_test, dataset):
    """
    images function
    """
    zdata = normal(loc=0, scale=4, size=(256, 2))
    synth = vae.decoder.predict(zdata)
    figure(figsize=(28, 28))
    for i in range(256):
        subplot(16, 16, i + 1)
        imshow(synth[i].reshape((28, 28)), cmap="Greys_r")
        axis("off")
    idx = 1280
    batch = expand_dims(x_train[idx], axis=0)
    batch_of_images = expand_dims(batch, axis=-1).astype("float32") / 255
    print(batch_of_images.shape)
    _, _, zdata = vae.encoder.predict(batch_of_images)
    synth = vae.decoder.predict([zdata])
    print(zdata)

    figure(figsize=(28, 28))
    subplot(1, 2, 1)
    axis("off")
    imshow(x_train[idx], cmap="Greys_r")
    subplot(1, 2, 2)
    axis("off")
    imshow(synth[0].reshape((28, 28)), cmap="Greys_r")

    labels = concatenate([y_train, y_test], axis=0)
    meu, _, _ = vae.encoder.predict(dataset)
    figure(figsize=(12, 10))
    scatter(meu[:, 0], meu[:, 1], c=labels)
    colorbar()
    xlabel("meu[0]")
    ylabel("meu[1]")


if __name__ == "__main__":
    wrapper()
    show()
