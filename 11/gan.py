"""
Generative Adversarial Networks
"""
from types import SimpleNamespace
from tensorflow import (
    random,
    config,
    data,
    shape,
    concat,
    ones,
    zeros,
    GradientTape,
)
from tensorflow.keras import datasets, models, layers, optimizers, losses, metrics
from numpy import concatenate, expand_dims, reshape, squeeze
from numpy.random import normal
from matplotlib.pyplot import figure, subplot, imshow, axis, show


def wrapper():
    """
    wrapper function
    """
    random.set_seed(1)
    print("Num GPUs Available: ", len(config.list_physical_devices("GPU")))
    (x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()
    dataset = concatenate([x_train, x_test], axis=0)
    dataset = expand_dims(dataset, -1).astype("float32") / 255
    batch_size = 64
    dataset = reshape(dataset, (-1, 28, 28, 1))
    dataset = data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    noise_dim = 150
    generator = models.Sequential(
        [
            layers.InputLayer(input_shape=(noise_dim,)),
            layers.Dense(7 * 7 * 256),
            layers.Reshape(target_shape=(7, 7, 256)),
            layers.Conv2DTranspose(
                256, 3, activation="LeakyReLU", strides=2, padding="same"
            ),
            layers.Conv2DTranspose(
                128, 3, activation="LeakyReLU", strides=2, padding="same"
            ),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ]
    )
    print(generator.summary())
    discriminator = models.Sequential(
        [
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(256, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(128, 3, activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    print(discriminator.summary())
    ker = SimpleNamespace(
        optimizer_g=optimizers.Adam(learning_rate=0.00001, beta_1=0.5),
        optimizer_d=optimizers.Adam(learning_rate=0.00003, beta_1=0.5),
        loss_fn=losses.BinaryCrossentropy(from_logits=True),
        g_acc_metric=metrics.BinaryAccuracy(),
        d_acc_metric=metrics.BinaryAccuracy(),
    )

    epoch_func(dataset, generator, noise_dim, discriminator, ker)


def train_d_step(datas, generator, noise_dim, discriminator, ker):
    """
    observe the annotation allows for efficient native tensoflow compiling
    """
    batch_size = shape(datas)[0]
    noise = random.normal(shape=(batch_size, noise_dim))
    y_true = concat([ones(batch_size, 1), zeros(batch_size, 1)], 0)
    with GradientTape() as tape:
        fake = generator(noise)
        xdata = concat([datas, fake], 0)
        y_pred = discriminator(xdata)
        discriminator_loss = ker.loss_fn(y_true, y_pred)
        grads = tape.gradient(discriminator_loss, discriminator.trainable_weights)
    ker.optimizer_d.apply_gradients(zip(grads, discriminator.trainable_weights))
    ker.d_acc_metric.update_state(y_true, y_pred)
    return {
        "discriminator_loss": discriminator_loss,
        "discriminator_accuracy": ker.d_acc_metric.result(),
    }


def train_g_step(datas, generator, noise_dim, discriminator, ker):
    """
    when training the generator, we want it to maximize the probability that its
    output is classified as real, remember the min-max game
    """
    batch_size = shape(datas)[0]
    noise = random.normal(shape=(batch_size, noise_dim))
    y_true = ones(batch_size, 1)
    with GradientTape() as tape:
        y_pred = discriminator(generator(noise))
        generator_loss = ker.loss_fn(y_true, y_pred)
        grads = tape.gradient(generator_loss, generator.trainable_weights)
    ker.optimizer_g.apply_gradients(zip(grads, generator.trainable_weights))
    ker.g_acc_metric.update_state(y_true, y_pred)
    return {
        "generator_loss": generator_loss,
        "generator_accuracy": ker.g_acc_metric.result(),
    }


def plot_images(model, noise_dim):
    """
    plot images function
    """
    images = model(normal(size=(81, noise_dim)))
    figure(figsize=(9, 9))
    for i, image in enumerate(images):
        subplot(9, 9, i + 1)
        imshow(squeeze(image, -1), cmap="Greys_r")
        axis("off")


def epoch_func(dataset, generator, noise_dim, discriminator, ker):
    """
    epoch function
    """
    for epoch in range(30):
        dat = SimpleNamespace(
            d_loss_sum=0, g_loss_sum=0, d_acc_sum=0, g_acc_sum=0, cnt=0
        )
        for batch in dataset:
            d_loss = train_d_step(batch, generator, noise_dim, discriminator, ker)
            dat.d_loss_sum += d_loss["discriminator_loss"]
            dat.d_acc_sum += d_loss["discriminator_accuracy"]
            g_loss = train_g_step(batch, generator, noise_dim, discriminator, ker)
            dat.g_loss_sum += g_loss["generator_loss"]
            dat.g_acc_sum += g_loss["generator_accuracy"]
            dat.cnt += 1
        print(
            f"E:{epoch},",
            f"Loss G:{round(dat.g_loss_sum / dat.cnt, 4)},",
            f"Loss D:{round(dat.d_loss_sum / dat.cnt, 4)},",
            f"Acc G:%{round(100 * dat.g_acc_sum / dat.cnt, 2)},",
            f"Acc D:%{round(100 * dat.d_acc_sum / dat.cnt, 2)}",
        )
        if epoch % 2 == 0:
            plot_images(generator, noise_dim)
    images = generator(normal(size=(81, noise_dim)))
    figure(figsize=(9, 9))
    for i, image in enumerate(images):
        subplot(9, 9, i + 1)
        imshow(squeeze(image, -1), cmap="Greys_r")
        axis("off")


if __name__ == "__main__":
    wrapper()
    show()
