"""
Introducting Keras
"""
from tensorflow.keras import datasets, models, layers, backend, utils
from matplotlib.pyplot import figure, title, imshow, show, get_cmap


def wrapper():
    """
    wrapper function
    """
    (mnist_train_images, mnist_train_labels), (
        mnist_test_images,
        mnist_test_labels,
    ) = datasets.mnist.load_data()
    if backend.image_data_format() == "channels_first":
        train_images = mnist_train_images.reshape(
            mnist_train_images.shape[0], 1, 28, 28
        )
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        train_images = mnist_train_images.reshape(
            mnist_train_images.shape[0], 28, 28, 1
        )
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")
    train_images /= 255
    test_images /= 255
    train_labels = utils.to_categorical(mnist_train_labels, 10)
    test_labels = utils.to_categorical(mnist_test_labels, 10)

    def display_sample(num):
        """
        Print the one-hot array of this sample's label
        Print the label converted back to a number
        Reshape the 768 values to a 28x28 image
        """
        figure()
        print(train_labels[num])
        label = train_labels[num].argmax(axis=0)
        image = train_images[num].reshape([28, 28])
        title("Sample: %d  Label: %d" % (num, label))
        imshow(image, cmap=get_cmap("gray_r"))

    display_sample(1234)
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))
    print(model.summary())
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs=10,
        validation_data=(test_images, test_labels),
    )
    score = model.evaluate(test_images, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    wrapper()
    show()
