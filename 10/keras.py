"""
Introducing Keras
export TF_CPP_MIN_LOG_LEVEL=2
"""
from tensorflow.keras import datasets, utils, models, layers, optimizers
from matplotlib.pyplot import title, figure, imshow, show, get_cmap


def wrapper():
    """
    wrapper function
    """
    (mnist_train_images, mnist_train_labels), (
        mnist_test_images,
        mnist_test_labels,
    ) = datasets.mnist.load_data()
    train_images = mnist_train_images.reshape(60000, 784)
    test_images = mnist_test_images.reshape(10000, 784)
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
    model.add(layers.Dense(512, activation="relu", input_shape=(784,)))
    model.add(layers.Dense(10, activation="softmax"))
    print(model.summary())
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        batch_size=100,
        epochs=10,
        validation_data=(test_images, test_labels),
    )
    score = model.evaluate(test_images, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    for xdata in range(1000):
        print(f"Checking image number {xdata}.")
        test_image = test_images[xdata, :].reshape(1, 784)
        predicted_cat = model.predict(test_image).argmax()
        label = test_labels[xdata].argmax()
        if predicted_cat != label:
            figure()
            title(f"Prediction: {predicted_cat} Label: {label}")
            imshow(test_image.reshape([28, 28]), cmap=get_cmap("gray_r"))


if __name__ == "__main__":
    wrapper()
    show()
