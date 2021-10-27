"""
Introducing Tensorflow
export TF_CPP_MIN_LOG_LEVEL=2
"""
from types import SimpleNamespace
from tensorflow import (
    Variable,
    GradientTape,
    nn,
    data,
    math,
    initializers,
    zeros,
    add,
    equal,
    matmul,
    cast,
    one_hot,
    clip_by_value,
    reduce_mean,
    reduce_sum,
)
from tensorflow.keras import datasets, optimizers
from numpy import float32, int64, concatenate, reshape, array, argmax
from matplotlib.pyplot import figure, show, title, imshow, get_cmap


def simple():
    """
    The world's simplest Tensorflow application
    """
    avar = Variable(1, name="a")
    bvar = Variable(2, name="b")
    fgraph = avar + bvar
    print("The sum of a and b is", fgraph)


def mnist_func():
    """
    Prepare MNIST data
    """
    num_classes = 10
    num_features = 784
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = array(x_train, float32), array(x_test, float32)
    x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape(
        [-1, num_features]
    )
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return SimpleNamespace(
        num_classes=num_classes,
        num_features=num_features,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def images_func(mni):
    """
    images function
    """
    figure()
    images = mni.x_train[0].reshape([1, 784])
    for i in range(1, 500):
        images = concatenate((images, mni.x_train[i].reshape([1, 784])))
    imshow(images, cmap=get_cmap("gray_r"))


def display_sample(num, mni):
    """
    Print this sample's label
    Reshape the 784 values to a 28x28 image
    """
    figure()
    label = mni.y_train[num]
    image = mni.x_train[num].reshape([28, 28])
    title(f"Sample: {num}  Label: {label}")
    imshow(image, cmap=get_cmap("gray_r"))


def neural_net(input_data, param):
    """
    Hidden fully connected layer with 512 neurons.
    Apply sigmoid to hidden_layer output for non-linearity.
    Output fully connected layer with a neuron for each class.
    Apply softmax to normalize the logits to a probability distribution.
    """
    hidden_layer = add(matmul(input_data, param.weights["h"]), param.biases["b"])
    hidden_layer = nn.sigmoid(hidden_layer)
    out_layer = matmul(hidden_layer, param.weights["out"]) + param.biases["out"]
    return nn.softmax(out_layer)


def cross_entropy(y_pred, y_true, mni):
    """
    Encode label to a one hot vector.
    Clip prediction values to avoid log(0) error.
    Compute cross-entropy.
    """
    y_true = one_hot(y_true, mni.num_classes, on_value=None, off_value=None)
    y_pred = clip_by_value(y_pred, 1e-9, 1.0)
    return reduce_mean(-reduce_sum(y_true * math.log(y_pred)))


def run_optimization(xdata, ydata, param, mni, optimizer):
    """
    Wrap computation inside a GradientTape for automatic differentiation.
    Variables to update, i.e. trainable variables.
    Compute gradients. > Update W and b following gradients.
    """
    with GradientTape() as gra:
        pred = neural_net(xdata, param)
        loss = cross_entropy(pred, ydata, mni)
    trainable_variables = list(param.weights.values()) + list(param.biases.values())
    gradients = gra.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def accuracy(y_pred, y_true):
    """
    Predicted class is the index of highest score in prediction vector (i.e. argmax).
    """
    correct_prediction = equal(argmax(y_pred, 1), cast(y_true, int64))
    return reduce_mean(cast(correct_prediction, float32), axis=-1)


def wrapper():
    """
    wrapper function
    """
    simple()
    mni = mnist_func()
    display_sample(1000, mni)
    images_func(mni)

    train_data = data.Dataset.from_tensor_slices((mni.x_train, mni.y_train))
    train_data = train_data.repeat().shuffle(60000).batch(250).prefetch(1)
    random_normal = initializers.RandomNormal()
    param = SimpleNamespace(
        learning_rate=0.001,
        training_steps=3000,
        display_step=100,
        n_images=200,
        weights={
            "h": Variable(random_normal([mni.num_features, 512])),
            "out": Variable(random_normal([512, mni.num_classes])),
        },
        biases={
            "b": Variable(zeros([512])),
            "out": Variable(zeros([mni.num_classes])),
        },
        test_images=mni.x_test[:200],
        test_labels=mni.y_test[:200],
    )
    optimizer = optimizers.SGD(param.learning_rate)
    for step, (batch_x, batch_y) in enumerate(train_data.take(param.training_steps), 1):
        run_optimization(batch_x, batch_y, param, mni, optimizer)
        if step % param.display_step == 0:
            pred = neural_net(batch_x, param)
            loss = cross_entropy(pred, batch_y, mni)
            acc = accuracy(pred, batch_y)
            print(f"Training epoch: {step}, Loss: {loss}, Accuracy: {acc}")
    pred = neural_net(mni.x_test, param)
    print(f"Test Accuracy: {accuracy(pred, mni.y_test)}")
    predictions = neural_net(param.test_images, param)
    for i in range(param.n_images):
        model_prediction = argmax(predictions.numpy()[i])
        if model_prediction != param.test_labels[i]:
            figure()
            imshow(reshape(param.test_images[i], [28, 28]), cmap="gray_r")
            print(f"Original Labels: {param.test_labels[i]}")
            print(f"Model prediction: {model_prediction}")


if __name__ == "__main__":
    wrapper()
    show()
