"""
Transfer Learning
"""
from os.path import join, dirname
from PIL import Image
from tensorflow.keras import applications, preprocessing
from numpy import expand_dims


def wrapper():
    """
    wrapper function
    """

    def image(filename):
        img_path = join(dirname(__file__), filename)
        Image.open(img_path).show()
        return img_path

    img_path = image(filename="fighterjet.jpg")
    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    xdata = preprocessing.image.img_to_array(img)
    xdata = expand_dims(xdata, axis=0)
    xdata = applications.resnet50.preprocess_input(xdata)
    model = applications.resnet50.ResNet50(weights="imagenet")
    preds = model.predict(xdata)
    print(
        f"Predicted {img_path}:\n",
        applications.resnet50.decode_predictions(preds, top=3)[0],
    )

    def classify(img_path):
        img_path = image(filename=img_path)
        img = preprocessing.image.load_img(img_path, target_size=(224, 224))
        xdata = preprocessing.image.img_to_array(img)
        xdata = expand_dims(xdata, axis=0)
        xdata = applications.resnet50.preprocess_input(xdata)
        preds = model.predict(xdata)
        print(
            f"Predicted {img_path}:\n",
            applications.resnet50.decode_predictions(preds, top=3)[0],
        )

    classify("bunny.jpg")
    classify("firetruck.jpg")
    classify("breakfast.jpg")
    classify("castle.jpg")
    classify("VLA.jpg")
    classify("bridge.jpg")


if __name__ == "__main__":
    wrapper()
