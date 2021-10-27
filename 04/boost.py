"""
XGBoost
"""
from types import SimpleNamespace
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import DMatrix, train as xgbtrain


def wrapper():
    """
    wrapper function
    """
    iris = SimpleNamespace(**load_iris())
    num_samples, num_features = iris.data.shape
    print(num_samples)
    print(num_features)
    print(list(iris.target_names))
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0
    )
    train = DMatrix(x_train, label=y_train)
    test = DMatrix(x_test, label=y_test)
    param = {"max_depth": 4, "eta": 0.3, "objective": "multi:softmax", "num_class": 3}
    epochs = 10
    model = xgbtrain(param, train, epochs)
    predictions = model.predict(test)
    print(predictions)
    print(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    wrapper()
