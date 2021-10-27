"""
Final Project
"""
from os.path import join, dirname
from types import SimpleNamespace
from io import BytesIO, StringIO
from PIL import Image
from pandas import DataFrame, read_csv
from numpy.random import seed
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers, models, wrappers


def wrapper():
    """
    wrapper function
    """
    masses_data = DataFrame(
        read_csv(join(dirname(__file__), "mammographic_masses.data.txt"))
    )
    print(masses_data.head())
    masses_data = DataFrame(
        read_csv(
            join(dirname(__file__), "mammographic_masses.data.txt"),
            na_values=["?"],
            names=["BI-RADS", "age", "shape", "margin", "density", "severity"],
        )
    )
    print(masses_data.head())
    print(masses_data.describe())
    print(
        masses_data.loc[
            (masses_data["age"].isnull())
            | (masses_data["shape"].isnull())
            | (masses_data["margin"].isnull())
            | (masses_data["density"].isnull())
        ]
    )
    masses_data.dropna(inplace=True)
    print(masses_data.describe())
    all_features = masses_data[["age", "shape", "margin", "density"]].values
    all_classes = masses_data["severity"].values
    feature_names = ["age", "shape", "margin", "density"]
    print(all_features)
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    print(all_features_scaled)
    seed(1234)
    (
        training_inputs,
        testing_inputs,
        training_classes,
        testing_classes,
    ) = train_test_split(
        all_features_scaled, all_classes, train_size=0.75, random_state=1
    )
    tts = SimpleNamespace(
        training_inputs=training_inputs,
        testing_inputs=testing_inputs,
        training_classes=training_classes,
        testing_classes=testing_classes,
    )

    decision(tts, feature_names, all_features, all_features_scaled, all_classes)


def create_model():
    """
    4 feature inputs going into an 6-unit layer
    "Deep learning" turns out to be unnecessary
    Output layer with a binary classification (benign or malignant)
    Compile model; adam seemed to work best
    """
    model = models.Sequential()
    model.add(
        layers.Dense(6, input_dim=4, kernel_initializer="normal", activation="relu")
    )
    model.add(layers.Dense(1, kernel_initializer="normal", activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def decision(tts, feature_names, all_features, all_features_scaled, all_classes):
    """
    decision tree classifier function
    """
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(tts.training_inputs, tts.training_classes)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
    graph = graph_from_dot_data(dot_data.getvalue())
    Image.open(BytesIO(graph.create_png())).show()
    print(clf.score(tts.testing_inputs, tts.testing_classes))
    clf = DecisionTreeClassifier(random_state=1)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    clf = RandomForestClassifier(n_estimators=10, random_state=1)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    cla = 1.0
    svc = SVC(kernel="linear", C=cla)
    cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    clf = KNeighborsClassifier(n_neighbors=10)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    for num in range(1, 50):
        clf = KNeighborsClassifier(n_neighbors=num)
        cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
        print(num, cv_scores.mean())
    scaler = MinMaxScaler()
    all_features_minmax = scaler.fit_transform(all_features)
    clf = MultinomialNB()
    cv_scores = cross_val_score(clf, all_features_minmax, all_classes, cv=10)
    print(cv_scores.mean())
    cla = 1.0
    svc = SVC(kernel="rbf", C=cla)
    cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    cla = 1.0
    svc = SVC(kernel="sigmoid", C=cla)
    cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    cla = 1.0
    svc = SVC(kernel="poly", C=cla)
    cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())
    clf = LogisticRegression()
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())

    estimator = wrappers.scikit_learn.KerasClassifier(
        build_fn=create_model, epochs=100, verbose=0
    )
    cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
    print(cv_scores.mean())


if __name__ == "__main__":
    wrapper()
