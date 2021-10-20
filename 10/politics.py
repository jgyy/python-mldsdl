"""
Keras Exercise
source export TF_CPP_MIN_LOG_LEVEL=2
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from tensorflow.keras import layers, models, wrappers
from sklearn.model_selection import cross_val_score


def wrapper():
    """
    wrapper function
    """
    feature_names = [
        "party",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missle",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa",
    ]
    voting_data = DataFrame(
        read_csv(
            join(dirname(__file__), "house-votes-84.data.txt"),
            na_values=["?"],
            names=feature_names,
        )
    )
    print(voting_data.head())
    print(voting_data.describe())
    voting_data.dropna(inplace=True)
    print(voting_data.describe())
    voting_data.replace(("y", "n"), (1, 0), inplace=True)
    voting_data.replace(("democrat", "republican"), (1, 0), inplace=True)
    print(voting_data.head())
    all_features = voting_data[feature_names].drop("party", axis=1).values
    all_classes = voting_data["party"].values

    def create_model():
        """
        16 feature inputs (votes) going into an 32-unit layer
        Another hidden layer of 16 units
        Output layer with a binary classification (Democrat or Republican political party)
        Compile model
        """
        model = models.Sequential()
        model.add(
            layers.Dense(
                32, input_dim=16, kernel_initializer="normal", activation="relu"
            )
        )
        model.add(layers.Dense(16, kernel_initializer="normal", activation="relu"))
        model.add(layers.Dense(1, kernel_initializer="normal", activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    estimator = wrappers.scikit_learn.KerasClassifier(
        build_fn=create_model, epochs=100, verbose=0
    )
    cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
    print(cv_scores.mean())


if __name__ == "__main__":
    wrapper()
