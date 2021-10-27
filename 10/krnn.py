"""
Recurring Neural Networks with Keras
"""
from tensorflow.keras import preprocessing, models, layers, datasets


def wrapper():
    """
    wrapper function
    """
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=20000)
    print(x_train[0])
    print(y_train[0])
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=80)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=80)
    model = models.Sequential()
    model.add(layers.Embedding(20000, 128))
    model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test)
    )
    score, acc = model.evaluate(x_test, y_test, batch_size=32)
    print("Test score:", score)
    print("Test accuracy:", acc)


if __name__ == "__main__":
    wrapper()
