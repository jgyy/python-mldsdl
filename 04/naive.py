"""
Naive Bayes (the easy way)
"""
from io import open
from os import walk
from os.path import join, dirname
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def read_files(path):
    """
    read files function
    """
    for root, _, filenames in walk(path):
        for filename in filenames:
            path = join(root, filename)
            in_body = False
            lines = []
            with open(path, "r", encoding="latin1") as file:
                for line in file:
                    if in_body:
                        lines.append(line)
                    elif line == "\n":
                        in_body = True
            message = "\n".join(lines)
            yield path, message


def data_frame_from_directory(path, classification):
    """
    data frame from directory function
    """
    rows = []
    index = []
    for filename, message in read_files(path):
        rows.append({"message": message, "class": classification})
        index.append(filename)
    return DataFrame(rows, index=index)


def wrapper():
    """
    wrapper function
    """
    email_spam = join(dirname(__file__), "emails/spam")
    email_ham = join(dirname(__file__), "emails/ham")
    data = DataFrame({"message": [], "class": []})
    data = data.append(data_frame_from_directory(email_spam, "spam"))
    data = data.append(data_frame_from_directory(email_ham, "ham"))
    print(data.head())
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data["message"].values)
    classifier = MultinomialNB()
    targets = data["class"].values
    classifier.fit(counts, targets)
    examples = ["Free Viagra now!!!", "Hi Bob, how about a game of golf tomorrow?"]
    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print(predictions)


if __name__ == "__main__":
    wrapper()
