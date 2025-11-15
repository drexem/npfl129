#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection


import urllib.request
import urllib.error
import ssl

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="imdb_sentiment.model", type=str, help="Model path")
# TODO: Add other arguments (typically hyperparameters) as you need.


class Dataset:
    """IMDB dataset.

    This is a modified IMDB dataset for sentiment classification. The text is
    already tokenized and partially normalized.
    """
    def __init__(self,
                 name="imdb_train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []
        with open(name) as f_imdb:
            for line in f_imdb:
                label, text = line.split("\t", 1)
                self.data.append(text)
                self.target.append(int(label))


def load_word_embeddings(
        name="imdb_embeddings.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
    """Load word embeddings.

    These are selected word embeddings from FastText. For faster download, it
    only contains words that are in the IMDB dataset.
    """
    if not os.path.exists(name):
        print("Downloading embeddings {}...".format(name), file=sys.stderr)

        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create opener with the SSL context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
        os.rename("{}.tmp".format(name), name)

    with open(name, "rb") as f_emb:
        data = np.load(f_emb)
        words = data["words"]
        vectors = data["vectors"]
    embeddings = {word: vector for word, vector in zip(words, vectors)}
    return embeddings

def preprocess_data(data, word_embeddings):
    _, value = next(iter(word_embeddings.items()))
    embedding_dim = value.shape[0]
    result = np.zeros((len(data), embedding_dim))

    for i, text in enumerate(data):
        words = text.strip().split()
        word_vectors = []

        for word in words:
            if word in word_embeddings and word not in sklearn.feature_extraction._stop_words.ENGLISH_STOP_WORDS:
                word_vectors.append(word_embeddings[word])

        if word_vectors:
            result[i] = np.mean(word_vectors, axis=0)

    return result


def preprocess_data_count_vectorizer(data, vectorizer=None, fit=True):
    if vectorizer is None:
        vectorizer = CountVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

    if fit:
        result = vectorizer.fit_transform(data)
    else:
        result = vectorizer.transform(data)

    return result.toarray(), vectorizer

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    word_embeddings = load_word_embeddings()

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        print("Preprocessing dataset.", file=sys.stderr)
        # TODO: Preprocess the text such that you have a single vector per movie
        # review. You can experiment with different ways of pooling the word
        # embeddings: averaging, max pooling, etc. You can also try to exclude
        # words that do not contribute much to the meaning of the sentence (stop
        # words). See `sklearn.feature_extraction._stop_words.ENGLISH_STOP_WORDS`.
        train_as_vectors, vectorizer = preprocess_data_count_vectorizer(train.data, fit=True)

        train_x, validation_x, train_y, validation_y = sklearn.model_selection.train_test_split(
            train_as_vectors, train.target, test_size=0.25, random_state=args.seed)

        print("Training.", file=sys.stderr)
        # TODO: Train a model of your choice on the given data.
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            random_state=args.seed,
            max_iter=300,
            verbose=True,
            early_stopping=False,
            tol=0.0,
            n_iter_no_change=1000
        )
        model.fit(train_x, train_y)


        print("Evaluation.", file=sys.stderr)
        validation_predictions = model.predict(validation_x)
        validation_accuracy = sklearn.metrics.accuracy_score(validation_y, validation_predictions)
        print("Validation accuracy {:.2f}%".format(100 * validation_accuracy))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model,vectorizer), model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            (model,vectorizer) = pickle.load(model_file)

        # TODO: Start by preprocessing the test data, ideally using the same
        # code as during training.
        test_as_vectors, _ = preprocess_data_count_vectorizer(test.data, vectorizer, False)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test_as_vectors)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
