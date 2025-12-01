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

import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.05, type=float)
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        param_grid = [
            {
                'tfidf__analyzer': ['word'],
                'tfidf__ngram_range': [(1, 2), (1, 3)],
                'tfidf__max_features': [10_000, 50_000, 70_000],
                'tfidf__min_df': [1, 2, 3],
                'tfidf__max_df': [0.9, 0.95, 1.0],
                'tfidf__stop_words': [None, 'english'],
                'classifier__alpha': [0.01, 0.1, 1.0, 10.0]
            },
            {
                'tfidf__analyzer': ['char'],
                'tfidf__ngram_range': [(1, 4), (1, 5), (2, 5)],
                'tfidf__max_features': [10_000, 50_000, 70_000],
                'tfidf__min_df': [1, 2],
                'tfidf__max_df': [0.9, 0.95, 1.0],
                'classifier__alpha': [0.01, 0.1, 1.0, 10.0]
            }
        ]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])

        print("Starting grid search...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(train_data, train_target)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        test_predictions = model.predict(test_data)
        f1 = sklearn.metrics.f1_score(test_target, test_predictions)
        print(f"Test F1 score: {f1:.4f}")

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        test_predictions = model.predict(test_data)
        f1 = sklearn.metrics.f1_score(test_target, test_predictions)
        print(f"Test F1 score: {f1:.4f}")

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
