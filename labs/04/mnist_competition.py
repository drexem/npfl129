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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import ssl


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--grid_search", default=True, action="store_true", help="Perform grid search")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url + name, context=ctx) as resp, open("{}.tmp".format(name), "wb") as out:
                out.write(resp.read())
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Normalize the data to 0-1 range
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train.data / 255.0)

        if args.grid_search:
            # Perform grid search to find best hyperparameters
            print("Performing grid search...", file=sys.stderr)

            param_grid = {
                'hidden_layer_sizes': [(128, 64), (256, 128), (128, 64, 32), (256, 128, 64)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }

            mlp_base = MLPClassifier(
                solver='adam',
                max_iter=30,
                random_state=args.seed,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )

            grid_search = GridSearchCV(
                mlp_base,
                param_grid,
                cv=3,
                verbose=2,
                n_jobs=-1,
                scoring='accuracy'
            )

            grid_search.fit(train_data, train.target)

            print("\nBest parameters found:", file=sys.stderr)
            print(grid_search.best_params_, file=sys.stderr)
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}", file=sys.stderr)

            mlp = grid_search.best_estimator_
        else:
            # Train an MLP neural network with default parameters
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=100,
                random_state=args.seed,
                verbose=True,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )

            print("Training neural network...", file=sys.stderr)
            mlp.fit(train_data, train.target)

        # Compress the model as suggested in the comments
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Store both the scaler and the model
        model = {'scaler': scaler, 'mlp': mlp}

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Normalize test data and generate predictions
        test_data = model['scaler'].transform(test.data / 255.0)
        predictions = model['mlp'].predict(test_data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
