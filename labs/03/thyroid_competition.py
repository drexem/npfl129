#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import warnings
from typing import Optional
import urllib.request
import urllib.error
import ssl

import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2, type=float, help="Test set size for evaluation")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url + name, context=ctx) as resp, open("{}.tmp".format(name), "wb") as out:
                out.write(resp.read())
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Split data for local evaluation
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed
        )

        preprocessor = sklearn.preprocessing.StandardScaler()

        models_params = [
            # Base SGDClassifier_Hinge - L2 penalty with expanded grid
            {
                'name': 'SGDClassifier_Hinge_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.000001, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                    'classifier__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                    'classifier__eta0': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    'classifier__average': [False, True, 3, 5, 10, 20, 50],
                    'classifier__power_t': [0.1, 0.25, 0.5, 0.75],
                }
            },
            # Base SGDClassifier_Hinge - L1 penalty with expanded grid
            {
                'name': 'SGDClassifier_Hinge_L1',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l1'))
                ]),
                'params': {
                    'classifier__alpha': [0.000001, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                    'classifier__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                    'classifier__eta0': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    'classifier__average': [False, True, 3, 5, 10, 20, 50],
                    'classifier__power_t': [0.1, 0.25, 0.5, 0.75],
                }
            },
            # ElasticNet with expanded grid
            {
                'name': 'SGDClassifier_Hinge_ElasticNet',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='elasticnet'))
                ]),
                'params': {
                    'classifier__alpha': [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02],
                    'classifier__l1_ratio': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                    'classifier__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                    'classifier__average': [False, True, 5, 10],
                    'classifier__eta0': [0.01, 0.05],
                }
            },
            # Polynomial interaction features - L2 with expanded grid
            {
                'name': 'Poly2_Interaction_SGDClassifier_Hinge_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                    ('scaler', sklearn.preprocessing.StandardScaler()),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    'classifier__learning_rate': ['optimal', 'adaptive', 'invscaling'],
                    'classifier__average': [False, True, 3, 5, 10, 20],
                    'classifier__eta0': [0.01, 0.05, 0.1],
                    'classifier__power_t': [0.25, 0.5],
                }
            },
            # Polynomial interaction features - L1 with expanded grid
            {
                'name': 'Poly2_Interaction_SGDClassifier_Hinge_L1',
                'estimator': sklearn.pipeline.Pipeline([
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                    ('scaler', sklearn.preprocessing.StandardScaler()),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l1'))
                ]),
                'params': {
                    'classifier__alpha': [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                    'classifier__learning_rate': ['optimal', 'adaptive', 'invscaling'],
                    'classifier__average': [False, True, 3, 5, 10, 20],
                    'classifier__eta0': [0.01, 0.05, 0.1],
                    'classifier__power_t': [0.25, 0.5],
                }
            },
            # Polynomial interaction features - ElasticNet
            {
                'name': 'Poly2_Interaction_SGDClassifier_Hinge_ElasticNet',
                'estimator': sklearn.pipeline.Pipeline([
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                    ('scaler', sklearn.preprocessing.StandardScaler()),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='elasticnet'))
                ]),
                'params': {
                    'classifier__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                    'classifier__l1_ratio': [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'classifier__learning_rate': ['optimal', 'adaptive', 'invscaling'],
                    'classifier__average': [False, True, 10],
                }
            },
            # Full polynomial features with L2
            {
                'name': 'Poly2_Full_SGDClassifier_Hinge_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
                    ('scaler', sklearn.preprocessing.StandardScaler()),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                    'classifier__learning_rate': ['optimal', 'adaptive'],
                    'classifier__average': [False, True, 5],
                }
            },
            # Full polynomial features with L1
            {
                'name': 'Poly2_Full_SGDClassifier_Hinge_L1',
                'estimator': sklearn.pipeline.Pipeline([
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
                    ('scaler', sklearn.preprocessing.StandardScaler()),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge', penalty='l1'))
                ]),
                'params': {
                    'classifier__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                    'classifier__learning_rate': ['optimal', 'adaptive'],
                    'classifier__average': [False, True, 5],
                }
            },
            # Fine-tuned with very granular alpha search
            {
                'name': 'SGDClassifier_Hinge_FineTuned',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='hinge'))
                ]),
                'params': {
                    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                    'classifier__alpha': [0.000005, 0.00001, 0.00002, 0.00003, 0.00005, 0.00008, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.002],
                    'classifier__l1_ratio': [0.15, 0.3, 0.5, 0.7],
                    'classifier__learning_rate': ['optimal', 'invscaling'],
                    'classifier__average': [False, True, 5, 10],
                    'classifier__fit_intercept': [True, False],
                }
            },
            # Modified Huber loss variant - expanded
            {
                'name': 'SGDClassifier_ModifiedHuber_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='modified_huber', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02],
                    'classifier__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                    'classifier__average': [False, True, 10],
                    'classifier__eta0': [0.01, 0.05],
                }
            },
            # Modified Huber loss with L1 - NEW
            {
                'name': 'SGDClassifier_ModifiedHuber_L1',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='modified_huber', penalty='l1'))
                ]),
                'params': {
                    'classifier__alpha': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                    'classifier__learning_rate': ['optimal', 'invscaling'],
                    'classifier__average': [False, True],
                }
            },
            # Log loss variant - NEW
            {
                'name': 'SGDClassifier_LogLoss_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='log_loss', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                    'classifier__learning_rate': ['optimal', 'invscaling'],
                    'classifier__average': [False, True],
                }
            },
            # Squared Hinge variant - NEW
            {
                'name': 'SGDClassifier_SquaredHinge_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.SGDClassifier(random_state=args.seed, max_iter=3000, tol=1e-4, loss='squared_hinge', penalty='l2'))
                ]),
                'params': {
                    'classifier__alpha': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                    'classifier__learning_rate': ['optimal', 'invscaling'],
                    'classifier__average': [False, True],
                }
            },
            # Logistic Regression as baseline - NEW
            {
                'name': 'LogisticRegression_L2',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.LogisticRegression(random_state=args.seed, max_iter=3000, penalty='l2', solver='saga'))
                ]),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
                }
            },
            # RidgeClassifier - NEW
            {
                'name': 'RidgeClassifier',
                'estimator': sklearn.pipeline.Pipeline([
                    ('scaler', preprocessor),
                    ('classifier', sklearn.linear_model.RidgeClassifier(random_state=args.seed))
                ]),
                'params': {
                    'classifier__alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
                    'classifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
                }
            },
        ]

        best_score = 0
        best_model = None
        best_model_name = None

        print("Starting EXPANDED grid search (~10,000+ combinations, estimated 25-35 minutes)...", file=sys.stderr)

        for model_config in models_params:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Searching {model_config['name']}...", file=sys.stderr)

            grid_search = sklearn.model_selection.GridSearchCV(
                model_config['estimator'],
                model_config['params'],
                cv=3,  # Reduced from 5 to 3 folds for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(train_data, train_target)

            print(f"{model_config['name']} - Best CV score: {grid_search.best_score_}")
            print(f"{model_config['name']} - Best params: {grid_search.best_params_}")

            test_score = grid_search.score(test_data, test_target)
            print(f"{model_config['name']} - Test accuracy: {test_score}")

            if test_score > best_score:
                best_score = test_score
                best_model = grid_search.best_estimator_
                best_model_name = model_config['name']

        print(f"Best model: {best_model_name}")
        print(f"Best test accuracy: {best_score}")

        model = best_model
        model.fit(train.data, train.target)

        final_train_accuracy = model.score(train.data, train.target)
        print(f"Final training accuracy (on full dataset): {final_train_accuracy:.4f}")

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate predictions with the test set
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
