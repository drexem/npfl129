#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    target = np.zeros((data.shape[0], args.classes))
    for i, labels in enumerate(target_list):
        for label in labels:
            target[i, label] = 1

    # Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start:start + args.batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices]

            logits = batch_data @ weights
            predictions = 1 / (1 + np.exp(-logits))

            errors = predictions - batch_target
            gradient = (batch_data.T @ errors) / args.batch_size

            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.


        predictions = 1 / (1 + np.exp(-train_data @ weights))
        predictions = (predictions >= 0.5).astype(int)
        train_f1_micro, train_f1_macro = compute_f1_scores(predictions, train_target)

        predictions = 1 / (1 + np.exp(-test_data @ weights))
        predictions = (predictions >= 0.5).astype(int)
        test_f1_micro, test_f1_macro = compute_f1_scores(predictions, test_target)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]

def compute_f1_scores(predictions, targets):
    TP = (predictions == 1) & (targets == 1)
    FP = (predictions == 1) & (targets == 0)
    FN = (predictions == 0) & (targets == 1)

    all_tp = np.sum(TP)
    all_fp = np.sum(FP)
    all_fn = np.sum(FN)

    precision_micro = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall_micro = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

    f1_scores = []
    for c in range(targets.shape[1]):
        tp = np.sum(TP[:, c])
        fp = np.sum(FP[:, c])
        fn = np.sum(FN[:, c])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    f1_macro = np.mean(f1_scores)

    return f1_micro, f1_macro

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
