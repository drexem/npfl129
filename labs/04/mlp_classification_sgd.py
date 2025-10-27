#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    target = target.reshape(-1,1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(target)
    target = enc.transform(target).toarray()

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        a = inputs @ weights[0] + biases[0]
        hidden = np.maximum(a, 0)

        logits = hidden @ weights[1] + biases[1]
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return a, hidden, output

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start:start + args.batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices]

            # Forward pass - get ALL intermediate values
            a, hidden, output = forward(batch_data)

            # Backward pass
            dL_dSoftInputs = output - batch_target

            # Output layer gradients
            dL_dWo = hidden.T @ dL_dSoftInputs
            dL_dBo = np.sum(dL_dSoftInputs, axis=0)

            # Hidden layer gradients
            dL_dh = dL_dSoftInputs @ weights[1].T
            dL_da = dL_dh * (a > 0)

            # Input layer gradients
            dL_dWi = batch_data.T @ dL_da
            dL_dBi = np.sum(dL_da, axis=0)

            weights[1] -= args.learning_rate * dL_dWo / args.batch_size
            biases[1] -= args.learning_rate * dL_dBo / args.batch_size
            weights[0] -= args.learning_rate * dL_dWi / args.batch_size
            biases[0] -= args.learning_rate * dL_dBi / args.batch_size

        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_pred = forward(train_data)[1]
        test_pred = forward(test_data)[1]

        train_accuracy = np.mean(np.argmax(train_pred, axis=1) == np.argmax(train_target, axis=1))
        test_accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_target, axis=1))

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(main_args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
