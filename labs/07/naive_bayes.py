#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.

    classes = np.unique(train_target)
    n_classes = args.classes
    n_features = train_data.shape[1]

    class_priors = np.zeros(n_classes)
    for i, cls in enumerate(classes):
        class_priors[i] = np.sum(train_target == cls) / len(train_target)


    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    if args.naive_bayes_type == "gaussian":
        class_feature_means = np.zeros((n_classes, n_features))
        class_feature_variances = np.zeros((n_classes, n_features))

        for i, cls in enumerate(classes):
            class_mask = train_target == cls
            class_data = train_data[class_mask]
            class_feature_means[i] = np.mean(class_data, axis=0)
            class_feature_variances[i] = np.var(class_data, axis=0) + args.alpha


    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.


    if args.naive_bayes_type == "multinomial":
        multi_p_kd = np.zeros((n_classes, n_features))

        for i, cls in enumerate(classes):
            class_mask = train_target == cls
            class_data = train_data[class_mask]

            feature_sums = np.sum(class_data, axis=0)
            total_count = np.sum(feature_sums)
            multi_p_kd[i] = (feature_sums + args.alpha) / (total_count + args.alpha * n_features)
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.


    if args.naive_bayes_type == "bernoulli":
        train_data = (train_data >= 8).astype(float)
        multi_p_kd = np.zeros((n_classes, n_features))

        for i, cls in enumerate(classes):
            class_mask = train_target == cls
            class_data = train_data[class_mask]

            feature_sums = np.sum(class_data, axis=0)
            multi_p_kd[i] = (feature_sums + args.alpha) / (class_data.shape[0] + 2 * args.alpha)

    n_test = test_data.shape[0]


    if args.naive_bayes_type == "bernoulli":
        bin_test_data = (test_data >= 8).astype(float)

    if args.naive_bayes_type == "gaussian":

        log_priors = np.log(class_priors).reshape(1, -1)

        test_expanded = test_data[:, np.newaxis, :]
        means_expanded = class_feature_means[np.newaxis, :, :]
        vars_expanded = class_feature_variances[np.newaxis, :, :]

        log_likelihoods = -0.5 * np.log(2 * np.pi * vars_expanded) - 0.5 * (
                    (test_expanded - means_expanded) ** 2) / vars_expanded
        class_log_probs = log_priors + np.sum(log_likelihoods, axis=2)

    elif args.naive_bayes_type == "multinomial":
        log_priors = np.log(class_priors).reshape(1, -1)
        class_log_probs = log_priors + np.dot(test_data, np.log(multi_p_kd).T)

    elif args.naive_bayes_type == "bernoulli":
        log_priors = np.log(class_priors).reshape(1, -1)

        log_p = np.log(multi_p_kd)
        log_1_minus_p = np.log(1 - multi_p_kd)

        class_log_probs = log_priors + np.dot(bin_test_data, log_p.T) + np.dot(1 - bin_test_data, log_1_minus_p.T)

    predictions = classes[np.argmax(class_log_probs, axis=1)]

    true_class_indices = np.searchsorted(classes, test_target)
    log_probabilities = class_log_probs[np.arange(n_test), true_class_indices]

    test_accuracy = np.mean(predictions == test_target)
    test_log_probability = np.sum(log_probabilities)


    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    # test_accuracy, test_log_probability = ...

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
