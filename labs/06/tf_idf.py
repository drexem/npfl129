#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import urllib.request
import warnings
from collections import Counter
import re

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from pyexpat import features

# Deliberately ignore the liblinear-is-deprecated-for-multiclass-classification warning.
warnings.filterwarnings("ignore", "Using the 'liblinear' solver.*is deprecated.")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=79, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)


    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    term_doc_counts = Counter()
    for doc in train_data:
        terms = re.findall(r'\w+', doc)
        term_doc_counts.update(terms)

    filtered_terms = [(term, count) for term, count in term_doc_counts.items() if count >= 2]
    vocabulary = {term: idx for idx, (term, count) in enumerate(filtered_terms)}


    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to the number of term occurrences but normalized to
    #   sum to 1 over all features of a document);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)

    def get_features(vocabulary, data, args):
        features = np.zeros((len(data), len(vocabulary)))

        term_doc_count = np.zeros(len(vocabulary))

        for i, doc  in enumerate(data):
            terms = re.findall(r'\w+', doc)
            terms_filtered = [term for term in terms if term in vocabulary]
            term_counts = Counter(terms_filtered)
            for term in set(terms_filtered):
                if term in vocabulary:
                    term_doc_count[vocabulary[term]] += 1
                    features[i, vocabulary[term]] = term_counts[term] / len(terms_filtered) if args.tf else 1


        return features, term_doc_count



    train_features, train_term_doc_count = get_features(vocabulary, train_data, args)

    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    def get_idf(data, term_doc_count, vocab):
        term_idfs = np.zeros(len(vocab))
        for term, idx in vocab.items():
            term_idfs[idx] = np.log(len(data)/(term_doc_count[idx]+1))
        return term_idfs

    term_idfs = get_idf(train_data, train_term_doc_count, vocabulary)
    if args.idf:
        train_features = train_features * term_idfs

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set. Note that we use this solver
    # because it is several times faster on our data than the other ones available.
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_features, train_target)

    test_features, _ = get_features(vocabulary, test_data, args)
    if args.idf:
        test_features = test_features * term_idfs

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(test_target, model.predict(test_features), average='macro')

    return 100 * f1_score


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))
