#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

czech_alphabet_lower = ['a', 'á', 'b', 'c', 'č', 'd', 'ď', 'e', 'é', 'ě', 'f', 'g', 'h', 'i', 'í', 'j', 'k', 'l', 'm',
                        'n', 'ň', 'o', 'ó', 'p', 'q', 'r', 'ř', 's', 'š', 't', 'ť', 'u', 'ú', 'ů', 'v', 'w', 'x', 'y',
                        'ý', 'z', 'ž']
czech_alphabet_upper = ['A', 'Á', 'B', 'C', 'Č', 'D', 'Ď', 'E', 'É', 'Ě', 'F', 'G', 'H', 'I', 'Í', 'J', 'K', 'L', 'M',
                        'N', 'Ň', 'O', 'Ó', 'P', 'Q', 'R', 'Ř', 'S', 'Š', 'T', 'Ť', 'U', 'Ú', 'Ů', 'V', 'W', 'X', 'Y',
                        'Ý', 'Z', 'Ž']
czech_alphabet = [' '] + czech_alphabet_lower + czech_alphabet_upper
alphabet_size = len(czech_alphabet)
char_to_index = {char: idx for idx, char in enumerate(czech_alphabet)}
LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"


class Dataset:
    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


def convert_to_onehot(ngram: str):
    mid = (len(ngram) - 1) // 2
    ch = ngram[mid]
    seen_whitespace_left = False
    seen_whitespace_right = False

    if ch.isalpha() and (ch in LETTERS_NODIA or ch in LETTERS_NODIA.upper()):
        left = [None] * mid
        right = [None] * mid

        # Check left side
        for i in range(mid - 1, -1, -1):
            if seen_whitespace_left:
                left[i] = ' '
            else:
                if ngram[i].isalpha():
                    left[i] = ngram[i]
                else:
                    left[i] = ' '
                    seen_whitespace_left = True

        # Check right side
        for i in range(mid + 1, len(ngram)):
            if seen_whitespace_right:
                right[mid + 1 - i] = ' '
            else:
                if ngram[i].isalpha():
                    right[mid + 1 - i] = ngram[i]
                else:
                    right[mid + 1 - i] = ' '
                    seen_whitespace_right = True

        new_ngram = ''.join(left) + ngram[mid] + ''.join(right)

        # print(f'New ngram:      {new_ngram}')

        one_hot_ngram = np.zeros((len(new_ngram) * alphabet_size,))

        for i, char in enumerate(new_ngram):
            start_idx = i * alphabet_size
            one_hot_ngram[start_idx + char_to_index[char]] = 1

        return one_hot_ngram

    else:
        return None


def process_data(train_text: str, target_text: str, n: int) -> (np.ndarray, np.ndarray):
    padding = (n - 1) // 2
    train_text = padding * " " + train_text + padding * " "
    target_text = padding * " " + target_text + padding * " "
    features = np.zeros((len(train_text), n * alphabet_size))
    targets = np.zeros((len(train_text) * alphabet_size,))
    index = 0

    for i in range(padding, len(train_text) - padding):
        ngram = train_text[i - padding:i + 1 + padding]

        # print(f'Old ngram:      {ngram}')

        onehot = convert_to_onehot(ngram)

        if onehot is not None:
            features[index, :] = onehot
            targets[index] = czech_alphabet.index(target_text[i])
            index += 1

    return features[:index], targets[:index]


def main(args: argparse.Namespace) -> Optional[str]:
    n = 7

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, train_targets = process_data(train.data, train.target, n)

        model = MLPClassifier(alpha=0.01, max_iter=10000, hidden_layer_sizes=(512,256,))
        model.fit(train_data, train_targets)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        padding = (n - 1) // 2
        test_data = padding * " " + test.data + padding * " "
        predictions = ''

        for i in range(padding, len(test_data) - padding):
            ngram = test_data[i - padding:i + 1 + padding]

            onehot = convert_to_onehot(ngram)

            if onehot is not None:
                pred = model.predict(onehot.reshape(1, -1))
                predictions += czech_alphabet[int(pred)]
            else:
                predictions += test_data[i]

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
