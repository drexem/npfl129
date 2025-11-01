#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--ngram_size", default=6, type=int, help="Size of n-gram context")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
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


def extract_features(text, position, ngram_size, all_chars):
    char = text[position].lower()

    char_features = np.zeros(len(all_chars))
    if char in all_chars:
        char_features[all_chars.index(char)] = 1

    context_size = ngram_size // 2
    context_features = []

    # Get context characters
    for offset in range(-context_size, context_size + 1):
        if offset == 0:
            continue
        pos = position + offset
        if 0 <= pos < len(text):
            context_char = text[pos].lower()
            context_vec = np.zeros(len(all_chars))
            if context_char in all_chars:
                context_vec[all_chars.index(context_char)] = 1
            context_features.extend(context_vec)
        else:
            context_features.extend(np.zeros(len(all_chars)))

    features = np.concatenate([char_features, context_features])
    return features


def create_letter_mapping():

    return {
        'a': 'á', 'c': 'č', 'd': 'ď', 'e': ['é', 'ě'], 'i': 'í',
        'n': 'ň', 'o': 'ó', 'r': 'ř', 's': 'š', 't': 'ť',
        'u': ['ú', 'ů'], 'y': 'ý', 'z': 'ž'
    }


def predict_text(text, models, all_chars, ngram_size, letter_mapping):
    """Predict diacritics for the given text using the trained models."""
    predictions = list(text)

    for i, char in enumerate(text):
        char_lower = char.lower()

        # Check if this letter can be diacritized
        if char_lower in models:
            features = extract_features(text, i, ngram_size, all_chars)
            features = features.reshape(1, -1)

            # Predict whether to add diacritics
            prediction = models[char_lower].predict(features)[0]

            if prediction == 1:
                # Add diacritics
                dia_char = letter_mapping.get(char_lower, char_lower)

                # Handle letters with multiple diacritized versions
                if isinstance(dia_char, list):
                    # For 'e' and 'u', use the first option for now
                    # (could be improved with additional model)
                    dia_char = dia_char[0]

                # Preserve case
                if char.isupper():
                    dia_char = dia_char.upper()

                predictions[i] = dia_char

    return ''.join(predictions)


def compute_accuracy(predictions, targets):
    """Compute the accuracy of predictions against the true targets."""
    correct = 0
    total = len(targets)

    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1

    return correct / total if total > 0 else 0


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Split data into train (90%) and dev (10%)
        data_len = len(train.data)
        train_end = int(0.9 * data_len)

        train_data = train.data[:train_end]
        train_target = train.target[:train_end]
        dev_data = train.data[train_end:]
        dev_target = train.target[train_end:]

        print(f"Dataset split: train={train_end}, dev={data_len - train_end}",
              file=sys.stderr)

        # Define all possible characters for encoding
        all_chars = list("abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýž .,!?;:-'\"()0123456789")

        # Get diacritizable letters
        diacritizable = list(set(Dataset.LETTERS_NODIA))

        # Train a model for each diacritizable letter
        models = {}

        for letter in diacritizable:
            print(f"Training model for letter '{letter}'...", file=sys.stderr)

            X_train = []
            y_train = []

            # Collect training data for this letter
            for i, char in enumerate(train_data):
                if char.lower() == letter:
                    features = extract_features(train_data, i, args.ngram_size, all_chars)
                    X_train.append(features)

                    # Target: 1 if diacritized, 0 if not
                    target_char = train_target[i]
                    has_dia = (target_char != char)
                    y_train.append(1 if has_dia else 0)

            if len(X_train) > 0:
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # Train logistic regression model
                model = LogisticRegression(max_iter=1000, random_state=args.seed)
                model.fit(X_train, y_train)
                models[letter] = model

                print(f"  Trained on {len(X_train)} examples", file=sys.stderr)

        # Create letter mapping for prediction
        letter_mapping = create_letter_mapping()

        model = {
            'models': models,
            'all_chars': all_chars,
            'ngram_size': args.ngram_size,
            'letter_mapping': letter_mapping
        }

        # Evaluate on dev set
        print("\nEvaluating on dev set...", file=sys.stderr)
        dev_predictions = predict_text(dev_data, models, all_chars, args.ngram_size, letter_mapping)
        dev_accuracy = compute_accuracy(dev_predictions, dev_target)
        print(f"Dev accuracy: {dev_accuracy:.2%}", file=sys.stderr)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        models = model['models']
        all_chars = model['all_chars']
        ngram_size = model['ngram_size']
        letter_mapping = model['letter_mapping']

        # Generate predictions
        predictions = predict_text(test.data, models, all_chars, ngram_size, letter_mapping)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
