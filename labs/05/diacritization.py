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
from sklearn.feature_extraction import DictVectorizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--window", default=4, type=int, help="Context window size")


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


def extract_features(text, window_size):
    """Extract character-level features as dictionaries."""
    features = []

    # Process each character
    for i in range(len(text)):
        char = text[i].lower()

        # Only process letters that can have diacritics
        if char not in Dataset.LETTERS_NODIA:
            continue

        feature_dict = {}

        # Context window features (character unigrams)
        for offset in range(-window_size, window_size + 1):
            pos = i + offset
            if pos < 0 or pos >= len(text):
                feature_dict[f'char_{offset}'] = '<PAD>'
            else:
                feature_dict[f'char_{offset}'] = text[pos].lower()

        # Bigrams
        for offset in range(-window_size, window_size):
            pos = i + offset
            if pos < 0 or pos >= len(text) - 1:
                feature_dict[f'bigram_{offset}'] = '<PAD>'
            else:
                feature_dict[f'bigram_{offset}'] = text[pos:pos+2].lower()

        # Trigrams around target
        for offset in range(-2, 2):
            pos = i + offset
            if pos < 0 or pos >= len(text) - 2:
                feature_dict[f'trigram_{offset}'] = '<PAD>'
            else:
                feature_dict[f'trigram_{offset}'] = text[pos:pos+3].lower()

        # Target letter
        feature_dict['letter'] = char

        # Capitalization
        feature_dict['is_upper'] = int(text[i].isupper())

        # Word position features
        word_start = i
        while word_start > 0 and text[word_start - 1].strip():
            word_start -= 1
        word_end = i
        while word_end < len(text) - 1 and text[word_end + 1].strip():
            word_end += 1

        word_length = word_end - word_start + 1
        pos_in_word = i - word_start

        feature_dict['pos_in_word'] = pos_in_word
        feature_dict['word_length'] = word_length
        feature_dict['at_word_start'] = int(pos_in_word == 0)
        feature_dict['at_word_end'] = int(pos_in_word == word_length - 1)

        features.append(feature_dict)

    return features


def extract_labels(data_text, target_text):
    """Extract labels for training."""
    labels = []

    for i in range(len(data_text)):
        char_nodia = data_text[i].lower()

        if char_nodia not in Dataset.LETTERS_NODIA:
            continue

        char_dia = target_text[i]

        # Find which diacritized version this is
        if char_dia.lower() == char_nodia:
            # No diacritic
            label_idx = 0
        else:
            # Has diacritic - find which one
            try:
                label_idx = Dataset.LETTERS_DIA.index(char_dia.lower()) + 1
            except ValueError:
                label_idx = 0

        labels.append(label_idx)

    return np.array(labels)


def apply_predictions(text, predictions):
    """Apply predicted diacritics to text."""
    result = list(text)
    pred_idx = 0

    for i in range(len(text)):
        char = text[i].lower()

        if char not in Dataset.LETTERS_NODIA:
            continue

        if pred_idx < len(predictions):
            label = predictions[pred_idx]
            pred_idx += 1

            if label > 0:
                # Apply diacritic
                dia_char = Dataset.LETTERS_DIA[label - 1]
                if text[i].isupper():
                    dia_char = dia_char.upper()
                result[i] = dia_char

    return ''.join(result)


def evaluate_accuracy(gold_text, predicted_text):
    """Calculate word accuracy."""
    gold_words = gold_text.split()
    pred_words = predicted_text.split()

    if len(gold_words) != len(pred_words):
        return 0.0

    correct = sum(1 for g, p in zip(gold_words, pred_words) if g == p)
    return correct / len(gold_words)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Split by lines instead of words
        print("Splitting into train/dev at line level...", file=sys.stderr)
        lines = train.data.split('\n')
        target_lines = train.target.split('\n')

        n_lines = len(lines)
        line_indices = np.random.permutation(n_lines)
        dev_size = int(n_lines * 0.1)
        dev_line_indices = set(line_indices[:dev_size].tolist())

        # Build train and dev texts
        train_data_lines = []
        train_target_lines = []
        dev_data_lines = []
        dev_target_lines = []

        for i in range(n_lines):
            if i in dev_line_indices:
                dev_data_lines.append(lines[i])
                dev_target_lines.append(target_lines[i])
            else:
                train_data_lines.append(lines[i])
                train_target_lines.append(target_lines[i])

        train_data_text = '\n'.join(train_data_lines)
        train_target_text = '\n'.join(train_target_lines)
        dev_data_text = '\n'.join(dev_data_lines)
        dev_target_text = '\n'.join(dev_target_lines)

        # Extract features for training
        print("Extracting train features...", file=sys.stderr)
        X_train_dict = extract_features(train_data_text, args.window)
        y_train = extract_labels(train_data_text, train_target_text)

        print(f"Train: {len(X_train_dict)} examples", file=sys.stderr)

        # Extract features for dev
        print("Extracting dev features...", file=sys.stderr)
        X_dev_dict = extract_features(dev_data_text, args.window)
        y_dev = extract_labels(dev_data_text, dev_target_text)

        print(f"Dev: {len(X_dev_dict)} examples", file=sys.stderr)

        # Vectorize features
        print("Vectorizing features...", file=sys.stderr)
        vectorizer = DictVectorizer(sparse=True)
        X_train = vectorizer.fit_transform(X_train_dict)
        X_dev = vectorizer.transform(X_dev_dict)

        print(f"Feature dimension: {X_train.shape[1]}", file=sys.stderr)

        # Train logistic regression
        print("Training model...", file=sys.stderr)
        classifier = LogisticRegression(
            max_iter=1000,
            solver='saga',
            C=2.0,
            random_state=args.seed,
            verbose=1,
            n_jobs=-1
        )
        classifier.fit(X_train, y_train)

        # Evaluate on dev set
        print("Evaluating on dev set...", file=sys.stderr)
        y_pred_dev = classifier.predict(X_dev)

        # Apply predictions to dev text
        dev_predicted = apply_predictions(dev_data_text, y_pred_dev)

        # Calculate word accuracy
        accuracy = evaluate_accuracy(dev_target_text, dev_predicted)
        print(f"Dev set word accuracy: {accuracy * 100:.2f}%", file=sys.stderr)

        # Package the model
        model = {
            'classifier': classifier,
            'vectorizer': vectorizer,
            'window_size': args.window
        }

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Extract features from test data
        X_test_dict = extract_features(test.data, model['window_size'])
        X_test = model['vectorizer'].transform(X_test_dict)

        # Predict
        y_pred = model['classifier'].predict(X_test)

        # Apply predictions to text
        predictions = apply_predictions(test.data, y_pred)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)