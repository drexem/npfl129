#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix, hstack

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--ngram_size", default=7, type=int, help="Size of n-gram context")
parser.add_argument("--char_ngram", default=5, type=int, help="Character n-gram size")
parser.add_argument("--max_ngrams", default=50000, type=int, help="Maximum n-grams in vocabulary")


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

def extract_features(text, position, ngram_size, all_chars, char_ngram_size=3, ngram_vocab=None):
    char = text[position].lower()
    features = []

    char_features = np.zeros(len(all_chars))
    if char in all_chars:
        char_features[all_chars.index(char)] = 1
    features.append(char_features)

    context_size = ngram_size // 2
    for offset in range(-context_size, context_size + 1):
        if offset == 0:
            continue
        pos = position + offset
        if 0 <= pos < len(text):
            context_char = text[pos].lower()
            context_vec = np.zeros(len(all_chars))
            if context_char in all_chars:
                context_vec[all_chars.index(context_char)] = 1
            features.append(context_vec)
        else:
            features.append(np.zeros(len(all_chars)))

    if ngram_vocab is not None and char_ngram_size >= 2:
        n = char_ngram_size
        for offset in range(-n + 1, 1):
            start = position + offset
            ngram_vec = np.zeros(len(ngram_vocab))
            if start >= 0 and start + n <= len(text):
                ngram = text[start:start + n].lower()
                if ngram in ngram_vocab:
                    ngram_vec[ngram_vocab[ngram]] = 1
            features.append(ngram_vec)

    all_features = np.concatenate(features)
    return all_features


def build_ngram_vocabulary(text, char_ngram_size, max_ngrams=5000):
    from collections import Counter

    ngram_counts = Counter()

    # Only collect n-grams of size char_ngram_size
    n = char_ngram_size
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n].lower()
        ngram_counts[ngram] += 1

    most_common = ngram_counts.most_common(max_ngrams)
    ngram_vocab = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
    print(f"Built n-gram vocabulary with {len(ngram_vocab)} {n}-grams", file=sys.stderr)
    return ngram_vocab


def predict_text(text, models, all_chars, ngram_size, char_ngram_size, letter_mapping, multiclass_models=None, ngram_vocab=None):
    predictions = list(text)
    for i, char in enumerate(text):
        char_lower = char.lower()

        if char_lower in models:
            features = extract_features(text, i, ngram_size, all_chars, char_ngram_size, ngram_vocab)
            features = features.reshape(1, -1)
            prediction = models[char_lower].predict(features)[0]

            if prediction == 1:
                dia_char = letter_mapping.get(char_lower, char_lower)
                if isinstance(dia_char, list) and multiclass_models and char_lower in multiclass_models:
                    variant_prediction = multiclass_models[char_lower].predict(features)[0]
                    dia_char = dia_char[variant_prediction] if variant_prediction < len(dia_char) else dia_char[0]
                elif isinstance(dia_char, list):
                    dia_char = dia_char[0]

                if char.isupper():
                    dia_char = dia_char.upper()

                predictions[i] = dia_char

    return ''.join(predictions)


def compute_accuracy(predictions, targets):
    correct = 0
    total = len(targets)

    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1

    return correct / total if total > 0 else 0


def compute_word_accuracy(predictions, targets):
    """Compute word-level accuracy."""
    pred_words = predictions.split()
    target_words = targets.split()

    if len(pred_words) != len(target_words):
        print(f"Warning: Different number of words - pred: {len(pred_words)}, target: {len(target_words)}", file=sys.stderr)

    correct = 0
    total = min(len(pred_words), len(target_words))

    for pred_word, target_word in zip(pred_words, target_words):
        if pred_word == target_word:
            correct += 1

    return 100.0 * correct / total if total > 0 else 0


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

        all_chars = list("abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýž .,!?;:-'\"()0123456789")

        diacritizable = list(set(Dataset.LETTERS_NODIA))
        models = {}
        ngram_vocab = build_ngram_vocabulary(train_data, args.char_ngram, args.max_ngrams)

        for letter in diacritizable:
            print(f"Training model for letter '{letter}'...")

            X_train = []
            y_train = []

            for i, char in enumerate(train_data):
                if char.lower() == letter:
                    features = extract_features(train_data, i, args.ngram_size, all_chars, args.char_ngram, ngram_vocab)
                    X_train.append(features)

                    target_char = train_target[i]
                    has_dia = (target_char != char)
                    y_train.append(1 if has_dia else 0)

            if len(X_train) > 0:
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                model = SGDClassifier(
                    max_iter=5000,
                    random_state=args.seed,
                    loss='log_loss',
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                models[letter] = model

                print(f"  Trained on {len(X_train)} examples")

        multiclass_models = {}
        for letter in ['e', 'u']:
            print(f"Training multiclass model for letter '{letter}' variants...")

            X_train = []
            y_train = []

            variant_map = {'e': {'é': 0, 'ě': 1}, 'u': {'ú': 0, 'ů': 1}}
            for i, char in enumerate(train_data):
                if char.lower() == letter:
                    target_char = train_target[i].lower()
                    if target_char in variant_map[letter]:
                        features = extract_features(train_data, i, args.ngram_size, all_chars, args.char_ngram, ngram_vocab)
                        X_train.append(features)
                        y_train.append(variant_map[letter][target_char])

            if len(X_train) > 0 and len(set(y_train)) > 1:
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                multiclass_model = SGDClassifier(
                    max_iter=5000,
                    random_state=args.seed,
                    loss='log_loss',
                    class_weight='balanced'
                )
                multiclass_model.fit(X_train, y_train)
                multiclass_models[letter] = multiclass_model

        letter_mapping = {
                'a': 'á', 'c': 'č', 'd': 'ď', 'e': ['é', 'ě'], 'i': 'í',
                'n': 'ň', 'o': 'ó', 'r': 'ř', 's': 'š', 't': 'ť',
                'u': ['ú', 'ů'], 'y': 'ý', 'z': 'ž'
            }

        model = {
            'models': models,
            'multiclass_models': multiclass_models,
            'all_chars': all_chars,
            'ngram_size': args.ngram_size,
            'char_ngram_size': args.char_ngram,
            'letter_mapping': letter_mapping,
            'ngram_vocab': ngram_vocab
        }

        print("Evaluating on dev set...")
        dev_predictions = predict_text(dev_data, models, all_chars, args.ngram_size, args.char_ngram, letter_mapping, multiclass_models, ngram_vocab)

        # Character-level accuracy
        char_accuracy = compute_accuracy(dev_predictions, dev_target)
        print(f"Dev character accuracy: {char_accuracy * 100:.2f}%", file=sys.stderr)

        # Word-level accuracy (this is what we need for the competition)
        word_accuracy = compute_word_accuracy(dev_predictions, dev_target)
        print(f"Dev WORD accuracy: {word_accuracy:.2f}% (target: 86.5%)", file=sys.stderr)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        models = model['models']
        all_chars = model['all_chars']
        ngram_size = model['ngram_size']
        char_ngram_size = model['char_ngram_size']
        letter_mapping = model['letter_mapping']
        multiclass_models = model.get('multiclass_models', None)
        ngram_vocab = model.get('ngram_vocab', None)

        predictions = predict_text(test.data, models, all_chars, ngram_size, char_ngram_size, letter_mapping, multiclass_models, ngram_vocab)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
