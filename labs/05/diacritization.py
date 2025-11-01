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
parser.add_argument("--ngram_size", default=2, type=int, help="Size of n-gram context")
parser.add_argument("--char_ngram", default=4, type=int, help="Character n-gram size")


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


def get_word_boundaries(text, position):
    non_letters = ' \n\t.,!?;:-"\'()'
    start = position
    while start > 0 and text[start - 1] not in non_letters:
        start -= 1

    end = position
    while end < len(text) - 1 and text[end + 1] not in non_letters:
        end += 1

    return start, end + 1


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

    if ngram_vocab is not None:
        for n in range(2, char_ngram_size + 1):
            for offset in range(-n + 1, 1):
                start = position + offset
                ngram_vec = np.zeros(len(ngram_vocab))
                if start >= 0 and start + n <= len(text):
                    ngram = text[start:start + n].lower()
                    if ngram in ngram_vocab:
                        ngram_vec[ngram_vocab[ngram]] = 1
                features.append(ngram_vec)

    word_start, word_end = get_word_boundaries(text, position)
    word_len = word_end - word_start
    pos_in_word = position - word_start

    position_features = np.zeros(5)
    if word_len > 0:
        position_features[0] = pos_in_word / word_len
        position_features[1] = 1 if pos_in_word == 0 else 0
        position_features[2] = 1 if pos_in_word == word_len - 1 else 0
        position_features[3] = min(word_len / 20.0, 1.0)
        position_features[4] = abs(pos_in_word - word_len / 2) / (word_len / 2) if word_len > 1 else 0
    features.append(position_features)

    type_features = np.zeros(5)
    if position > 0:
        type_features[0] = 1 if text[position - 1] == ' ' else 0
    if position < len(text) - 1:
        type_features[1] = 1 if text[position + 1] == ' ' else 0
    type_features[2] = 1 if text[position].isupper() else 0

    nearby_chars = text[max(0, position - 3):min(len(text), position + 4)].lower()
    vowels = sum(1 for c in nearby_chars if c in 'aeiouy')
    type_features[3] = vowels / len(nearby_chars) if nearby_chars else 0
    type_features[4] = 1 - type_features[3]

    features.append(type_features)
    all_features = np.concatenate(features)
    return all_features


def build_ngram_vocabulary(text, char_ngram_size, max_ngrams=5000):
    from collections import Counter

    ngram_counts = Counter()

    for n in range(2, char_ngram_size + 1):
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n].lower()
            ngram_counts[ngram] += 1

    most_common = ngram_counts.most_common(max_ngrams)
    ngram_vocab = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
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
        ngram_vocab = build_ngram_vocabulary(train_data, args.char_ngram)

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

                model = LogisticRegression(
                    max_iter=5000,
                    random_state=args.seed,
                    C=1.0,
                    solver='lbfgs',
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

                multiclass_model = LogisticRegression(
                    max_iter=5000,
                    random_state=args.seed,
                    C=1.0,
                    solver='lbfgs',
                    class_weight='balanced',
                    multi_class='multinomial'
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
        dev_accuracy = compute_accuracy(dev_predictions, dev_target)
        print(f"Dev accuracy: {dev_accuracy:}%")


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
