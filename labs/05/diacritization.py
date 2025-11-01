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
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--ngram_size", default=7, type=int, help="Size of n-gram context")
parser.add_argument("--char_ngram", default=3, type=int, help="Character n-gram size")


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


def extract_features(text, position, ngram_size, all_chars, char_ngram_size=3):
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

    # 3. Character n-grams (substrings)
    # Create features for character n-grams around the position
    for n in range(2, char_ngram_size + 1):
        for offset in range(-n + 1, 1):
            start = position + offset
            if start >= 0 and start + n <= len(text):
                ngram = text[start:start + n].lower()
                # Hash the n-gram to a fixed size feature space (to avoid explosion)
                ngram_hash = hash(ngram) % 1000
                ngram_vec = np.zeros(1000)
                ngram_vec[ngram_hash] = 1
                features.append(ngram_vec)
            else:
                features.append(np.zeros(1000))

    # 4. Position within word
    word_start, word_end = get_word_boundaries(text, position)
    word_len = word_end - word_start
    pos_in_word = position - word_start

    position_features = np.zeros(5)
    if word_len > 0:
        # Relative position in word
        position_features[0] = pos_in_word / word_len
        # Is first character
        position_features[1] = 1 if pos_in_word == 0 else 0
        # Is last character
        position_features[2] = 1 if pos_in_word == word_len - 1 else 0
        # Word length (normalized)
        position_features[3] = min(word_len / 20.0, 1.0)
        # Distance from middle
        position_features[4] = abs(pos_in_word - word_len / 2) / (word_len / 2) if word_len > 1 else 0
    features.append(position_features)

    # 5. Character type features
    type_features = np.zeros(5)
    if position > 0:
        type_features[0] = 1 if text[position - 1] == ' ' else 0  # After space
    if position < len(text) - 1:
        type_features[1] = 1 if text[position + 1] == ' ' else 0  # Before space
    type_features[2] = 1 if text[position].isupper() else 0  # Is uppercase

    # Count vowels and consonants in nearby context
    nearby_chars = text[max(0, position - 3):min(len(text), position + 4)].lower()
    vowels = sum(1 for c in nearby_chars if c in 'aeiouy')
    type_features[3] = vowels / len(nearby_chars) if nearby_chars else 0
    type_features[4] = 1 - type_features[3]  # Consonant ratio

    features.append(type_features)

    # Combine all features
    all_features = np.concatenate(features)
    return all_features


def create_letter_mapping():
    return {
        'a': 'á', 'c': 'č', 'd': 'ď', 'e': ['é', 'ě'], 'i': 'í',
        'n': 'ň', 'o': 'ó', 'r': 'ř', 's': 'š', 't': 'ť',
        'u': ['ú', 'ů'], 'y': 'ý', 'z': 'ž'
    }


def predict_text(text, models, scalers, all_chars, ngram_size, char_ngram_size, letter_mapping, multiclass_models=None):
    """Predict diacritics for the given text using the trained models."""
    predictions = list(text)

    for i, char in enumerate(text):
        char_lower = char.lower()

        # Check if this letter can be diacritized
        if char_lower in models:
            features = extract_features(text, i, ngram_size, all_chars, char_ngram_size)
            features = features.reshape(1, -1)

            # Scale features
            scaler = scalers[char_lower]
            features = scaler.transform(features)

            # Predict whether to add diacritics
            prediction = models[char_lower].predict(features)[0]

            if prediction == 1:
                # Add diacritics
                dia_char = letter_mapping.get(char_lower, char_lower)

                # Handle letters with multiple diacritized versions using multiclass models
                if isinstance(dia_char, list) and multiclass_models and char_lower in multiclass_models:
                    # Use the multiclass model to determine which variant
                    variant_prediction = multiclass_models[char_lower].predict(features)[0]
                    dia_char = dia_char[variant_prediction] if variant_prediction < len(dia_char) else dia_char[0]
                elif isinstance(dia_char, list):
                    # Fallback to first option
                    dia_char = dia_char[0]

                # Preserve case
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

        # Define all possible characters for encoding
        all_chars = list("abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýž .,!?;:-'\"()0123456789")

        # Get diacritizable letters
        diacritizable = list(set(Dataset.LETTERS_NODIA))

        # Train a model for each diacritizable letter
        models = {}
        scalers = {}

        for letter in diacritizable:
            print(f"Training model for letter '{letter}'...", file=sys.stderr)

            X_train = []
            y_train = []

            # Collect training data for this letter
            for i, char in enumerate(train_data):
                if char.lower() == letter:
                    features = extract_features(train_data, i, args.ngram_size, all_chars, args.char_ngram)
                    X_train.append(features)

                    # Target: 1 if diacritized, 0 if not
                    target_char = train_target[i]
                    has_dia = (target_char != char)
                    y_train.append(1 if has_dia else 0)

            if len(X_train) > 0:
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Train logistic regression model with regularization
                model = LogisticRegression(
                    max_iter=5000,
                    random_state=args.seed,
                    C=1.0,
                    solver='lbfgs',
                    class_weight='balanced'
                )
                model.fit(X_train_scaled, y_train)
                models[letter] = model
                scalers[letter] = scaler

                print(f"  Trained on {len(X_train)} examples", file=sys.stderr)

        # Train multiclass models for letters with multiple diacritized variants (e and u)
        multiclass_models = {}
        multiclass_scalers = {}

        for letter in ['e', 'u']:
            print(f"Training multiclass model for letter '{letter}' variants...", file=sys.stderr)

            X_train = []
            y_train = []

            # Mapping for variants: e->é(0), ě(1); u->ú(0), ů(1)
            variant_map = {'e': {'é': 0, 'ě': 1}, 'u': {'ú': 0, 'ů': 1}}

            # Collect training data for diacritized instances only
            for i, char in enumerate(train_data):
                if char.lower() == letter:
                    target_char = train_target[i].lower()
                    # Only include diacritized instances
                    if target_char in variant_map[letter]:
                        features = extract_features(train_data, i, args.ngram_size, all_chars, args.char_ngram)
                        X_train.append(features)
                        y_train.append(variant_map[letter][target_char])

            if len(X_train) > 0 and len(set(y_train)) > 1:  # Need at least 2 classes
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # Use the same scaler as the binary model
                scaler = scalers[letter]
                X_train_scaled = scaler.transform(X_train)

                # Train multiclass logistic regression
                multiclass_model = LogisticRegression(
                    max_iter=5000,
                    random_state=args.seed,
                    C=1.0,
                    solver='lbfgs',
                    class_weight='balanced',
                    multi_class='multinomial'
                )
                multiclass_model.fit(X_train_scaled, y_train)
                multiclass_models[letter] = multiclass_model

                print(f"  Trained on {len(X_train)} examples with {len(set(y_train))} variants", file=sys.stderr)
            else:
                print(f"  Insufficient data for multiclass model", file=sys.stderr)

        # Create letter mapping for prediction
        letter_mapping = create_letter_mapping()

        model = {
            'models': models,
            'scalers': scalers,
            'multiclass_models': multiclass_models,
            'all_chars': all_chars,
            'ngram_size': args.ngram_size,
            'char_ngram_size': args.char_ngram,
            'letter_mapping': letter_mapping
        }

        # Evaluate on dev set
        print("\nEvaluating on dev set...", file=sys.stderr)
        dev_predictions = predict_text(dev_data, models, scalers, all_chars, args.ngram_size, args.char_ngram, letter_mapping, multiclass_models)
        dev_accuracy = compute_accuracy(dev_predictions, dev_target)
        print(f"Dev accuracy (character-level): {dev_accuracy:.2%}", file=sys.stderr)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        models = model['models']
        scalers = model['scalers']
        all_chars = model['all_chars']
        ngram_size = model['ngram_size']
        char_ngram_size = model['char_ngram_size']
        letter_mapping = model['letter_mapping']
        multiclass_models = model.get('multiclass_models', None)

        # Generate predictions
        predictions = predict_text(test.data, models, scalers, all_chars, ngram_size, char_ngram_size, letter_mapping, multiclass_models)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
