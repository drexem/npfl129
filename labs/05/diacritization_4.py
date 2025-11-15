#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import ssl
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--n", default=7, type=int, help="N-gram window size")


class Dataset:
    # ... existing code ...

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)

            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            licence_name = name.replace(".txt", ".LICENSE")

            # Use SSL context for downloads
            with urllib.request.urlopen(url + licence_name, context=ssl_context) as response:
                with open(licence_name, 'wb') as f:
                    f.write(response.read())

            with urllib.request.urlopen(url + name, context=ssl_context) as response:
                with open("{}.tmp".format(name), 'wb') as f:
                    f.write(response.read())

            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)

            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            licence_name = name.replace(".txt", ".LICENSE")

            # Use SSL context for downloads
            with urllib.request.urlopen(url + licence_name, context=ssl_context) as response:
                with open(licence_name, 'wb') as f:
                    f.write(response.read())

            with urllib.request.urlopen(url + name, context=ssl_context) as response:
                with open("{}.tmp".format(name), 'wb') as f:
                    f.write(response.read())

            os.rename("{}.tmp".format(name), name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants



class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)

            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            licence_name = name.replace(".txt", ".LICENSE")

            with urllib.request.urlopen(url + licence_name, context=ssl_context) as response:
                with open(licence_name, 'wb') as f:
                    f.write(response.read())

            with urllib.request.urlopen(url + name, context=ssl_context) as response:
                with open("{}.tmp".format(name), 'wb') as f:
                    f.write(response.read())

            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


class ContextWindowEncoder:
    """Encodes character context windows into feature vectors."""

    def __init__(self, window_length, vocab, char_map, restorable_chars):
        self.window_length = window_length
        self.half_window = (window_length - 1) // 2
        self.vocab = vocab
        self.char_map = char_map
        self.restorable_chars = restorable_chars
        self.vocab_size = len(vocab)

    def should_process(self, character):
        """Determine if character requires diacritic restoration."""
        return character.isalpha() and character in self.restorable_chars

    def extract_features(self, context_string):
        center_idx = self.half_window
        center_character = context_string[center_idx]

        if not self.should_process(center_character):
            return None

        left_chars = []
        word_boundary_hit = False
        for idx in range(center_idx - 1, -1, -1):
            if word_boundary_hit:
                left_chars.append(' ')
            else:
                if context_string[idx].isalpha():
                    left_chars.append(context_string[idx])
                else:
                    left_chars.append(' ')
                    word_boundary_hit = True

        left_chars.reverse()

        right_chars = []
        word_boundary_hit = False
        for idx in range(center_idx + 1, len(context_string)):
            if word_boundary_hit:
                right_chars.append(' ')
            else:
                if context_string[idx].isalpha():
                    right_chars.append(context_string[idx])
                else:
                    right_chars.append(' ')
                    word_boundary_hit = True

        normalized = ''.join(left_chars) + center_character + ''.join(right_chars)

        feature_vec = np.zeros(len(normalized) * self.vocab_size, dtype=np.float32)

        for pos, char in enumerate(normalized):
            base_idx = pos * self.vocab_size
            feature_vec[base_idx + self.char_map[char]] = 1

        return feature_vec


class DiacriticPredictor:

    def __init__(self, window_size, vocab, char_map, restorable_chars, random_state=42, dict=None):
        self.encoder = ContextWindowEncoder(window_size, vocab, char_map, restorable_chars)
        self.window_size = window_size
        self.half_window = (window_size - 1) // 2
        self.vocab = vocab
        self.network = None
        self.random_state = random_state
        self.dict = dict

    def prepare_training_examples(self, input_text, gold_text):
        pad_size = self.half_window
        input_padded = ' ' * pad_size + input_text + ' ' * pad_size
        gold_padded = ' ' * pad_size + gold_text + ' ' * pad_size

        X_list = []
        y_list = []

        for position in range(pad_size, len(input_padded) - pad_size):
            window = input_padded[position - pad_size:position + 1 + pad_size]
            feature_vector = self.encoder.extract_features(window)

            if feature_vector is not None:
                X_list.append(feature_vector)
                y_list.append(self.vocab.index(gold_padded[position]))

        return np.array(X_list), np.array(y_list)

    def fit(self, training_input, training_gold):
        np.random.seed(self.random_state)

        split_point = int(0.9 * len(training_input))
        train_input = training_input[:split_point]
        train_gold = training_gold[:split_point]
        dev_input = training_input[split_point:]
        dev_gold = training_gold[split_point:]

        print("Extracting training features...", file=sys.stderr)
        X, y = self.prepare_training_examples(train_input, train_gold)

        print(f"Training neural network on {len(X)} examples...", file=sys.stderr)
        self.network = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            max_iter=50,
            random_state=self.random_state,
            verbose=True
        )
        self.network.fit(X, y)
        dev_predictions = self.predict_text(dev_input)

        correct_chars = sum(1 for p, g in zip(dev_predictions, dev_gold) if p == g)
        total_chars = len(dev_gold)
        char_accuracy = 100.0 * correct_chars / total_chars if total_chars > 0 else 0

        print(f"Dev char accuracy: {char_accuracy:}%", file=sys.stderr)

    def predict_text(self, input_text):
        pad_size = self.half_window
        input_padded = ' ' * pad_size + input_text + ' ' * pad_size
        result = []

        current_word = ''
        current_word_nodia = ''

        for position in range(pad_size, len(input_padded) - pad_size):
            curr_char = input_padded[position]

            if not curr_char.isalpha():
                if current_word_nodia and self.dict:
                    if current_word_nodia in self.dict.variants:
                        variants = self.dict.variants[current_word_nodia]
                        if current_word in variants:
                            result.append(current_word)
                        else:
                            result.append(variants[0])
                    else:
                        result.append(current_word)
                elif current_word:
                    result.append(current_word)

                result.append(curr_char)

                current_word = ''
                current_word_nodia = ''
                continue

            window = input_padded[position - pad_size:position + 1 + pad_size]
            feature_vector = self.encoder.extract_features(window)

            if feature_vector is not None:
                prediction = self.network.predict(feature_vector.reshape(1, -1))
                predicted_char = self.vocab[int(prediction)]
                current_word += predicted_char
                current_word_nodia += curr_char
            else:
                current_word += curr_char
                current_word_nodia += curr_char

        if current_word_nodia and self.dict:
            if current_word_nodia in self.dict.variants:
                variants = self.dict.variants[current_word_nodia]
                if current_word in variants:
                    result.append(current_word)
                else:
                    result.append(variants[0])
            else:
                result.append(current_word)
        elif current_word:
            result.append(current_word)

        return ''.join(result)


def main(args: argparse.Namespace) -> Optional[str]:
    n = args.n
    restorable = Dataset.LETTERS_NODIA + Dataset.LETTERS_NODIA.upper()

    _lower_chars = "aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž"
    _upper_chars = "AÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽ"
    _complete_vocab = " " + _lower_chars + _upper_chars
    alphabet = list(_complete_vocab)


    if args.predict is None:
        np.random.seed(args.seed)
        train_dataset = Dataset()
        dictionary = Dictionary()

        predictor = DiacriticPredictor(
            window_size=n,
            vocab=alphabet,
            char_map={c: i for i, c in enumerate(alphabet)},
            restorable_chars=restorable,
            random_state=args.seed,
            dict=dictionary
        )

        predictor.fit(train_dataset.data, train_dataset.target)


        model_package = {
            'predictor': predictor,
            'window_size': n,
            'dictionary': dictionary
                         }
        with lzma.open(args.model_path, "wb") as file_out:
            pickle.dump(model_package, file_out)

    else:
        test_dataset = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as file_in:
            model_package = pickle.load(file_in)

        predictor = model_package['predictor']
        output = predictor.predict_text(test_dataset.data)

        return output



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)