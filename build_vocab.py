"""Build vocabularies of amino acids and classe from datasets"""

import argparse
from collections import Counter
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for amino acids in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for labels in the dataset", type=int)
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")

# Hyper parameters for the vocab
PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'


def save_vocab_to_txt_file(vocab, txt_path):
    with open(txt_path, "w") as f:
        for token in vocab:
            f.write(token + '\n')
            

def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip())
    return i + 1


def update_label(txt_path, vocab):
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split())
    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    chars = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), chars)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/sentences.txt'), chars)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), chars)
    print("- done.")

    # Build tag vocab with train and test datasets
    print("Building tag vocabulary...")
    classes = Counter()
    size_train_tags = update_label(os.path.join(args.data_dir, 'train/labels.txt'), classes)
    size_dev_tags = update_label(os.path.join(args.data_dir, 'val/labels.txt'), classes)
    size_test_tags = update_label(os.path.join(args.data_dir, 'test/labels.txt'), classes)
    print("- done.")

    # Assert same number of examples in datasets
    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    # Only keep most frequent tokens
    chars = [tok for tok, count in chars.items() if count >= args.min_count_word]
    classes = [tok for tok, count in classes.items() if count >= args.min_count_tag]

    # Add pad tokens
    if PAD_WORD not in chars: chars.append(PAD_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(chars, os.path.join(args.data_dir, 'chars.txt'))
    save_vocab_to_txt_file(classes, os.path.join(args.data_dir, 'classes.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(chars),
        'number_of_classes': len(classes),
        'pad_word': PAD_WORD,
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
