"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
from itertools import groupby
from Bio import SeqIO
import random


def load_dataset(path_fasta):
    """
    Loads dataset into memory from fasta file
    """
    fasta_sequences = SeqIO.parse(open(path_fasta),'fasta')
    dataset = []
    for fasta in fasta_sequences:
        label, sequence = fasta.description.split(" ")[1].split("-")[0], str(fasta.seq)
        dataset.append((sequence, label))
    return dataset


def save_dataset(dataset, save_dir):
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for sequence, label in dataset:
                file_sentences.write("{}\n".format(sequence))
                file_labels.write("{}\n".format(label))
    print("- done.")

def fasta_iter(path_fasta):
    """
    given a fasta file. yield tuples of header, sequence
    """
    with open(path_fasta) as fa:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        # we know they alternate.
        faiter = (x[1] for x in groupby(fa, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">"
            header = header.next()[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.next())
            yield header, seq

if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/deeploc_data.fasta'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")
    random.seed(11)
    random.shuffle(dataset)

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    val_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/train')
    save_dataset(val_dataset, 'data/val')
    save_dataset(test_dataset, 'data/test')