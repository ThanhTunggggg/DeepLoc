import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils 
import logging

class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, data_dir, params):
        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)        
        
        # loading vocab (we require this to map words to their indices)
        vocab_path = os.path.join(data_dir, 'chars.txt')
        self.vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i
        
        self.pad_ind = self.vocab[self.dataset_params.pad_word]

        tags_path = os.path.join(data_dir, 'classes.txt')
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d):
        sentences = []
        labels = []

        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of UNK_WORD
                s = [self.vocab[token] for token in sentence]
                sentences.append(s)
        
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = self.tag_map[sentence]
                labels.append(l)        

        # checks to ensure there is a tag for each token
        assert len(labels) == len(sentences)

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']+1)//params.batch_size):
            # fetch sentences and tags
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])

            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))

            #batch_labels = np.eye(len(self.tag_map))[batch_tags]
            batch_labels = np.array(batch_tags)

            # copy the data to the numpy array
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                #batch_labels[j][:cur_len] = batch_tags[j]

            # since all data are indices, we convert them to torch LongTensors
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            # shift tensors to GPU if available
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            # convert them to Variables to record operations in the computational graph
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
            #logging.info(batch_data.size())
            #batch_labels.squeeze_()
    
            yield batch_data, batch_labels
