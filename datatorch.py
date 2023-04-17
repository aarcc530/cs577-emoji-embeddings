import torch
import pandas as pd
import numpy as np
import string
from torch.utils.data import Dataset


class EmojipastaDataset(Dataset):



    def __init__(self, dataloc):
        self.dict_index = 1
        self.word2index = { ".": 0 }
        self.index_is_emoji = { 0: False }
        input = pd.read_csv(dataloc, index_col=0)
        input['tensor'] = [torch.tensor(self.split_words(sent)) for sent in input['selftext']]

        self.tensor = torch.nested.nested_tensor(list(input['tensor'].values))
        
        self.length = len(input['tensor'])


    def split_words(self, input):
        words = []
        for word in input.lower().split():
            if word == '':
                continue
            
            #Is it an emoji?
            is_emoji = (word[0] == ':')

            #Remove punctuation, but mark that there was puncuation end of sentence
            add_period = False
            while not (is_emoji and word[-1] != ':') and  word[-1] in string.punctuation:
                if word[-1] == '.':
                    add_period = True
                word = word[:-1]
                if word == '':
                    break
            if word == '':
                continue
            #Append word and add to dict if necessary
            if word in self.word2index.keys():
                words.append(self.word2index[word])
            else:
                self.word2index[word] = self.dict_index
                self.index_is_emoji[self.dict_index] = is_emoji
                words.append(self.dict_index)
                self.dict_index += 1

            if add_period:
                words.append(0)
        return words
            
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Tensor's don't like slicing
        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])
        return self.tensor[idx]