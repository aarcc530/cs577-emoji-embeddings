import torch
import pandas as pd
import numpy as np
import string
from torch.utils.data import Dataset


class EmojipastaDataset(Dataset):
    def __init__(self, dataloc, window_size=4, device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')):
        self.dict_index = 1
        self.emoji_index = 1
        self.window_size = window_size
        self.word2index = { ".": 0 }
        input = pd.read_csv(dataloc, index_col=0)

        posts = [self.split_words(sent) for sent in input['selftext']]

        windows = []
        side_len = self.window_size//2
        for post in posts:
            for i in range(side_len, len(post)-(side_len)):
                window = post[i-(side_len):i] + post[i+1:i+(side_len)+1]
                windows.append(window)


        tensors = [torch.tensor(window) for window in windows]
        #input['tensor'] = [torch.tensor(self.split_words(sent)) for sent in input['selftext']]

        self.tensor = torch.stack(tensors).to(device)
        
        self.length = len(tensors)
    

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
                if is_emoji:
                    self.word2index[word] = -self.emoji_index
                    words.append(-self.emoji_index)
                    self.emoji_index += 1
                else:
                    self.word2index[word] = self.dict_index
                    words.append(self.dict_index)
                    self.dict_index += 1

            if add_period:
                words.append(0)
        return words
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.tensor[idx]