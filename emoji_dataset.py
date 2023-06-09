import torch
import pandas as pd
import numpy as np
import string
from torch.utils.data import Dataset




class CBOWDataset(Dataset):
    def __init__(self, dataloc, window_size=4, device=torch.device('cpu'), emoji_windows_only=True, word_embeddings=None, emoji_embeddings=None):
        self.EMOJI_NOT_FOUND = -2**30

        self.dict_index = 1
        self.emoji_index = 1
        self.device = device
        self.window_size = window_size
        self.word2index = { ".": 0 }
        self.index2word = { 0: "." }
        self.emoji_embeddings = emoji_embeddings
        self.word_embeddings = word_embeddings
        self.emoji_windows_only = emoji_windows_only
        input = pd.read_csv(dataloc, index_col=0)

        posts = [self.split_words(sent) for sent in input['selftext']]

        windows = []
        words = []
        side_len = self.window_size//2
        for post in posts:
            for i in range(side_len, len(post)-(side_len)):
                window = post[i-(side_len):i] + post[i+1:i+(side_len)+1]
                include_window = True
                contains_emojis = False
                if post[i] == self.EMOJI_NOT_FOUND:
                    continue
                for j in range(2 *side_len):
                    if window[j] == self.EMOJI_NOT_FOUND:
                        include_window = False
                        break
                    if window[j] < 0:
                        contains_emojis = True
                if include_window and (not emoji_windows_only or contains_emojis):
                    words.append(post[i])
                    windows.append(window)


        tensors = [torch.tensor(window) for window in windows]
        self.words = torch.tensor(words)
        #input['tensor'] = [torch.tensor(self.split_words(sent)) for sent in input['selftext']]

        self.windows = torch.stack(tensors).to(device)
        self.words = torch.tensor(words).to(device)
        
        self.length = len(tensors)
    

    def split_words(self, input):
        words = []
        for word in input.split():
            if word == '':
                continue
            
            #Is it an emoji?
            is_emoji = (word[0] == ':')

            #Remove punctuation, but mark that there was puncuation end of sentence
            add_period = False
            while not (is_emoji and word[-1] == ':') and  word[-1] in string.punctuation:
                if word[-1] == '.':
                    add_period = True
                word = word[:-1]
                if word == '':
                    break
            if word == '':
                continue
            if not is_emoji:
                word = word.lower()
            #Append word and add to dict if necessary
            if word in self.word2index.keys():
                words.append(self.word2index[word])
            else:
                if is_emoji:
                    if word not in self.emoji_embeddings:
                        words.append(self.EMOJI_NOT_FOUND)
                    else:
                        self.word2index[word] = -self.emoji_index
                        self.index2word[-self.emoji_index] = word
                        words.append(-self.emoji_index)
                        self.emoji_index += 1
                else:
                    self.word2index[word] = self.dict_index
                    self.index2word[self.dict_index] = word
                    words.append(self.dict_index)
                    self.dict_index += 1
            if add_period:
                words.append(0)
        return words
        
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.windows[idx]
    
    def getWord(self, idx):
        return self.words[idx]
    
    def getWordPos(self, idx):
        res = self.words[idx]
        res2 = (self.dict_index + self.emoji_index - 1) * torch.ones_like(res) + res
        return torch.where(res < 0, res2, res)
    
    def getOneHot(self, idx):
        res = []
        for i in range(self.length)[idx]:
            onehot = torch.zeros(self.dict_index + self.emoji_index - 1).to(self.device)
            onehot[self.words[i]] = 1
            res.append(onehot)
        return torch.stack(res).to(self.device)
    
    
class NGramDataset(Dataset):
    def __init__(self, dataloc, window_size=4, device=torch.device('cpu'), emoji_windows_only=True,
                 word_embeddings=None, emoji_embeddings=None):
        self.EMOJI_NOT_FOUND = -2 ** 30

        self.dict_index = 1
        self.emoji_index = 1
        self.device = device
        self.window_size = int(window_size)
        self.word2index = {".": 0}
        self.index2word = {0: "."}
        self.emoji_embeddings = emoji_embeddings
        self.word_embeddings = word_embeddings
        self.emoji_windows_only = emoji_windows_only
        input = pd.read_csv(dataloc, index_col=0)

        posts = [self.split_words(sent) for sent in input['selftext']]

        windows = []
        words = []
        side_len = self.window_size
        for post in posts:
            for i in range(side_len, len(post) - (side_len)):
                window = post[i - (side_len):i]
                include_window = True
                contains_emojis = False
                if post[i] == self.EMOJI_NOT_FOUND:
                    continue
                for j in range(len(window)):
                    #print(f'j = {j}, window size = {len(window)}')
                    if window[j] == self.EMOJI_NOT_FOUND:
                        include_window = False
                        break
                    if window[j] < 0:
                        contains_emojis = True
                if include_window and (not emoji_windows_only or contains_emojis):
                    words.append(post[i])
                    windows.append(window)

        tensors = [torch.tensor(window) for window in windows]
        self.words = torch.tensor(words)
        # input['tensor'] = [torch.tensor(self.split_words(sent)) for sent in input['selftext']]

        self.windows = torch.stack(tensors).to(device)
        self.words = torch.tensor(words).to(device)

        self.length = len(tensors)

    def split_words(self, input):
        words = []
        for word in input.split():
            if word == '':
                continue

            # Is it an emoji?
            is_emoji = (word[0] == ':')

            # Remove punctuation, but mark that there was puncuation end of sentence
            add_period = False
            while not (is_emoji and word[-1] == ':') and word[-1] in string.punctuation:
                if word[-1] == '.':
                    add_period = True
                word = word[:-1]
                if word == '':
                    break
            if word == '':
                continue
            if not is_emoji:
                word = word.lower()
            # Append word and add to dict if necessary
            if word in self.word2index.keys():
                words.append(self.word2index[word])
            else:
                if is_emoji:
                    if word not in self.emoji_embeddings:
                        words.append(self.EMOJI_NOT_FOUND)
                    else:
                        self.word2index[word] = -self.emoji_index
                        self.index2word[-self.emoji_index] = word
                        words.append(-self.emoji_index)
                        self.emoji_index += 1
                else:
                    self.word2index[word] = self.dict_index
                    self.index2word[self.dict_index] = word
                    words.append(self.dict_index)
                    self.dict_index += 1
            if add_period:
                words.append(0)
        return words

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.windows[idx]

    def getWord(self, idx):
        return self.words[idx]

    def getWordPos(self, idx):
        res = self.words[idx]
        res2 = (self.dict_index + self.emoji_index - 1) * torch.ones_like(res) + res
        return torch.where(res < 0, res2, res)

    def getOneHot(self, idx):
        res = []
        for i in range(self.length)[idx]:
            onehot = torch.zeros(self.dict_index + self.emoji_index - 1).to(self.device)
            onehot[self.words[i]] = 1
            res.append(onehot)
        return torch.stack(res).to(self.device)

   # def skipgram(sentences, window_size=1):
   #     skip_grams = []
   #     for i in range(window_size, len(word_sequence) - window_size):
   #         target = word_sequence[i]
   #         context = [word_sequence[i - window_size], word_sequence[i + window_size]]
   #         for w in context:
   #            skip_grams.append([target, w])

   #return skip_grams

#class SkipgramDataset(Dataset):
#    def __init__(self, dataloc, window_size=4, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#        self.dict_index = 1
#        self.emoji_index = 1
#        self.window_size = window_size
#        self.word2index = { ".": 0 }
#        input = pd.read_csv(dataloc, index_col=0)
#
#        posts = [self.split_words(sent) for sent in input['selftext']]
#
#        windows = []
#        words = []
#        side_len = self.window_size//2
#        for post in posts:
#            for i in range(side_len, len(post)-(side_len)):
#                window = post[i-(side_len):i] + post[i+1:i+(side_len)+1]
#                words.append(post[i])
#                windows.append(window)
#
#
#        tensors = [torch.tensor(window) for window in windows]
#        self.words = torch.tensor(words)
#        #input['tensor'] = [torch.tensor(self.split_words(sent)) for sent in input['selftext']]
#
#        self.context = torch.stack(tensors).to(device)
#        self.words = torch.tensor(words).to(device)
#        
#        self.length = len(tensors)
#    
#
#    def split_words(self, input):
#        words = []
#        for word in input.lower().split():
#            if word == '':
#                continue
#            
#            #Is it an emoji?
#            is_emoji = (word[0] == ':')
#
#            #Remove punctuation, but mark that there was puncuation end of sentence
#            add_period = False
#            while not (is_emoji and word[-1] != ':') and  word[-1] in string.punctuation:
#                if word[-1] == '.':
#                    add_period = True
#                word = word[:-1]
#                if word == '':
#                    break
#            if word == '':
#                continue
#            #Append word and add to dict if necessary
#            if word in self.word2index.keys():
#                words.append(self.word2index[word])
#            else:
#                if is_emoji:
#                    self.word2index[word] = -self.emoji_index
#                    words.append(-self.emoji_index)
#                    self.emoji_index += 1
#                else:
#                    self.word2index[word] = self.dict_index
#                    words.append(self.dict_index)
#                    self.dict_index += 1
#
#            if add_period:
#                words.append(0)
#        return words
#        
#            
#    def __len__(self):
#        return self.length
#
#    def __getitem__(self, idx):
#        return self.words[idx]
#    
#    def getContext(self, idx):
#        return self.context[idx]
