import torch
from torch.utils.data import DataLoader
from cbow import CBOW
from emoji_dataset import CBOWDataset, SkipgramDataset, NGramDataset
import argparse
import gensim
import gensim.downloader as api

import pandas as pd
from matplotlib import pyplot as plt

from ngram import NGram

parser = argparse.ArgumentParser(
    prog='Emoji Embeddings',
    description='Learning Emoji Embeddings!',
    epilog='Text at the bottom of help')

parser.add_argument('-w', '--window', default=4)
parser.add_argument('-g', '--gpu', default=True)
parser.add_argument('-e', '--emojiEmbeddings', default=None)
parser.add_argument('-l', '--learningrate', default=0.1)

args = parser.parse_args()

window = args.window

# Determine whether to use the gpu
if args.gpu:
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    gpu = torch.device('cpu')

# Load Embeddings
word_model = None
if args.embeddings == 'word2vec':
    word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin')
    emb_dim = len(word_model.vectors[0])
    assert (emb_dim == 300)
elif args.embeddings == 'glove':
    assert (word_model is None)
    word_model = api.load("glove-wiki-gigaword-50")
    emb_dim = len(word_model.vectors[0])
else:
    assert (word_model is None)
    emb_dim = 50

# Load Emoji2Vec is neccessary (WIP)
emoji_model = gensim.models.KeyedVectors.load_word2vec_format(args.emojiEmbeddings, binary=False)

test_dataset = NGramDataset('.\\data\\test.csv', window_size=window, device=gpu, emoji_windows_only=False,
                            word_embeddings=word_model, emoji_embeddings=emoji_model)
train_dataset = NGramDataset('.\\data\\train.csv', window_size=window, device=gpu, emoji_windows_only=False,
                             word_embeddings=word_model, emoji_embeddings=emoji_model)

word_weights_test = None
if not word_weights_test is None:
    word_list = []
    for i in range(test_dataset.dict_index):
        word = test_dataset.index2word[i]
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights_test = torch.stack(word_list)

emoji_weights_test = None
if args.emoji2vec:
    emoji_list = []
    for i in range(test_dataset.emoji_index):
        emoji = test_dataset.index2word[-i]
        if emoji in emoji_model.index_to_key:
            emoji_list.append(torch.tensor(emoji_model[emoji]))
        else:
            emoji_list.append(torch.zeros(emb_dim))
    emoji_weights_test = torch.stack(emoji_list)

word_weights_train = None
if not word_weights_train is None:
    word_list = []
    for i in range(train_dataset.dict_index):
        word = train_dataset.index2word[i]
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights_train = torch.stack(word_list)

emoji_weights_train = None
if args.emoji2vec:
    emoji_list = []
    for i in range(train_dataset.emoji_index):
        emoji = train_dataset.index2word[-i]
        if emoji in emoji_model.index_to_key:
            emoji_list.append(torch.tensor(emoji_model[emoji]))
        else:
            emoji_list.append(torch.zeros(emb_dim))
    emoji_weights_train = torch.stack(emoji_list)

model = NGram(train_dataset.dict_index, train_dataset.emoji_index, window=window, emb_dim=emb_dim,
              word_embeddings=word_weights_train, emoji_embeddings=emoji_weights_train).to(gpu)

# Setup optimizer
#
max_iter = 1000

learn_rate = args.learningrate
batch_size = 16
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()

data_len = len(train_dataset)
batch_count = (data_len - 1) // batch_size + 1

accuracies = []
losses = []

# Initial no training eval
model.eval()
criterion.eval()
correct_list = []
loss_list = []
split = 4
split_size = (data_len + split - 1) // split
for j in range(split):
    # if args.gpu and torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    batch_start = j * split_size
    batch_end = min(data_len, (j + 1) * split_size)
    batch_slice = slice(batch_start, batch_end)
    res = model(train_dataset[batch_slice]).to(gpu)
    preds = model.predict(res).to(gpu)
    actual = train_dataset.getWord(batch_slice)
    actual2 = train_dataset.getWordPos(batch_slice)
    correct = torch.where(preds == actual, 1, 0)
    loss_list.append(criterion(res, actual2).item() * (batch_end - batch_start))
    correct_list.append(torch.sum(correct).item())
loss = sum(loss_list) / data_len
accuracy = sum(correct_list) / data_len
accuracies.append(accuracy)
losses.append(loss)
print("Iteration:", 0, "Accuracy:", accuracy, "Loss:", loss)
criterion.train()
model.train()

print("Starting Training")
for i in range(1, max_iter + 1):
    # Train for this iteration in batches
    for j in range(0, batch_count):
        batch_start = j * batch_size
        batch_end = min(data_len, (j + 1) * batch_size)
        model.train()
        optimizer.zero_grad()
        pred = model(train_dataset[batch_start:batch_end].to(gpu))
        actual = train_dataset.getWordPos(slice(batch_start, batch_end, None))
        loss = criterion(pred, actual)
        loss.backward()
        optimizer.step()
        model.reset_zero()

    # Evaluate the model
    model.eval()
    criterion.eval()
    correct_list = []
    loss_list = []
    split = 4
    split_size = (data_len + split - 1) // split
    for j in range(split):
        # if args.gpu and torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        batch_start = j * split_size
        batch_end = min(data_len, (j + 1) * split_size)
        batch_slice = slice(batch_start, batch_end)
        res = model(train_dataset[batch_slice]).to(gpu)
        preds = model.predict(res).to(gpu)
        actual = train_dataset.getWord(batch_slice)
        actual2 = train_dataset.getWordPos(batch_slice)
        correct = torch.where(preds == actual, 1, 0)
        loss_list.append(criterion(res, actual2).item() * (batch_end - batch_start))
        correct_list.append(torch.sum(correct).item())
    loss = sum(loss_list) / data_len
    accuracy = sum(correct_list) / data_len
    accuracies.append(accuracy)
    losses.append(loss)
    print("Iteration:", i, "Accuracy:", accuracy, "Loss:", loss)
    criterion.train()
    model.train()

output_location = './results/'

(acc_fig, acc_ax) = plt.subplots()
acc_ax.set(xlabel='Epochs', ylabel='Accuracy', title='Accuracy Over Epochs for ' + args.model)
acc_ax.plot(accuracies)
acc_fig.savefig(output_location + 'accuracy-' + args.model + '-' + str(max_iter) + '-iters-' + (
    'emoji2vec' if args.emoji2vec else 'random') + '.png')

(loss_fig, loss_ax) = plt.subplots()
loss_ax.set(xlabel='Epochs', ylabel='Loss', title='Loss Over Epochs for ' + args.model)
loss_ax.plot(losses)
loss_fig.savefig(output_location + 'loss-' + args.model + '-' + str(max_iter) + '-iters-' + (
    'emoji2vec' if args.emoji2vec else 'random') + '.png')


print('Testing')


test_data_len = len(test_dataset)
batch_count = test_data_len

accuracies = []
losses = []

# Initial no training eval
model.eval()
criterion.eval()
correct_list = []
loss_list = []
split = 4
split_size = (test_data_len + split - 1) // split
for j in range(split):
    # if args.gpu and torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    batch_start = j * split_size
    batch_end = min(test_data_len, (j + 1) * split_size)
    batch_slice = slice(batch_start, batch_end)
    res = model(test_dataset[batch_slice]).to(gpu)
    preds = model.predict(res).to(gpu)
    actual = test_dataset.getWord(batch_slice)
    actual2 = test_dataset.getWordPos(batch_slice)
    correct = torch.where(preds == actual, 1, 0)
    loss_list.append(criterion(res, actual2).item() * (batch_end - batch_start))
    correct_list.append(torch.sum(correct).item())
loss = sum(loss_list) / data_len
accuracy = sum(correct_list) / data_len
accuracies.append(accuracy)
losses.append(loss)
print("Accuracy:", accuracy, "Loss:", loss)