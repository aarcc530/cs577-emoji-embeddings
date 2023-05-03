import torch
from torch.utils.data import DataLoader
from cbow import CBOW
from emoji_dataset import CBOWDataset, NGramDataset
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
parser.add_argument('-E', '--embeddings', default='glove')
parser.add_argument('-i', '--iterations', type=int, default=10)
parser.add_argument('-t', '--title', default='unknown')

args = parser.parse_args()

window = args.window

# Determine whether to use the gpu
if args.gpu:
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    gpu = torch.device('cpu')

# Load Embeddings
word_model = api.load("glove-wiki-gigaword-50")
emb_dim = len(word_model.vectors[0])

# Load Emoji2Vec is neccessary (WIP)
emoji_model = None
if not args.emojiEmbeddings is None:
    emoji_model = gensim.models.KeyedVectors.load_word2vec_format(args.emojiEmbeddings, binary=False)
else:
    emoji_model = gensim.models.KeyedVectors.load_word2vec_format('.\\results\\preTrainedEmoji2Vec.txt', binary=False)

test_dataset = NGramDataset('.\\data\\test.csv', window_size=window, device=gpu, emoji_windows_only=False,
                            word_embeddings=word_model, emoji_embeddings=emoji_model)
train_dataset = NGramDataset('.\\data\\train.csv', window_size=window, device=gpu, emoji_windows_only=False,
                             word_embeddings=word_model, emoji_embeddings=emoji_model)

word_weights_test = None
if not word_model is None:
    word_list = []
    for i in range(test_dataset.dict_index):
        word = test_dataset.index2word[i]
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights_test = torch.stack(word_list)

emoji_weights_test = None
if not args.emojiEmbeddings is None:
    emoji_list = []
    for i in range(test_dataset.emoji_index):
        emoji = test_dataset.index2word[-i]
        if emoji in emoji_model:
            emoji_list.append(torch.tensor(emoji_model[emoji]))
        else:
            emoji_list.append(torch.zeros(emb_dim))
    emoji_weights_test = torch.stack(emoji_list)

word_weights_train = None
if not word_model is None:
    word_list = []
    for i in range(train_dataset.dict_index):
        word = train_dataset.index2word[i]
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights_train = torch.stack(word_list)

emoji_weights_train = None
if not args.emojiEmbeddings is None:
    emoji_list = []
    for i in range(train_dataset.emoji_index):
        emoji = train_dataset.index2word[-i]
        if emoji in emoji_model:
            emoji_list.append(torch.tensor(emoji_model[emoji]))
        else:
            emoji_list.append(torch.zeros(emb_dim))
    emoji_weights_train = torch.stack(emoji_list)

model = NGram(train_dataset.dict_index, train_dataset.emoji_index, window=window, emb_dim=emb_dim,
              word_embeddings=word_weights_train, emoji_embeddings=emoji_weights_train).to(gpu)

# Setup optimizer
#
max_iter = args.iterations

learn_rate = args.learningrate
batch_size = 16
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()

data_len_train = len(train_dataset)
data_len_test = len(test_dataset)
batch_count = (data_len_train - 1) // batch_size + 1

train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

## Initial no training eval
#model.eval()
#criterion.eval()
#correct_list_train = []
#loss_list_train = []
#correct_list_test = []
#loss_list_test = []
#split = 4
#split_size_train = (data_len_train + split - 1) // split
#split_size_test = (data_len_test + split - 1) // split
#for j in range(split):
#    # if args.gpu and torch.cuda.is_available():
#    #    torch.cuda.empty_cache()
#    batch_start = j * split_size_train
#    batch_end = min(data_len_train, (j + 1) * split_size_train)
#    batch_slice = slice(batch_start, batch_end)
#    res = model(train_dataset[batch_slice]).to(gpu)
#    preds = model.predict(res).to(gpu)
#    actual = train_dataset.getWord(batch_slice)
#    actual2 = train_dataset.getWordPos(batch_slice)
#    correct = torch.where(preds == actual, 1, 0)
#    loss_list_train.append(criterion(res, actual2).item() * (batch_end - batch_start))
#    correct_list_train.append(torch.sum(correct).item())
#loss_train = sum(loss_list_train) / data_len_train
#accuracy_train = sum(correct_list_train) / data_len_train
#train_accuracies.append(accuracy_train)
#train_losses.append(loss_train)
#for j in range(split):
#    # if args.gpu and torch.cuda.is_available():
#    #    torch.cuda.empty_cache()
#    batch_start = j * split_size_test
#    batch_end = min(data_len_test, (j + 1) * split_size_test)
#    batch_slice = slice(batch_start, batch_end)
#    res = model(test_dataset[batch_slice]).to(gpu)
#    preds = model.predict(res).to(gpu)
#    actual = test_dataset.getWord(batch_slice)
#    actual2 = test_dataset.getWordPos(batch_slice)
#    correct = torch.where(preds == actual, 1, 0)
#    loss_list_test.append(criterion(res, actual2).item() * (batch_end - batch_start))
#    correct_list_test.append(torch.sum(correct).item())
#loss_test = sum(loss_list_test) / data_len_test
#accuracy_test = sum(correct_list_test) / data_len_test
#test_accuracies.append(accuracy_test)
#test_losses.append(loss_test)
#print("Iteration:", 0, "Train Accuracy:", accuracy_train, "Train Loss:", loss_train, "Test Accuracy:", accuracy_test, "Test Loss:", loss_test)
#criterion.train()
#model.train()

print("Starting Training for", 'random' if args.emojiEmbeddings is None else args.title)
for i in range(1, max_iter + 1):
    # Train for this iteration in batches
    for j in range(0, batch_count):
        batch_start = j * batch_size
        batch_end = min(data_len_train, (j + 1) * batch_size)
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
    correct_list_train = []
    loss_list_train = []
    correct_list_test = []
    loss_list_test = []
    split = 4
    split_size_train = (data_len_train + split - 1) // split
    split_size_test = (data_len_test + split - 1) // split
    for j in range(split):
        # if args.gpu and torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        batch_start = j * split_size_train
        batch_end = min(data_len_train, (j + 1) * split_size_train)
        batch_slice = slice(batch_start, batch_end)
        res = model(train_dataset[batch_slice]).to(gpu)
        preds = model.predict(res).to(gpu)
        actual = train_dataset.getWord(batch_slice)
        actual2 = train_dataset.getWordPos(batch_slice)
        correct = torch.where(preds == actual, 1, 0)
        loss_list_train.append(criterion(res, actual2).item() * (batch_end - batch_start))
        correct_list_train.append(torch.sum(correct).item())
    loss_train = sum(loss_list_train) / data_len_train
    accuracy_train = sum(correct_list_train) / data_len_train
    train_accuracies.append(accuracy_train)
    train_losses.append(loss_train)
    for j in range(split):
        # if args.gpu and torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        batch_start = j * split_size_test
        batch_end = min(data_len_test, (j + 1) * split_size_test)
        batch_slice = slice(batch_start, batch_end)
        res = model(test_dataset[batch_slice]).to(gpu)
        preds = model.predict(res).to(gpu)
        actual = test_dataset.getWord(batch_slice)
        actual2 = test_dataset.getWordPos(batch_slice)
        correct = torch.where(preds == actual, 1, 0)
        loss_list_test.append(criterion(res, actual2).item() * (batch_end - batch_start))
        correct_list_test.append(torch.sum(correct).item())
    loss_test = sum(loss_list_test) / data_len_test
    accuracy_test = sum(correct_list_test) / data_len_test
    test_accuracies.append(accuracy_test)
    test_losses.append(loss_test)
    print("Iteration:", i, "Train Accuracy:", accuracy_train, "Train Loss:", loss_train, "Test Accuracy:", accuracy_test, "Test Loss:", loss_test)
    criterion.train()
    model.train()

output_location = './results/'

(acc_fig, acc_ax) = plt.subplots()
acc_ax.set(xlabel='Epochs', ylabel='Accuracy', title='Ngram Train Accuracy Over Epochs')
acc_ax.plot(train_accuracies)
acc_fig.savefig(output_location + 'ngram-train-accuracy-' + str(max_iter) + '-iters-' + (
    'random' if args.emojiEmbeddings is None else args.title) + '.png')

(loss_fig, loss_ax) = plt.subplots()
loss_ax.set(xlabel='Epochs', ylabel='Loss', title='Ngram Train Loss Over Epochs')
loss_ax.plot(train_losses)
loss_fig.savefig(output_location + 'ngram-train-loss-' + str(max_iter) + '-iters-' + (
    'random' if args.emojiEmbeddings is None else args.title) + '.png')

(acc_fig, acc_ax) = plt.subplots()
acc_ax.set(xlabel='Epochs', ylabel='Accuracy', title='Ngram Test Accuracy Over Epochs')
acc_ax.plot(test_accuracies)
acc_fig.savefig(output_location + 'ngram-test-accuracy-' + str(max_iter) + '-iters-' + (
    'random' if args.emojiEmbeddings is None else args.title) + '.png')

(loss_fig, loss_ax) = plt.subplots()
loss_ax.set(xlabel='Epochs', ylabel='Loss', title='Ngram Test Loss Over Epochs')
loss_ax.plot(test_losses)
loss_fig.savefig(output_location + 'ngram-test-loss-' + str(max_iter) + '-iters-' + (
    'random' if args.emojiEmbeddings is None else args.title) + '.png')


print('Testing')


test_data_len = len(test_dataset)
batch_count = test_data_len

train_accuracies = []
train_losses = []

# Initial no training eval
model.eval()
criterion.eval()
correct_list_test = []
loss_list_test = []
pred_list = []
split = 4
split_size_test = (data_len_test + split - 1) // split
for j in range(split):
    batch_start = j * split_size_test
    batch_end = min(data_len_test, (j + 1) * split_size_test)
    batch_slice = slice(batch_start, batch_end)
    res = model(test_dataset[batch_slice]).to(gpu)
    preds = model.predict(res).to(gpu)
    pred_list.append(preds)
    actual = test_dataset.getWord(batch_slice)
    actual2 = test_dataset.getWordPos(batch_slice)
    correct = torch.where(preds == actual, 1, 0)
    loss_list_test.append(criterion(res, actual2).item() * (batch_end - batch_start))
    correct_list_test.append(torch.sum(correct).item())
loss_test = sum(loss_list_test) / data_len_test
accuracy_test = sum(correct_list_test) / data_len_test
test_accuracies.append(accuracy_test)
test_losses.append(loss_test)
criterion.train()
model.train()
print("Final Test Stats for", 'random' if args.emojiEmbeddings is None else args.title, 'Accuracy:', accuracy_test, 'Loss:', loss_test)


pred = torch.concat(pred_list)

#results = pd.DataFrame.from_records(pred.tolist(), columns=['Prediction', 'Embedding'])
#results.to_csv(output_location + 'ngram-test-results-' + str(max_iter) + '-iters-' + ('random' if args.emojiEmbeddings is None else args.emojiEmbeddings) + + '.csv', index=False)

