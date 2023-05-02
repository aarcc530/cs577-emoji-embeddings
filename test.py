import torch
from torch.utils.data import DataLoader
from cbow import CBOW
from emoji_dataset import CBOWDataset, SkipgramDataset
import argparse
import gensim
import gensim.downloader as api

import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
                    prog='Emoji Embeddings',
                    description='Learning Emoji Embeddings!',
                    epilog='Text at the bottom of help')

parser.add_argument('-f', '--filename', default='./Posts.csv')
parser.add_argument('-m', '--model', choices=['cbow', 'skipgram'], default='cbow')
parser.add_argument('-w', '--window', default=4)
parser.add_argument('-E', '--embeddings', choices=['glove', 'word2vec', 'none'], default='glove')
parser.add_argument('-e', '--emoji2vec', default=False)
parser.add_argument('-g', '--gpu', default=True)
parser.add_argument('-l', '--learningrate', default=0.1)

args = parser.parse_args()

window = args.window

# Determine whether to use the gpu
if args.gpu:
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    gpu = torch.device('cpu')

# Determine the Dataset
if args.model == 'cbow':
    dataset = CBOWDataset(args.filename, window_size=window, device=gpu, emoji_windows_only=True)
elif args.model == 'skipgram':
    dataset = SkipgramDataset(args.filename, window_size=window, device=gpu)
    assert(False)



# Load Embeddings
word_model = None
if args.embeddings == 'word2vec':
    word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin')
    emb_dim = 300
elif args.embeddings == 'glove':
    assert(word_model is None)
    word_model = api.load("glove-wiki-gigaword-50")
    emb_dim = 50
else:
    assert(word_model is None)
    emb_dim = 50


# Set up the Correct Loaded Embeddings
word_weights = None
if not word_model is None:
    word_list = []
    word_dict = dataset.word2index
    for word in word_dict.keys():
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights = torch.stack(word_list)


# Load Emoji2Vec is neccessary (WIP)
if args.emoji2vec:
    assert(emb_dim == 300)
    model = gensim.models.KeyedVectors.load_word2vec_format('emoji2vec.bin')
    emoji_weights = torch.FloatTensor(model.vectors)
else:
    emoji_weights = None

# Setup Model (Skipgram WIP)
if args.model == 'cbow':
    model = CBOW(dataset.dict_index, dataset.emoji_index, window=window, emb_dim=emb_dim, word_embeddings=word_weights).to(gpu)
elif args.model == 'skipgram':
    print('Not Implemented Yet')
    assert(False)


# Setup optimizer
max_iter = 1000
learn_rate = args.learningrate
batch_size = 16
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
#criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()


data_len = len(dataset)
batch_count = (data_len -1) // batch_size + 1

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
    #if args.gpu and torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    batch_start = j * split_size
    batch_end = min(data_len, (j + 1) * split_size)
    batch_slice = slice(batch_start, batch_end)
    res = model(dataset[batch_slice]).to(gpu)
    preds = model.predict(res).to(gpu)
    actual = dataset.getWord(batch_slice)
    actual2 = dataset.getWordPos(batch_slice)
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
    for j in range (0, batch_count):
        batch_start = j * batch_size
        batch_end = min(data_len, (j + 1) * batch_size)
        model.train()
        optimizer.zero_grad()
        pred = model(dataset[batch_start:batch_end].to(gpu))
        actual = dataset.getWordPos(slice(batch_start, batch_end, None))
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
        #if args.gpu and torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        batch_start = j * split_size
        batch_end = min(data_len, (j + 1) * split_size)
        batch_slice = slice(batch_start, batch_end)
        res = model(dataset[batch_slice]).to(gpu)
        preds = model.predict(res).to(gpu)
        actual = dataset.getWord(batch_slice)
        actual2 = dataset.getWordPos(batch_slice)
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
acc_fig.savefig(output_location + 'accuracy-' + args.model + '-' + str(max_iter) + '-iters.png')

(loss_fig, loss_ax) = plt.subplots()
loss_ax.set(xlabel='Epochs', ylabel='Loss', title='Loss Over Epochs for ' + args.model)
loss_ax.plot(losses)
loss_fig.savefig(output_location + 'loss-' + args.model + '-' + str(max_iter) + '-iters.png')

pairs = []
for emoji in range(-dataset.emoji_index + 1, 1):
    pairs.append((dataset.index2word[emoji], model.get_word_embeddings(torch.tensor(emoji).to(gpu)).tolist()))


results = pd.DataFrame.from_records(pairs, columns=['Emoji', 'Embedding'])
results.to_csv(output_location + 'emoji-embeddings-' + args.model + '-' + str(max_iter) + '-iters.csv', index=False)


print("done")