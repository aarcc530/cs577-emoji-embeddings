import torch
from torch.utils.data import DataLoader
from cbow import CBOW
from emoji_dataset import CBOWDataset, NGramDataset #, SkipgramDataset
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
parser.add_argument('-m', '--model', choices=['cbow', 'ngram'], default='cbow')
parser.add_argument('-w', '--window', type=int, default=4)
parser.add_argument('-E', '--embeddings', choices=['glove', 'word2vec', 'none'], default='glove')
parser.add_argument('-e', '--emoji2vec', action='store_true')
parser.add_argument('-eF', '--emojiFile', default='./preTrainedEmoji2Vec.txt')
parser.add_argument('-ng', '--nogpu', action='store_true')
parser.add_argument('-l', '--learningrate', type=int, default=0.1)
parser.add_argument('-uW', '--unfreezeWords', action='store_true')
parser.add_argument('-fE', '--freezeEmojis', action='store_true')
parser.add_argument('-i', '--iterations', type=int, default=500)
parser.add_argument('-b', '--batchsize', type=int, default=16)
parser.add_argument('-s', '--split', type=int, default=4)

args = parser.parse_args()

window = args.window

# Determine whether to use the gpu
if args.nogpu:
    gpu = torch.device('cpu')
else:
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Embeddings
word_model = None
if args.embeddings == 'word2vec':
    word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin')
    emb_dim = len(word_model.vectors[0])
    assert(emb_dim == 300)
elif args.embeddings == 'glove':
    assert(word_model is None)
    word_model = api.load("glove-wiki-gigaword-50")
    emb_dim = len(word_model.vectors[0])
else:
    assert(word_model is None)
    emb_dim = 50


# Load Emoji2Vec is neccessary (WIP)
emoji_model = gensim.models.KeyedVectors.load_word2vec_format(args.emojiFile, binary=False)
if args.emoji2vec:   
    assert(emb_dim == len(emoji_model.vectors[0]))



# Determine the Dataset
if args.model == 'cbow':
    dataset = CBOWDataset(args.filename, window_size=window, device=gpu, emoji_windows_only=True, word_embeddings=word_model, emoji_embeddings=emoji_model)
elif args.model == 'skipgram':
    #dataset = SkipgramDataset(args.filename, window_size=window, device=gpu)
    assert(False)
elif args.model == 'ngram':
    print('moved')
    assert(False)
    

# Set up the Correct Loaded Embeddings
word_weights = None
if not word_model is None:
    word_list = []
    for i in range(dataset.dict_index):
        word = dataset.index2word[i]
        if word in word_model:
            word_list.append(torch.tensor(word_model[word]))
        else:
            word_list.append(torch.zeros(emb_dim))
    word_weights = torch.stack(word_list)

emoji_weights = None
if args.emoji2vec:
    emoji_list = []
    for i in range(dataset.emoji_index):
        emoji = dataset.index2word[-i]
        if emoji in emoji_model:
            emoji_list.append(torch.tensor(emoji_model[emoji]))
        else:
            emoji_list.append(torch.zeros(emb_dim))
    emoji_weights = torch.stack(emoji_list)


# Setup Model 
if args.model == 'cbow':
    model = CBOW(dataset.dict_index, dataset.emoji_index, emb_dim=emb_dim, word_embeddings=word_weights, 
                 emoji_embeddings=emoji_weights, freeze_pretrained_words=(not args.unfreezeWords), freeze_pretrained_emojis=args.freezeEmojis).to(gpu)
elif args.model == 'ngram':
    print('Moved files')
    assert(False)


# Setup Conditions
max_iter = args.iterations
learn_rate = args.learningrate
batch_size = args.batchsize
split = args.split

# Setup Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
#criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()

# Setup appropriate variables
data_len = len(dataset)
batch_count = (data_len -1) // batch_size + 1
split_size = (data_len + split - 1) // split 
accuracies = []
losses = []

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



# Create Graphs and Output
output_location = './results/'

(acc_fig, acc_ax) = plt.subplots()
acc_ax.set(xlabel='Epochs', ylabel='Accuracy', title=args.model + ' Accuracy Over Epochs for ' + args.model + ' with ' + ('emoji2vec' if args.emoji2vec else 'random') + ('' if args.freezeEmojis else ' + tuning') + ' embeddings')
acc_ax.plot(accuracies)
acc_fig.savefig(output_location + 'accuracy-' + args.model + '-' + str(max_iter) + '-iters-' + ('emoji2vec' if args.emoji2vec else 'random') + '.png')

(loss_fig, loss_ax) = plt.subplots()
loss_ax.set(xlabel='Epochs', ylabel='Loss', title=args.model + ' Loss Over Epochs for ' + args.model + ' with ' + ('emoji2vec' if args.emoji2vec else 'random') + ('' if args.freezeEmojis else ' + tuning') + ' embeddings')
loss_ax.plot(losses)
loss_fig.savefig(output_location + 'loss-' + args.model + '-' + str(max_iter) + '-iters-' + ('emoji2vec' if args.emoji2vec else 'random') + ('' if args.freezeEmojis else '-tuned') + '.png')

if args.model == 'cbow':
    pairs = []
    for emoji in range(-dataset.emoji_index + 1, 1):
        pairs.append((dataset.index2word[emoji], model.get_word_embeddings(torch.tensor(emoji).to(gpu)).tolist()))


    results = pd.DataFrame.from_records(pairs, columns=['Emoji', 'Embedding'])
    results.to_csv(output_location + 'emoji-embeddings-' + args.model + '-' + str(max_iter) + '-iters-' + ('emoji2vec' if args.emoji2vec else 'random') + ('' if args.freezeEmojis else '-tuned') + '.csv', index=False)


print("done")