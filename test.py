import torch
from torch.utils.data import DataLoader
from cbow import CBOW
from emoji_dataset import CBOWDataset, SkipgramDataset
import argparse
import gensim
import gensim.downloader as api

cpu = torch.device('cpu')
#gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.device('cpu')
parser = argparse.ArgumentParser(
                    prog='Emoji Embeddings',
                    description='Learning Emoji Embeddings!',
                    epilog='Text at the bottom of help')

parser.add_argument('-f', '--filename', default='./Posts.csv')
parser.add_argument('-m', '--model', choices=['cbow', 'skipgram'], default='cbow')
parser.add_argument('-w', '--window', default=4)
parser.add_argument('-E', '--embeddings', choices=['glove', 'word2vec', 'none'], default='glove')
parser.add_argument('-e', '--emoji2vec', default=False)
parser.add_argument('-d', '--hiddenSize', default=10)
parser.add_argument('-g', '--gpu', default=False)

args = parser.parse_args()

window = args.window

if args.gpu:
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    gpu = torch.device('cpu')


if args.model == 'cbow':
    dataset = CBOWDataset(args.filename, window_size=window, device=gpu, emoji_windows_only=True)
elif args.model == 'skipgram':
    dataset = SkipgramDataset(args.filename, window_size=window, device=gpu)
    assert(False)


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

if args.emoji2vec:
    assert(emb_dim == 300)
    model = gensim.models.KeyedVectors.load_word2vec_format('emoji2vec.bin')
    emoji_weights = torch.FloatTensor(model.vectors)
else:
    emoji_weights = None


if args.model == 'cbow':
    model = CBOW(dataset.dict_index, dataset.emoji_index, window=window, emb_dim=emb_dim, word_embeddings=word_weights, hidden_size=args.hiddenSize, emoji_windows_only=True).to(gpu)
elif args.model == 'skipgram':
    print('Not Implemented Yet')
    assert(False)

# SWITCH BEFORE TURN IN, SHOULD BE OPPOSITE (testing true, full_sail False)
full_sail = False
testing = True

# This was my evaluation function so I could leave it running over night to do parts of it
max_iter = 10000
learn_rate = 1
batch_size = 4
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
#criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()


data_len = len(dataset)
#data_len = 4
batch_count = (data_len -1) // batch_size + 1

print("Starting Training")
for i in range(1, max_iter + 1):
    for j in range (0, batch_count):
        torch.cuda.empty_cache()
        batch_start = j * batch_size
        batch_end = min(data_len, (j + 1) * batch_size)
        model.train()
        optimizer.zero_grad()
        pred = model(dataset[batch_start:batch_end].to(gpu))
        #actual = dataset.getOneHot(slice(batch_start, batch_end, None)).to(gpu)
        actual = dataset.getWordPos(slice(batch_start, batch_end, None))
        loss = criterion(pred, actual)
        loss.backward()
        optimizer.step()
    model.eval()
    if args.gpu:
        torch.cuda.empty_cache()
        correct_list = []
        split = 4
        split_size = (data_len + split - 1) // split 
        for j in range(split):
            torch.cuda.empty_cache()
            batch_start = j * split_size
            batch_end = min(data_len, (j + 1) * split_size)
            preds = model.predict(model(dataset[batch_start:batch_end]).to(gpu)).to(gpu)
            #actual = dataset.getWord(slice(batch_start, batch_end)).to(gpu)
            actual = dataset.getWordPos(slice(batch_start, batch_end, None))
            correct = torch.where(preds == actual, 1, 0)
            correct_list.append( torch.sum(correct).item())
        print("Iteration:", i, "Accuracy:", sum(correct_list) / data_len)
    else:
        res = model(dataset[0:data_len])
        preds = model.predict(res)
        actual = dataset.getWord(slice(0, data_len))
        #actual2 = dataset.getOneHot(slice(batch_start, batch_end, None)).to(gpu)
        actual2 = dataset.getWordPos(slice(0, data_len))
        loss = criterion(res, actual2)
        correct = torch.where(preds == actual, 1, 0)
        if i % 4 == 0:
            print("Iteration:", i, "Accuracy:", torch.sum(correct).item() / data_len, "Loss:", loss.item())
    model.train()




print("done")