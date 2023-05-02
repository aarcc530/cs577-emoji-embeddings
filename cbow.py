from torch import nn
import torch

class CBOW(nn.Module):
    def __init__(self, word_len, emoji_len, emb_dim=50, word_embeddings=None, freeze_pretrained_words=True, emoji_embeddings=None, freeze_pretrained_emojis=False):
        super(CBOW, self).__init__()

        self.word_len = word_len
        self.emoji_len = emoji_len
        self.freeze_pretrain_words = freeze_pretrained_words
        self.freeze_pretrained_emojis = freeze_pretrained_emojis


        # Create/Load Word Emebeddings, zeroing out the 0/period
        if word_embeddings is None:
            assert(emb_dim > 0)
            self.emb_dim = emb_dim
            self.word_embeddings = nn.Embedding(word_len, self.emb_dim)
            self.word_embeddings.weight.data.uniform_(-1, 1)
            with torch.no_grad():
                self.word_embeddings.weight.data[0] = torch.zeros(self.emb_dim)
        else:
            self.emb_dim = len(word_embeddings[0])
            self.word_len = len(word_embeddings)

            self.word_embeddings =  nn.Embedding.from_pretrained(word_embeddings, freeze=freeze_pretrained_words)
            self.word_embeddings.weight[0] = torch.zeros(self.emb_dim)

        # Create/Load Eomji Emebeddings, zeroing out the 0/period
        if emoji_embeddings is None:
            self.emoji_embeddings = nn.Embedding(emoji_len, self.emb_dim)
            self.emoji_embeddings.weight.data.uniform_(-1, 1)
            with torch.no_grad():
                self.emoji_embeddings.weight.data[0] = torch.zeros(self.emb_dim)
        else:
            assert(self.emb_dim == len(word_embeddings[0]))
            self.eomji_len = len(word_embeddings)

            self.emoji_embeddings =  nn.Embedding.from_pretrained(emoji_embeddings, freeze=freeze_pretrained_emojis)
            self.emoji_embeddings.weight[0] = torch.zeros(self.emb_dim)


        # As far as I can tell, it should just be embed -> linear -> softmax
        self.learning_layer = nn.Sequential(
            #nn.Linear(hidden_size, word_len + emoji_len - 1),
            #nn.Linear(hidden_size * 4, word_len + emoji_len - 1),
            nn.Linear(self.emb_dim, word_len + emoji_len - 1),
            #nn.Tanh(), 
            nn.LogSoftmax(dim=1)
        )

    def reset_zero(self):
        with torch.no_grad():
            self.word_embeddings.weight.data[0] = torch.zeros(self.emb_dim)
            self.emoji_embeddings.weight.data[0] = torch.zeros(self.emb_dim)


    def forward(self, X: torch.Tensor):
        # BECAUSE IT WONT LET ME DISABLE CHANGING THIS ONE VECTOR for no good reason, we will just do it again and again
        with torch.no_grad():
            zero = torch.zeros(self.emb_dim)
            zero.requires_grad = False
            zero = zero.detach()
            self.emoji_embeddings.weight.data[0] = zero
        # This is a mess I came up with on how to word embed all of these across the two since negatives are emoji
        zeroes = torch.zeros_like(X)
        zeroed_out_emojis = torch.where(X < 0, zeroes, X)
        zeroed_out_words = torch.where(X < 0, torch.negative(X), zeroes)
        words = self.word_embeddings(zeroed_out_emojis)
        emojis = self.emoji_embeddings(zeroed_out_words)
        embedded = torch.add(emojis, words)
        #samples = torch.flatten(embedded, start_dim=1)
        #out1 = self.learning_layer1(embedded)
        #out2 = torch.mean(out1, dim=1)
        #out2 = torch.flatten(out1, start_dim=1)

        # Take Mean of input, then stick it through the learning layer
        out1 = torch.mean(embedded, dim=1)
        out2 = self.learning_layer(out1)

        return out2
    
    # Allows retrieval of embeddings
    def get_word_embeddings(self, X):
        zeroes = torch.zeros_like(X)
        zeroed_out_emojis = torch.where(X < 0, zeroes, X)
        zeroed_out_words = torch.where(X < 0, torch.negative(X), zeroes)
        words = self.word_embeddings(zeroed_out_emojis)
        emojis = self.emoji_embeddings(zeroed_out_words)
        return torch.add(words, emojis)
    

    # Converts probabilities into predictions
    def predict(self, X: torch.tensor):
        if X.ndim == 1:
            results = torch.argmax(X)
        else:
            results = torch.argmax(X, dim=1)
        results2 = results - ((self.word_len + self.emoji_len - 1) * torch.ones_like(results))
        return torch.where(results >= self.word_len, results2, results)
    

class SkipGram(nn.Module):
    def __init__(self, word_len, emoji_len, word_embeddings=None, freeze_pretrained_words=True, emb_dim=50, window=4):
        super(CBOW, self).__init__()

        self.word_len = word_len
        self.emoji_len = emoji_len


        # Create/Load Word Emebeddings, zeroing out the 0/period
        if word_embeddings is None:
            assert(emb_dim > 0)
            self.emb_dim = emb_dim
            self.word_embeddings = nn.Embedding(word_len, self.emb_dim)
            self.word_embeddings.weight.data.uniform_(-1, 1)
            with torch.no_grad():
                self.word_embeddings.weight.data[0] = torch.zeros(self.emb_dim)
                self.word_embeddings.weight[0] = self.word_embeddings.weight[0].detach()
        else:
            self.emb_dim = len(word_embeddings[0])
            self.word_len - len(word_embeddings)

            self.word_embeddings =  nn.Embedding.from_pretrained(word_embeddings, freeze=freeze_pretrained_words)
            self.word_embeddings.weight[0] = torch.zeros(self.emb_dim)


        # Create Emoji Embeddings, zeroing out 0/the period
        self.emoji_embeddings = nn.Embedding(emoji_len, self.emb_dim)
        self.emoji_embeddings.weight.data.uniform_(-1, 1)
        with torch.no_grad():
            self.emoji_embeddings.weight.data[0] = torch.zeros(self.emb_dim)
            self.emoji_embeddings.weight[0] = self.emoji_embeddings.weight[0].detach()



        # As far as I can tell, Skipgram is word -> embedding -> different layers for different context words
        self.layer_list = [
            nn.Sequential(
                nn.Linear(self.emb_dim, word_len + emoji_len - 1),
                nn.Softmax(dim=1)
            ) for i in range(window)
        ]

    def forward(self, X: torch.Tensor):
        # This is a mess I came up with on how to word embed all of these across the two since negatives are emoji
        zeroes = torch.zeros_like(X)
        zeroed_out_emojis = torch.where(X < 0, zeroes, X)
        zeroed_out_words = torch.where(X < 0, torch.negative(X), zeroes)
        words = self.word_embeddings(zeroed_out_emojis)
        emojis = self.emoji_embeddings(zeroed_out_words)
        embedded = torch.add(emojis, words)


        # Predict all the outpus words, then stack and ship
        out1 = [layer(embedded) for layer in self.layer_list]
        out2 = torch.stack(out1, dim=1)

        return out2
    
    # Allows retrieval of embeddings
    def get_word_embeddings(self, X):
        zeroes = torch.zeros_like(X)
        zeroed_out_emojis = torch.where(X < 0, zeroes, X)
        zeroed_out_words = torch.where(X < 0, torch.negative(X), zeroes)
        words = self.word_embeddings(zeroed_out_emojis)
        emojis = self.emoji_embeddings(zeroed_out_words)
        return torch.add(words, emojis)
    
    # Converts probabilities into predictions
    def predict(self, X: torch.tensor):
        if X.ndim == 1:
            results = torch.argmax(X)
        else:
            results = torch.argmax(X, dim=1)
        results2 = results - ((self.word_len + self.emoji_len - 1) * torch.ones_like(results))
        return torch.where(results >= self.word_len, results2, results)