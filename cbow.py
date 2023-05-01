from torch import nn
import torch

class CBOW(nn.Module):
    def __init__(self, word_len, emoji_len, word_embeddings=None, freeze_pretrained_words=True, emb_dim=50, window=4, hidden_size=10):
        super(CBOW, self).__init__()

        self.word_len = word_len
        self.emoji_len = emoji_len

        # Create/Load Word Emebeddings, zeroing out the 0/period
        if word_embeddings is None:
            assert(emb_dim > 0)
            self.emb_dim = emb_dim

            self.word_embeddings = nn.Embedding(word_len, self.emb_dim)
            temp_state = self.word_embeddings.state_dict()
            temp_state['weight'][0] = torch.zeros(self.emb_dim)
            self.word_embeddings.load_state_dict(temp_state)
        else:
            self.emb_dim = len(word_embeddings[0])
            self.word_len - len(word_embeddings)

            self.word_embeddings =  nn.Embedding.from_pretrained(word_embeddings, freeze=freeze_pretrained_words)
            temp_state = self.word_embeddings.state_dict()
            temp_state['weight'][0] = torch.zeros(self.emb_dim)
            self.word_embeddings.load_state_dict(temp_state)

        # Create Emoji Embeddings, zeroing out 0/the period
        self.emoji_embeddings = nn.Embedding(emoji_len, self.emb_dim, sparse=True)
        temp_state2 = self.emoji_embeddings.state_dict()
        temp_state2['weight'][0] = torch.zeros(self.emb_dim)
        self.emoji_embeddings.load_state_dict(temp_state2)

        assert(window > 0)
        self.window = window


        # As far as I can tell, it should just be embed -> linear -> softmax
        self.learning_layer = nn.Sequential(
            #nn.Linear(hidden_size, word_len + emoji_len - 1),
            #nn.Linear(hidden_size * 4, word_len + emoji_len - 1),
            nn.Linear(self.emb_dim, word_len + emoji_len - 1),
            #nn.Tanh(), 
            nn.Softmax(dim=1)
        )

    def forward(self, X: torch.Tensor):
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
    def get_word_embeddings(self, words):
        emojis = self.emoji_embeddings(words)
        words = self.word_embeddings(words)
        return torch.where(words < 0, emojis, words)
    

    # Converts probabilities into predictions
    def predict(self, X: torch.tensor):
        if X.ndim == 1:
            results = torch.argmax(X)
        else:
            results = torch.argmax(X, dim=1)
        results2 = results - ((self.word_len + self.emoji_len - 1) * torch.ones_like(results))
        return torch.where(results >= self.word_len, results2, results)
    

class SkipGram(nn.Module):
    def __init__(self, word_len, emoji_len, word_embeddings=None, freeze_pretrained_words=True, emb_dim=50, window=4, hidden_size=10):
        super(CBOW, self).__init__()

        self.word_len = word_len
        self.emoji_len = emoji_len


        # Create/Load Word Emebeddings, zeroing out the 0/period
        if word_embeddings is None:
            assert(emb_dim > 0)
            self.emb_dim = emb_dim
            self.word_embeddings = nn.Embedding(word_len, self.emb_dim)
            temp_state = self.word_embeddings.state_dict()
            temp_state['weight'][0] = torch.zeros(self.emb_dim)
            self.word_embeddings.load_state_dict(temp_state)
        else:
            self.emb_dim = len(word_embeddings[0])
            self.word_len - len(word_embeddings)

            self.word_embeddings =  nn.Embedding.from_pretrained(word_embeddings, freeze=freeze_pretrained_words)
            temp_state = self.word_embeddings.state_dict()
            temp_state['weight'][0] = torch.zeros(self.emb_dim)
            self.word_embeddings.load_state_dict(temp_state)


        # Create Emoji Embeddings, zeroing out 0/the period
        self.emoji_embeddings = nn.Embedding(emoji_len, self.emb_dim)
        temp_state2 = self.emoji_embeddings.state_dict()
        temp_state2['weight'][0] = torch.zeros(self.emb_dim)
        self.emoji_embeddings.load_state_dict(temp_state2)


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
    def get_word_embeddings(self, words):
        emojis = self.emoji_embeddings(words)
        words = self.word_embeddings(words)
        return torch.where(words < 0, emojis, words)
    
    # Converts probabilities into predictions
    def predict(self, X: torch.tensor):
        if X.ndim == 1:
            results = torch.argmax(X)
        else:
            results = torch.argmax(X, dim=1)
        results2 = results - ((self.word_len + self.emoji_len - 1) * torch.ones_like(results))
        return torch.where(results >= self.word_len, results2, results)