from torch import nn
import torch

class CBOW(nn.Module):
    def __init__(self, word_len, emoji_len, word_embeddings=None, freeze_pretrained_words=True, emb_dim=50, window=4, hidden_size=10):
        super(CBOW, self).__init__()

        self.word_len = word_len
        self.emoji_len = emoji_len

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

        self.emoji_embeddings = nn.Embedding(emoji_len, self.emb_dim)
        temp_state2 = self.emoji_embeddings.state_dict()
        temp_state2['weight'][0] = torch.zeros(self.emb_dim)
        self.emoji_embeddings.load_state_dict(temp_state2)

        assert(window > 0)
        self.window = window
        self.learning_layer1 = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_size),
            nn.Tanh()
        )
        self.learning_layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, word_len + emoji_len - 1),
            nn.Tanh(), 
            nn.Softmax(dim=1)
        )

    def forward(self, X: torch.Tensor):
        # Assuming this is data loaded
        zeroes = torch.zeros_like(X)
        zeroed_out_emojis = torch.where(X < 0, zeroes, X)
        zeroed_out_words = torch.where(X < 0, torch.negative(X), zeroes)
        words = self.word_embeddings(zeroed_out_emojis)
        emojis = self.emoji_embeddings(zeroed_out_words)
        embedded = torch.add(emojis, words)
        #samples = torch.flatten(embedded, start_dim=1)
        out1 = self.learning_layer1(embedded)
        out2 = torch.mean(out1, dim=1)
        out3 = self.learning_layer2(out2)

        return out3
    
    def get_word_embedding(self, word_num):
        if word_num < 0:
            return self.emoji_embeddings(-word_num)
        return self.word_embeddings(word_num)
    
    def predict(self, X: torch.tensor):
        if X.ndim == 1:
            results = torch.argmax(X)
        else:
            results = torch.argmax(X, dim=1)
        results2 = results - ((self.word_len + self.emoji_len - 1) * torch.ones_like(results))
        return torch.where(results >= self.word_len, results2, results)