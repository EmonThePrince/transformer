import torch
import math

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        limit = (6 / (in_features + out_features)) ** 0.5
        self.w = torch.rand(out_features, in_features) * (2 * limit) - limit
        self.w.requires_grad = True

        if bias:
            self.b = torch.rand(out_features) * (2 * limit) - limit
            self.b.requires_grad = True
        else:
            self.b = None

    def __call__(self, x):
        assert x.shape[-1] == self.w.shape[1]

        out = x @ self.w.T
        if self.b is not None:
            out = out + self.b
        return out
    
    def parameters(self):
        params = [self.w]
        if self.b is not None:
            params.append(self.b)
        return params

    def to(self, device):
        self.w = self.w.detach().to(device).requires_grad_(True)
        if self.b is not None:
            self.b = self.b.detach().to(device).requires_grad_(True)
        return self

    def train(self):
        return self

    def eval(self):
        return self

class Embedding:
    def __init__(self, vocab_size, embedding_dim, padding_idx=None):
        self.embedding = torch.randn(vocab_size, embedding_dim)
        self.embedding.requires_grad = True
        self.padding_idx = padding_idx

        if padding_idx is not None:
            self.embedding[padding_idx] = 0

            def zero_padding_grad(grad):
                grad = grad.clone()
                grad[padding_idx] = 0
                return grad

            self.embedding.register_hook(zero_padding_grad)

    def __call__(self, x):
        if self.padding_idx is not None:
            self.embedding.data[self.padding_idx] = 0

        out = self.embedding[x]
        return out

    def parameters(self):
        return [self.embedding]

    def to(self, device):
        self.embedding = self.embedding.detach().to(device).requires_grad_(True)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding[self.padding_idx].zero_()

            def zero_padding_grad(grad):
                grad = grad.clone()
                grad[self.padding_idx] = 0
                return grad

            self.embedding.register_hook(zero_padding_grad)
        return self

    def train(self):
        return self

    def eval(self):
        return self

class PositionalEncodingSinusoidal:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.pe = pe.requires_grad_(False)
    def __call__(self, x):
        T = x.shape[1]
        return x + self.pe[:T]
    def parameters(self):
        return []
    def to(self, device):
        self.pe = self.pe.to(device)
        return self
    def train(self):
        return self
    def eval(self):
        return self

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        # Learnable positional embeddings
        self.embedding = torch.randn(max_len, d_model)
        self.embedding.requires_grad = True
    
    def __call__(self, x):
        T = x.shape[1]
        return x + self.embedding[:T].to(x.device)
    
    def parameters(self):
        return [self.embedding]
    
    def to(self, device):
        self.embedding = self.embedding.detach().to(device).requires_grad_(True)
        return self
    
    def train(self):
        return self
    
    def eval(self):
        return self


class Softmax:
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, x):
        m = x.max(dim=self.dim, keepdim=True).values
        e = torch.exp(x - m)
        return e / e.sum(dim=self.dim, keepdim=True)
    
def ReLU(x):
    return torch.clamp(x, min=0)

class Dropout:
    def __init__(self, p: float = 0.1):
        self.p = p
        self.training = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0 or not self.training:
            return x
        
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        return (x * mask) / (1 - self.p)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    def parameters(self):
        return []

    def to(self, device):
        return self

class MultiHeadAttention:
    def __init__(self, embed_dim, n_head, is_causal = True, max_seq_len = 1024, dropout = 0.0):
        assert embed_dim % n_head == 0
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.is_causal = is_causal
        self.head_dim = self.embed_dim // self.n_head
        self.qkv_proj = Linear(self.embed_dim, 3*self.embed_dim, bias=False)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
        self.softmax = Softmax(dim = -1)
        self.max_seq_len = max_seq_len
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def __call__(self, x):
        B, T, _ = x.shape

        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_dim).permute(2,0,3,1,4) 
        q , k , v = qkv[0] , qkv[1], qkv[2]

        score = (q @ k.transpose(-2, -1))/(self.head_dim**0.5)

        if self.is_causal:
            score = score.masked_fill( ~self.mask[:T, :T], -1e9)

        attn = self.softmax(score)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1,2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out
    
    def parameters(self):
        return self.qkv_proj.parameters() + self.out_proj.parameters()

    def to(self, device):
        self.qkv_proj.to(device)
        self.out_proj.to(device)
        self.mask = self.mask.to(device)
        self.attn_dropout.to(device)
        self.resid_dropout.to(device)
        return self
    
    def train(self):
        self.attn_dropout.train()
        self.resid_dropout.train()

    def eval(self):
        self.attn_dropout.eval()
        self.resid_dropout.eval()

class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = torch.ones(embed_dim, requires_grad=True)
        self.beta = torch.zeros(embed_dim, requires_grad=True)

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized_x + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def to(self, device):
        # keep gamma and beta as leaf tensors with gradients
        self.gamma = self.gamma.detach().to(device).requires_grad_(True)
        self.beta = self.beta.detach().to(device).requires_grad_(True)
        return self

    def train(self):
        return self

    def eval(self):
        return self
    
class FeedForward:
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        x = self.linear1(x)
        x = ReLU(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters() + self.dropout.parameters()
    
    def train(self):
        self.dropout.train()
    def eval(self):
        self.dropout.eval()

    def to(self, device):
        self.linear1.to(device)
        self.linear2.to(device)
        self.dropout.to(device)
        return self