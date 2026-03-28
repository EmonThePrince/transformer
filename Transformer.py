import tools
import torch


class DecoderOnlyTransformerLayer:
    def __init__(self, embed_dim, n_head, max_seq_len=1024, dropout=0.0):
        
        self.attention = tools.MultiHeadAttention(embed_dim, n_head, is_causal=True, max_seq_len=max_seq_len, dropout=dropout)
        self.feedforward = tools.FeedForward(embed_dim, embed_dim * 4, dropout)
        self.layernorm1 = tools.LayerNorm(embed_dim)
        self.layernorm2 = tools.LayerNorm(embed_dim)
        self.dropout = tools.Dropout(dropout)
        self.training = True
        
    def parameters(self):
         return self.attention.parameters() + self.feedforward.parameters() + self.layernorm1.parameters() + self.layernorm2.parameters()
    
    def __call__(self, x):
        x = x + self.dropout(self.attention(self.layernorm1(x)))
        x = x + self.dropout(self.feedforward(self.layernorm2(x)))
        return x
    
    def train(self):
        self.training = True
        self.attention.train()
        self.dropout.train()
        self.feedforward.train()
    
    def eval(self):
        self.training = False
        self.attention.eval()
        self.dropout.eval()
        self.feedforward.eval()

    def to(self, device):
        self.attention.to(device)
        self.feedforward.to(device)
        self.layernorm1.to(device)
        self.layernorm2.to(device)
        self.dropout.to(device)
        return self

class DecoderOnlyTransformer:
    def __init__(self, vocab_size, embed_dim, n_head, n_layer, max_seq_len=1024, dropout=0.0, padding_idx=None):
        self.token_embedding = tools.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_embedding = tools.PositionalEncoding(embed_dim, max_seq_len)
        self.layers = [DecoderOnlyTransformerLayer(embed_dim, n_head, max_seq_len, dropout) for _ in range(n_layer)]
        self.ln_f = tools.LayerNorm(embed_dim)
        self.padding_idx = padding_idx
        
        # self.output_proj = tools.Linear(embed_dim, vocab_size)
        # self.output_proj.w = self.token_embedding.embedding
        
        
        self.output_proj = tools.Linear(embed_dim, vocab_size)
        self.training = True
        self.dropout = tools.Dropout(dropout)
    
    def __call__(self, x, y = None):
        key_padding_mask = None
        if self.padding_idx is not None:
            key_padding_mask = (x == self.padding_idx)

        tok = self.token_embedding(x)
        pos = self.pos_embedding(tok)  
        x = self.dropout(tok + pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.output_proj(x)

        if y is None:
            loss = None
        else:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            return logits, loss
        
        return logits
    
    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = tools.Softmax(dim=-1)(logits) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def parameters(self):
        params = []
        params += self.token_embedding.parameters()
        params += self.pos_embedding.parameters()
        for layer in self.layers:
            params += layer.parameters()
        params += self.ln_f.parameters()
       
        # if self.output_proj.w is not self.token_embedding.embedding:
        #     params.append(self.output_proj.w)
        # if self.output_proj.b is not None:
        #     params.append(self.output_proj.b)
        
        params += self.output_proj.parameters()
        return params
    
    def state_dict(self):
        
        state = {}

        # embeddings
        state["token_embedding.embedding"] = self.token_embedding.embedding.detach().clone()
        state["pos_embedding.embedding"] = self.pos_embedding.embedding.detach().clone()

        # transformer blocks
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"

            # attention projections
            state[f"{prefix}.attention.qkv_proj.w"] = layer.attention.qkv_proj.w.detach().clone()
            if layer.attention.qkv_proj.b is not None:
                state[f"{prefix}.attention.qkv_proj.b"] = layer.attention.qkv_proj.b.detach().clone()
            state[f"{prefix}.attention.out_proj.w"] = layer.attention.out_proj.w.detach().clone()
            if layer.attention.out_proj.b is not None:
                state[f"{prefix}.attention.out_proj.b"] = layer.attention.out_proj.b.detach().clone()

            # feedforward
            state[f"{prefix}.feedforward.linear1.w"] = layer.feedforward.linear1.w.detach().clone()
            if layer.feedforward.linear1.b is not None:
                state[f"{prefix}.feedforward.linear1.b"] = layer.feedforward.linear1.b.detach().clone()
            state[f"{prefix}.feedforward.linear2.w"] = layer.feedforward.linear2.w.detach().clone()
            if layer.feedforward.linear2.b is not None:
                state[f"{prefix}.feedforward.linear2.b"] = layer.feedforward.linear2.b.detach().clone()

            # layer norms
            state[f"{prefix}.layernorm1.gamma"] = layer.layernorm1.gamma.detach().clone()
            state[f"{prefix}.layernorm1.beta"] = layer.layernorm1.beta.detach().clone()
            state[f"{prefix}.layernorm2.gamma"] = layer.layernorm2.gamma.detach().clone()
            state[f"{prefix}.layernorm2.beta"] = layer.layernorm2.beta.detach().clone()

        # final layer norm
        state["ln_f.gamma"] = self.ln_f.gamma.detach().clone()
        state["ln_f.beta"] = self.ln_f.beta.detach().clone()

        # output projection
        state["output_proj.w"] = self.output_proj.w.detach().clone()
        if self.output_proj.b is not None:
            state["output_proj.b"] = self.output_proj.b.detach().clone()

        return state
    
    def load_state_dict(self, state_dict):
        
        # embeddings
        self.token_embedding.embedding.data.copy_(
            state_dict["token_embedding.embedding"].to(self.token_embedding.embedding.device)
        )
        self.pos_embedding.embedding.data.copy_(
            state_dict["pos_embedding.embedding"].to(self.pos_embedding.embedding.device)
        )

        # transformer blocks
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"

            # attention projections
            layer.attention.qkv_proj.w.data.copy_(
                state_dict[f"{prefix}.attention.qkv_proj.w"].to(layer.attention.qkv_proj.w.device)
            )
            if layer.attention.qkv_proj.b is not None and f"{prefix}.attention.qkv_proj.b" in state_dict:
                layer.attention.qkv_proj.b.data.copy_(
                    state_dict[f"{prefix}.attention.qkv_proj.b"].to(layer.attention.qkv_proj.b.device)
                )
            layer.attention.out_proj.w.data.copy_(
                state_dict[f"{prefix}.attention.out_proj.w"].to(layer.attention.out_proj.w.device)
            )
            if layer.attention.out_proj.b is not None and f"{prefix}.attention.out_proj.b" in state_dict:
                layer.attention.out_proj.b.data.copy_(
                    state_dict[f"{prefix}.attention.out_proj.b"].to(layer.attention.out_proj.b.device)
                )

            # feedforward
            layer.feedforward.linear1.w.data.copy_(
                state_dict[f"{prefix}.feedforward.linear1.w"].to(layer.feedforward.linear1.w.device)
            )
            if layer.feedforward.linear1.b is not None and f"{prefix}.feedforward.linear1.b" in state_dict:
                layer.feedforward.linear1.b.data.copy_(
                    state_dict[f"{prefix}.feedforward.linear1.b"].to(layer.feedforward.linear1.b.device)
                )
            layer.feedforward.linear2.w.data.copy_(
                state_dict[f"{prefix}.feedforward.linear2.w"].to(layer.feedforward.linear2.w.device)
            )
            if layer.feedforward.linear2.b is not None and f"{prefix}.feedforward.linear2.b" in state_dict:
                layer.feedforward.linear2.b.data.copy_(
                    state_dict[f"{prefix}.feedforward.linear2.b"].to(layer.feedforward.linear2.b.device)
                )

            # layer norms
            layer.layernorm1.gamma.data.copy_(
                state_dict[f"{prefix}.layernorm1.gamma"].to(layer.layernorm1.gamma.device)
            )
            layer.layernorm1.beta.data.copy_(
                state_dict[f"{prefix}.layernorm1.beta"].to(layer.layernorm1.beta.device)
            )
            layer.layernorm2.gamma.data.copy_(
                state_dict[f"{prefix}.layernorm2.gamma"].to(layer.layernorm2.gamma.device)
            )
            layer.layernorm2.beta.data.copy_(
                state_dict[f"{prefix}.layernorm2.beta"].to(layer.layernorm2.beta.device)
            )

        # final layer norm
        self.ln_f.gamma.data.copy_(
            state_dict["ln_f.gamma"].to(self.ln_f.gamma.device)
        )
        self.ln_f.beta.data.copy_(
            state_dict["ln_f.beta"].to(self.ln_f.beta.device)
        )

        # output projection
        self.output_proj.w.data.copy_(
            state_dict["output_proj.w"].to(self.output_proj.w.device)
        )
        if self.output_proj.b is not None and "output_proj.b" in state_dict:
            self.output_proj.b.data.copy_(
                state_dict["output_proj.b"].to(self.output_proj.b.device)
            )

        return self
    
    def train(self):
        self.training = True
        self.token_embedding.train()
        self.pos_embedding.train()
        for layer in self.layers:
            layer.train()
        self.ln_f.train()
        self.output_proj.train()
        self.dropout.train()

    def eval(self):
        self.training = False
        self.token_embedding.eval()
        self.pos_embedding.eval()
        for layer in self.layers:
            layer.eval()
        self.ln_f.eval()
        self.output_proj.eval()
        self.dropout.eval()

    def to(self, device):
        self.token_embedding.to(device)
        self.pos_embedding.to(device)
        for layer in self.layers:
            layer.to(device)
        self.ln_f.to(device)
        self.output_proj.to(device)
        
        # self.output_proj.w = self.token_embedding.embedding
        return self

    def backward(self, loss):
        loss.backward()

    def zero_grad(self, set_to_none=True):
        seen = set()
        for param in self.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
