import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm1d: 
  def __init__(self, dim, device, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim).to(device)
    self.beta = torch.zeros(dim).to(device)

  def __call__(self, x):
    xmean = x.mean(-1, keepdim=True) 
    xvar = x.var(-1, keepdim=True) 
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.nn1 = nn.Linear(dim, 4*dim)
        self.nn2 = nn.Linear(4*dim, dim)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = self.dropout(self.nn2(x))
        return x
      
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_dim, context_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B,T,C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scaled_dot_prod = (Q@torch.transpose(K,1,2))/(C**0.5)
        mask = scaled_dot_prod.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = F.softmax(mask, dim=-1)
        outputs = attention@V
        return outputs
      
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_size):
        super().__init__()
        self.attn_heads = nn.ModuleList()
        head_dim = embedding_dim//num_heads
        for i in range(num_heads):
            self.attn_heads.append(SelfAttention(embedding_dim, head_dim, context_size))
    
    def forward(self, x):
        out = []
        for attn_head in self.attn_heads:
            out.append(attn_head(x))
        return torch.cat(out, dim=-1)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_size, device):
        super().__init__()
        self.masked_multihead_attn = MultiHeadSelfAttention(embedding_dim, num_heads, context_size)
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm1 = LayerNorm1d(embedding_dim, device)
        self.layer_norm2 = LayerNorm1d(embedding_dim, device)
        
    def forward(self, x):
        x = x + self.masked_multihead_attn(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, num_heads, num_blocks, device):
        super().__init__()
        self.device = device
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(context_size, embedding_dim)
        self.tranformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.tranformer_blocks.append(TransformerBlock(embedding_dim, num_heads, context_size, device))
        self.layer_norm = LayerNorm1d(embedding_dim, device)
        self.linear_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        B, C = x.shape
        inp_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(torch.arange(C, device = self.device))
        x = inp_emb+pos_emb
        for block in self.tranformer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.linear_projection(x)
        return x