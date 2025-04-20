import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedQueryAttentionClassifier(nn.Module):
    def __init__(self, input_dims, embed_dim=256, num_heads=16, num_queries=50, inner_embed_dim=256):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads

        # Step 1: Embedding networks per Zᵢ
        self.embedding_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, inner_embed_dim),
                nn.Sigmoid()
            )
            for d in input_dims
        ])

        # Step 2: Shared trainable projections (Wk, Wv)
        self.Wk = nn.Parameter(torch.randn(inner_embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(inner_embed_dim, embed_dim))

        # Step 3: Learned queries (shared across batch)
        self.learned_queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Final projection & classifier
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(num_queries * embed_dim, 1)

    def forward(self, Z_list):
        batch_size = Z_list[0].size(0)

        # === Compute Keys and Values ===
        keys = []
        values = []

        for z, embed_net in zip(Z_list, self.embedding_nets):
            e = embed_net(z)                             # (batch, inner_embed_dim)
            k = e @ self.Wk                              # (batch, embed_dim)
            v = e @ self.Wv                              # (batch, embed_dim)
            keys.append(k)
            values.append(v)

        K = torch.stack(keys, dim=1)                     # (batch, 4, embed_dim)
        V = torch.stack(values, dim=1)                   # (batch, 4, embed_dim)

        # === Learned Queries ===
        Q = self.learned_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 3, embed_dim)

        # === Multi-head attention ===
        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Qh = split_heads(Q)      # (batch, heads, 3, head_dim)
        Kh = split_heads(K)      # (batch, heads, 4, head_dim)
        Vh = split_heads(V)      # (batch, heads, 4, head_dim)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, Vh)     # (batch, heads, 3, head_dim)

        # === Merge heads ===
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.num_queries, self.embed_dim)
        output = self.output_proj(attn_output)           # (batch, 3, embed_dim)

        # === Classify ===
        flat = output.view(batch_size, -1)
        logits = self.classifier(flat).squeeze(-1)       # (batch,)
        return logits