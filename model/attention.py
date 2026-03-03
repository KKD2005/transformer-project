import torch
import torch.nn as nn
from typing import Tuple, Optional

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        RMS Normalization

        Args:
            hidden_size: The size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, hidden_size)
        """
        rms = torch.sqrt(torch.mean(torch.square(hidden_states), dim = -1, keepdim=True))
        normalize = (hidden_states/(rms+self.eps)).to(hidden_states.device)
        return self.weight * normalize

class AttentionHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        """
        Single attention head implementation

        Args:
            hidden_size: Input dimension
            head_dim: Dimension of each attention head
        """
        super().__init__()

        self.q = nn.Linear(in_features = hidden_size, out_features = head_dim, bias=False)
        self.k = nn.Linear(in_features = hidden_size, out_features = head_dim, bias=False)
        self.v = nn.Linear(in_features = hidden_size, out_features = head_dim, bias=False)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention head

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attn_mask: Attention mask (batch_size, seq_len, seq_len) - 1 for attend, 0 for mask

        Returns:
            attention_output: (batch_size, seq_len, head_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))/( (self.head_dim)**.5)

        if attn_mask!=None:
          attention_mask = torch.where(attn_mask.bool(), attention_scores, float('-inf'))
        else:
          attention_mask=attention_scores

        softmax = nn.Softmax(dim=-1)
        attention_weights = softmax(attention_mask)

        return (attention_weights@value), attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        """
        Multi-head attention implementation

        Args:
            hidden_size: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        assert ((hidden_size%num_heads)==0)
        self.attention_heads = []
        for _ in range(0, num_heads):
          self.attention_heads.append(AttentionHead(hidden_size, hidden_size//num_heads))
        self.attention_heads=  nn.ModuleList(self.attention_heads)
        self.linear = nn.Linear(in_features = hidden_size, out_features = hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (1, seq_len, seq_len)

        Returns:
            attention_output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        outputs = []
        attention_weights = []
        for attention_head in self.attention_heads:
          output, attention_weight = attention_head.forward(hidden_states, attention_mask)
          outputs.append(output)
          attention_weights.append(attention_weight)
        return self.linear(torch.cat(outputs, dim=-1)), torch.stack(attention_weights, dim=1)

