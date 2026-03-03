import attention
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from config import TransformerConfig


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Position-wise feed-forward network
        
        Args:
            hidden_size: Model dimension
            intermediate_size: Hidden dimension of FFN
            activation_fn: Activation function ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        """
        Complete transformer block with attention and feed-forward

        Args:
            hidden_size: Model dimension
            num_heads: Number of attention heads
            intermediate_size: FFN hidden dimension
        """
        super().__init__()

        self.norm_before_attention = attention.RMSNorm(hidden_size)
        self.attention = attention.MultiHeadAttention(hidden_size, num_heads)
        self.norm_before_ffn = attention.RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer block

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask

        Returns:
            hidden_states: Output tensor (batch_size, seq_len, hidden_size)
        """
       
        attention_norm = self.norm_before_attention.forward(hidden_states)
        attention_output, attention_weights = self.attention.forward(attention_norm, attention_mask)
        residual_connection = attention_output+hidden_states
        ffn_norm = self.norm_before_ffn.forward(residual_connection)
        feed_forward = self.ffn.forward(ffn_norm)
        residual_connection = residual_connection+feed_forward
        return residual_connection

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Causal mask of shape (1, seq_len, seq_len)
    """
    
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)  


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Complete transformer model for causal language modeling
        """
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.positional_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.transformer_layers = []
        for _ in range(self.config.num_hidden_layers):
          self.transformer_layers.append(TransformerBlock(self.config.hidden_size, self.config.num_attention_heads, self.config.intermediate_size))
        self.transformer_layers = nn.ModuleList(self.transformer_layers )
        self.final_norm = attention.RMSNorm(self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer model

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len, seq_len)

        Returns:
            hidden_states: Final hidden states (batch_size, seq_len, hidden_size)
        """
        token_embeddings = self.token_embeddings(input_ids)
        counts = np.arange(input_ids.size()[1])
        torch_counts = torch.from_numpy(counts)
        torch_counts = torch_counts.unsqueeze(0)
        input_positions = torch_counts.repeat(input_ids.size()[0],1).to(input_ids.device)
        positional_embeddings = self.positional_embeddings(input_positions)
        inputs = token_embeddings+positional_embeddings
        if (attention_mask==None and self.config.use_causal_mask):
          attention_mask = create_causal_mask(input_ids.size()[1], input_ids.device)
        for transformer_layer in self.transformer_layers:
          inputs = transformer_layer.forward(inputs, attention_mask)
        return self.final_norm.forward(inputs)

class CausalLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Causal language model with transformer backbone"""
        super().__init__()

        self.transformer = TransformerModel(config)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.transformer_config = config

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for language model

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Target labels for loss computation (batch_size, seq_len)

        Returns:
            If labels provided: (loss, logits)
            Else: logits only
        """
       
        hidden_states = self.transformer.forward(input_ids)
        logits = self.linear(hidden_states)
        if labels==None:
          return logits
        loss =nn.CrossEntropyLoss()
        subset_labels = labels[:, 1:]
        subset_logits = logits[:, :-1,:]
        subset_logits = torch.transpose(subset_logits, 1, 2)
        loss = loss(subset_logits, subset_labels)
        return loss, logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text using the language model

        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        
        for _ in range(max_new_tokens):
          logits = self.forward(input_ids, None)
          autoregressive_logits = logits[:, -1, :]
          temperature_logits = autoregressive_logits/temperature
          softmax = nn.Softmax(dim =-1)
          generation = torch.multinomial(softmax(temperature_logits), num_samples=1, replacement=True)
          input_ids = torch.cat((input_ids, generation), dim=1)
        return input_ids
