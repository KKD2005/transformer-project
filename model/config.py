from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration class for transformer model"""
    vocab_size: int = 8192
    hidden_size: int = 384
    num_attention_heads: int = 6
    num_hidden_layers: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 1024
    use_causal_mask: bool = True