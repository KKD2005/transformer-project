import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))        # project root → model/
sys.path.insert(0, _here)                             # training/ → lr_schedule
sys.path.insert(0, os.path.join(_here, '..', 'data')) # data/ → generate_dataset

from dataclasses import dataclass
from typing import List
from model import config, transformer
import torch
import torch.nn as nn
from lr_schedule import get_lr_scheduler
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from generate_dataset import GSM8KDataset

@dataclass
class TrainingConfig:
    # Model hyperparameters
    vocab_size: int = 8192
    hidden_size: int = 384
    num_attention_heads: int = 6
    num_hidden_layers: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 1024
    use_causal_mask: bool = True

    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 3 #TBD
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 200

    # Paths
    output_dir: str = "./gsm9k_model"
    log_dir: str = "./logs"

_tokenizer_path = os.path.join(_here, '..', 'data', 'gsm8k_tokenizer.json')
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=_tokenizer_path,
    pad_token="<pad>",
    eos_token="<eos>",
    bos_token="<bos>",
)

training_config = TrainingConfig(vocab_size=tokenizer.vocab_size)
model_config = config.TransformerConfig(
    vocab_size=training_config.vocab_size,
    hidden_size=training_config.hidden_size,
    num_attention_heads=training_config.num_attention_heads,
    num_hidden_layers=training_config.num_hidden_layers,
    intermediate_size=training_config.intermediate_size,
    max_position_embeddings=training_config.max_position_embeddings
)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = transformer.CausalLanguageModel(model_config)
print(f"Model initialized with {count_parameters(model):,} parameters")


class TrainingMetrics:
    """Track training metrics"""
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.step = 0
    
    def update(self, loss: float, lr: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step += 1
    
    def get_avg_loss(self, last_n: int = 100):
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses[-last_n:])

def evaluate_model(model, tokenizer, test_prompts: List[str], temperature: float = 0.7):
    """Evaluate model with test prompts"""
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    print("Generating samples from trained model:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        print("-" * 40)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with different temperatures
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids, 
                max_new_tokens=150,
                temperature=temperature
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Temperature {temperature}: {generated_text}")
            print()

def save_model(model, tokenizer, save_path: str):
    """Save model and tokenizer"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    if hasattr(model, 'config'):
        torch.save(model.config.__dict__, os.path.join(save_path, "config.json"))
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

class Trainer:
    def __init__(self, model, train_dataset, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.train_dataset = train_dataset
        self.batch_size = config.batch_size


        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.total_steps = (len(self.train_dataset) // self.batch_size) * config.num_epochs
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.warmup_steps,
            self.total_steps
        )

        self.metrics = TrainingMetrics()
        self.global_step = 0
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train_step(self, batch) -> float:
        """
        Single training step

        Args:
            batch: Batch of data

        Returns:
            loss: Training loss for this step
        """
        
        self.model.train()
        input_ids_batch = batch["input_ids"].to(device=self.device)
        labels_batch = batch["labels"].to(device = self.device)
        loss, logits = self.model.forward(input_ids_batch, labels_batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def evaluate_step(self, batch) -> float:
        """Evaluation step"""
        self.model.eval()

        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss, logits = self.model(input_ids, labels)
            return loss.item()


    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total steps: {self.total_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Create indices for shuffling
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)

            epoch_loss = 0
            num_batches = len(indices) // self.batch_size

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")

            for batch_idx in progress_bar:
                # Get batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch = {
                    'input_ids': [],
                    'labels': []
                }

                for idx in batch_indices:
                    sample = self.train_dataset[idx]
                    batch['input_ids'].append(sample['input_ids'])
                    batch['labels'].append(sample['labels'])


                batch['input_ids'] = torch.stack(batch['input_ids'])
                batch['labels'] = torch.stack(batch['labels'])

                loss = self.train_step(batch)
                epoch_loss += loss

                current_lr = self.scheduler.get_last_lr()[0]
                self.metrics.update(loss, current_lr)
                self.global_step += 1

                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{self.metrics.get_avg_loss():.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                if self.global_step % self.config.eval_steps == 0:
                    print(f"\nEvaluating model at step {self.global_step}:")
                    print("-" * 50)
                    evaluate_model(self.model, self.tokenizer, ["Once upon a time", "The little girl"])
                    print("-" * 50)
                    checkpoint_path = os.path.join(
                        self.config.output_dir,
                        f"checkpoint-{self.global_step}"
                    )
                    save_model(self.model, self.tokenizer, checkpoint_path)


            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        save_model(self.model, self.tokenizer, self.config.output_dir)
        print("Training completed!")


if __name__ == "__main__":
    raw_dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = GSM8KDataset(
        raw_dataset, tokenizer,
        max_length=training_config.max_position_embeddings
    )
    trainer = Trainer(model, train_dataset, tokenizer, training_config)
    trainer.train()