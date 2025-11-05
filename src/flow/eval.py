"""
Eval of discrete FM models.
"""
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

from flow.campbell_flow import MaskedFMModel
from flow.transformer import Config, IgnorantTransformer

# %%

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

small_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=128,
    mlp_dim=256,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=16,
    context_len=384,
    num_layers=8,
    timestep_scale=1000.0,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

large_config = Config(
    num_tokens=len(tokenizer),
    embed_dim=512,
    mlp_dim=2048,
    frequency_embedding_dim=128,
    num_heads=8,
    head_dim=64,
    context_len=384,
    num_layers=16,
    timestep_scale=1000.0,
    debug=True,
    add_residual=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)
config = large_config

# %%

def gpt2_model(device):
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    return model

def sentence_bert_model(device):
    model = SentenceTransformer(
        'Qwen/Qwen3-Embedding-0.6B', 
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"}).to(device)
    return model

# %%

class PerplexityEvaluator:
    def __init__(self, model = None, tokenizer = None, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model if model is not None else gpt2_model(device)
        self.tokenizer = tokenizer if tokenizer is not None else GPT2Tokenizer.from_pretrained('gpt2')
        # Set pad token to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def calculate_perplexity(self, texts: list[str], batch_size: int = 16, max_length: int = 1024):
        if len(texts) == 0:
            return float("nan")

        self.model.eval()
        total_nll = 0.0
        total_eff_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            # Build labels with pad positions ignored
            labels = enc.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # ignored by loss

            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    labels=labels,
                )
                # HF computes mean over *valid* labels (labels != -100) after shifting
                batch_loss = outputs.loss.item()

            # Count effective tokens EXACTLY like the model’s loss:
            # the loss uses labels[..., 1:], so count valid labels after shift
            eff_tokens = (labels[:, 1:] != -100).sum().item()

            total_nll += batch_loss * eff_tokens
            total_eff_tokens += eff_tokens

        if total_eff_tokens == 0:
            return float("nan")

        avg_nll = total_nll / total_eff_tokens
        ppl = math.exp(avg_nll)
        return ppl

# %%

def test_perplexity_evaluator():

    mask_token_id = tokenizer.mask_token_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/scratch/inath/checkpoints/tinystories_campbell_flow_full_final_model.pt"
    model = IgnorantTransformer(config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print("Model loaded")
    fm = MaskedFMModel(config, model, mask_token_id)
    print("FM model loaded")
    evaluator = PerplexityEvaluator(device=device)

    example_texts = [
        "The curious cat chased the butterfly around the garden.",
        "Children laughed as they built a tall snowman in the park.",
        "A gentle breeze rustled the leaves on the old maple tree.",
        "Sarah painted a colorful picture of her favorite horse.",
        "The teacher smiled and handed out gold stars to the class.",
        "Beneath the waves, the fish darted between the coral reefs.",
        "Lucas practiced the piano every evening before dinner.",
        "Thunder rumbled as dark clouds gathered over the hills.",
        "Grandpa told stories while everyone roasted marshmallows.",
        "The baker woke up early to prepare fresh loaves of bread."
    ]

    evaluator.calculate_perplexity(example_texts)
    print("Perplexity of example texts: ", evaluator.calculate_perplexity(example_texts))

    with torch.inference_mode():
        example_texts = fm.sample(32, dt = 1e-2)
    generated_texts = tokenizer.batch_decode(example_texts, skip_special_tokens=True)
    evaluator.calculate_perplexity(generated_texts)
    print("Perplexity of generated texts: ", evaluator.calculate_perplexity(generated_texts))


# %%

if __name__ == "__main__":
    test_perplexity_evaluator()

# %%

class SentenceBERTEvaluator:
    def __init__(self, model = None, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model if model is not None else sentence_bert_model(device)
        self.device = device
        self.cached_embeddings = None

    def cache_embeddings(self, texts: list[str]):
        self.cached_embeddings = self.model.encode(texts, show_progress_bar=True) # (cached_size, emb_dim)

    def calculate_max_correlation_batch(self, new_sentences, model, top_k=1):
        """
        Compute cosine similarity between new_sentences and cached reference embeddings.
        Return top_k matches and distances.
        """
        with torch.no_grad():
            new_emb = model.encode(new_sentences, convert_to_tensor=True, device=self.device) # (bs, emb_dim)
            correlation = self.model.similarity(new_emb, self.cached_embeddings) # (bs, cached_size)
        closest_idx = correlation.argmax(dim=1) # (bs,)
        return correlation.max(dim=1).values, closest_idx # (bs,), (bs,)

    def calculate_distance_batch(self, new_sentences, model, top_k=1):
        """
        Compute distance between new_sentences and cached reference embeddings.
        Return top_k matches and distances.
        """
        with torch.no_grad():
            new_emb = model.encode(new_sentences, convert_to_tensor=True, device=self.device)
            correlation = self.model.similarity(new_emb, self.cached_embeddings) # (bs, cached_size)
        closest_idx = correlation.argmax(dim=1) # (bs,)
        return 1 - correlation.max(dim=1).values, closest_idx # (bs,), (bs,)

# %%
