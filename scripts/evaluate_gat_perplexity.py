"""
Script to evaluate the perplexity of pre-trained GAT flow matching models
(UsualFlow or QuadraticRandomFlow)
"""

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from collections import defaultdict
import pickle
from typing import Union, Dict, Any, Type

from flow.transformer import TimeAwareTransformer, IgnorantTransformer
from flow.utils import Config
from flow.general_flow import UsualFlow, QuadraticRandomFlow, PolynomialCorrectorScheduler
from flow.eval import PerplexityEvaluator

# %%
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def evaluate_entropy(text_samples, tokenizer):
    """
    text_samples: List[str], iterables of raw text
    tokenizer: Huggingface tokenizer or similar
    Returns: average entropy per token position, computed over tokenized samples
    """
    # Tokenize samples with tokenizer
    encodings = tokenizer(
        text_samples,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = encodings["input_ids"]  # shape: (num_samples, seq_len)
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    # input_ids is (num_samples, seq_len)
    num_samples, seq_len = input_ids.shape
    entropies = []

    for pos in range(seq_len):
        tokens_at_pos = input_ids[:, pos]  # tensor of shape (num_samples,)
        # Count occurrences for each token id
        token_ids, counts = torch.unique(tokens_at_pos, return_counts=True)
        probs = counts.float() / counts.sum()
        # Compute entropy at this position (add epsilon for numerical stability)
        entropy_pos = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        entropies.append(entropy_pos)

    avg_entropy = float(torch.tensor(entropies).mean())
    return avg_entropy


def sample_from_gat_flow(
    flow_model: Union[UsualFlow, QuadraticRandomFlow],
    batch_size: int,
    mask_token_id: int,
    dt: float = 0.01,
    use_corrector: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Generate samples from a GAT flow matching model.
    
    Args:
        flow_model: UsualFlow or QuadraticRandomFlow instance (temperature is set at initialization)
        batch_size: Number of samples to generate
        mask_token_id: Token ID to use for masked initialization
        dt: Time step size for Euler integration
        use_corrector: If True, use corrector_sample instead of forward_sample
        **kwargs: Additional arguments (currently unused, for compatibility)
    
    Returns:
        Generated token sequences of shape (batch_size, context_len)
    """
    config = flow_model.config
    device = config.device
    
    # Create initial x0 - start from masked tokens
    x0 = torch.full((batch_size, config.context_len), mask_token_id, dtype=torch.long, device=device)
    
    if use_corrector:
        samples = flow_model.corrector_sample(x0, dt=dt)
    else:
        samples = flow_model.forward_sample(x0, dt=dt)
    
    return samples


def evaluate_gat_perplexity(
    model: nn.Module,
    config: Config,
    flow_class: Type[Union[UsualFlow, QuadraticRandomFlow]],
    args_dict: Dict[str, Dict[str, Any]],
    mask_token_id: int,
    num_samples: int = 500,
    batch_size: int = 10,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate perplexity and entropy for a GAT flow matching model with different parameter configurations.
    
    Args:
        model: The transformer model (TimeAwareTransformer or IgnorantTransformer)
        config: Config object
        flow_class: UsualFlow or QuadraticRandomFlow class
        args_dict: Dictionary mapping configuration names to parameter dicts
                  Each parameter dict can contain: dt, temperature, use_corrector
        mask_token_id: Token ID to use for masked initialization
        num_samples: Total number of samples to generate per configuration
        batch_size: Batch size for generation
        output_path: Optional path to save results as pickle file
    
    Returns:
        Dictionary containing time_taken, strings, perplexities, and entropies
    """
    perplexity_evaluator = PerplexityEvaluator(device=config.device)
    
    time_taken = defaultdict(float)
    strings = defaultdict(list)
    perplexities = defaultdict(float)
    entropies = defaultdict(float)
    
    def evaluate_samples(name: str, args: Dict[str, Any]):
        """Generate samples and evaluate for a given configuration."""
        start_time = time.time()
        
        # Extract parameters
        dt = args.get("dt", 0.01)
        temperature = args.get("temperature", 1.0)
        use_corrector = args.get("use_corrector", False)
        
        # Create flow model with specified temperature
        flow_model = flow_class(config, model, temperature=temperature)
        
        # Set corrector scheduler if using corrector
        if use_corrector:
            corrector_alpha = args.get("corrector_alpha", 10)
            corrector_a = args.get("corrector_a", 0.25)
            corrector_b = args.get("corrector_b", 0.25)
            corrector_scheduler = PolynomialCorrectorScheduler(
                config, 
                alpha=corrector_alpha, 
                a=corrector_a, 
                b=corrector_b
            )
            flow_model.set_corrector_scheduler(corrector_scheduler)
        
        with torch.inference_mode():
            while len(strings[name]) < num_samples:
                # Generate samples
                generated_sentences = sample_from_gat_flow(
                    flow_model,
                    batch_size=batch_size,
                    mask_token_id=mask_token_id,
                    dt=dt,
                    use_corrector=use_corrector,
                    **{k: v for k, v in args.items() if k not in ["dt", "temperature", "use_corrector", "corrector_alpha", "corrector_a", "corrector_b"]}
                )
                generated_texts = tokenizer.batch_decode(generated_sentences, skip_special_tokens=True)
                strings[name].extend(generated_texts)
        
        end_time = time.time()
        time_taken[name] = end_time - start_time
        perplexities[name] = perplexity_evaluator.calculate_perplexity(strings[name])
        entropies[name] = evaluate_entropy(strings[name], tokenizer)
        print(f"{name}: perplexity={perplexities[name]:.2f}, entropy={entropies[name]:.2f}")
    
    # Evaluate for each configuration
    for name, args in tqdm(args_dict.items(), desc="Evaluating configurations"):
        evaluate_samples(name, args)
    
    results = {
        "time_taken": dict(time_taken),
        "strings": dict(strings),
        "perplexities": dict(perplexities),
        "entropies": dict(entropies)
    }
    
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_path}")
    
    return results


# %%
# Example usage and configuration

def create_default_args_dict() -> Dict[str, Dict[str, Any]]:
    """
    Create a default dictionary of argument configurations to evaluate.
    Sweeps over different PolynomialCorrectorScheduler configurations and temperatures.
    """
    args_dict = {}
    
    # Fixed dt = 0.01
    dt = 0.01
    
    # Temperature sweep values
    temperatures = [0.7, 0.8, 0.9, 1.0]
    
    # Baseline: No corrector, sweep over temperatures
    for temp in temperatures:
        args_dict[f"no_corrector_temp_{temp}"] = {
            "dt": dt,
            "temperature": temp,
            "use_corrector": False,
        }
    
    # Sweep over different corrector scheduler configurations
    # PolynomialCorrectorScheduler parameters: alpha, a, b
    # Default: alpha=10, a=0.25, b=0.25
    # Only 3 corrector configurations per temperature
    
    for temp in temperatures:
        # 1. Default corrector configuration
        args_dict[f"corrector_default_temp_{temp}"] = {
            "dt": dt,
            "temperature": temp,
            "use_corrector": True,
            "corrector_alpha": 10,
            "corrector_a": 0.25,
            "corrector_b": 0.25,
        }
        
        # 2. Lower alpha (weaker corrector)
        args_dict[f"corrector_alpha_1_temp_{temp}"] = {
            "dt": dt,
            "temperature": temp,
            "use_corrector": True,
            "corrector_alpha": 1,
            "corrector_a": 0.25,
            "corrector_b": 0.25,
        }
        
        # 3. Higher a parameter (more time dependence at start)
        args_dict[f"corrector_a_0.5_temp_{temp}"] = {
            "dt": dt,
            "temperature": temp,
            "use_corrector": True,
            "corrector_alpha": 10,
            "corrector_a": 0.5,
            "corrector_b": 0.25,
        }
    
    return args_dict


# %%
if __name__ == "__main__":
    # Configuration
    config = Config(
        num_tokens=len(tokenizer),
        embed_dim=512,
        mlp_dim=2048,
        frequency_embedding_dim=128,
        num_heads=8,
        head_dim=64,
        context_len=384,
        num_layers=16,
        output_channels=2,
        timestep_scale=1000.0,
        debug=True,
        add_residual=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Load model and create flow
    # Example: Load a checkpoint and create UsualFlow or QuadraticRandomFlow
    model_path = "/scratch/inath/checkpoints/tinystories_general_flow_full_final_model.pt"
    model = TimeAwareTransformer(config)
    
    # Load checkpoint and handle "transformer." prefix if present
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict):
        # Check if keys have "transformer." prefix
        if any(k.startswith("transformer.") for k in checkpoint.keys()):
            # Strip "transformer." prefix from all keys
            checkpoint = {k.replace("transformer.", "", 1): v for k, v in checkpoint.items()}
        # Load with strict=False to allow missing keys (e.g., timestep_embedder might not be in checkpoint)
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(config.device)
    model.eval()
    # 
    # Choose flow type
    flow_class = UsualFlow  # or QuadraticRandomFlow
    mask_token_id = tokenizer.mask_token_id
    
    # For demonstration, create a default args_dict
    args_dict = create_default_args_dict()
    
    # Uncomment to run evaluation:
    results = evaluate_gat_perplexity(
        model=model,
        config=config,
        flow_class=flow_class,
        args_dict=args_dict,
        mask_token_id=mask_token_id,
        num_samples=500,
        batch_size=10,
        output_path="/scratch/inath/pickles/tinystories_general_flow_perplexity_results.pkl"
    )
