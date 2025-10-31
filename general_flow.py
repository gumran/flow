"""
Implementations of general flows like Gat et al.
"""

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from dataclasses import dataclass
import einops

from transformer import SmallModel

# %%

@dataclass
class Config:
    num_tokens: int = 20                        # number of tokens in the vocabulary
    embed_dim: int = 16                         # dimension of the embedding
    mlp_dim: int = 32                           # dimension of the MLP
    frequency_embedding_dim: int = 32           # dimension of the frequency embedding
    num_heads: int = 4                          # number of heads in the attention
    head_dim: int = 4                           # dimension of each head
    context_len: int = 1024                     # maximum length of the context
    num_layers: int = 6                         # number of layers in the transformer
    timestep_scale: float = 1000.0              # how much to scale the timestep embedding
    debug: bool = False                         # whether to print debug information
    device: str = "cuda" \
        if torch.cuda.is_available() \
        else "cpu"                              # device to use
    num_input_tokens: int = None                # number of input tokens, can be different from num_tokens (useful for masking, etc.)
    add_residual: bool = False                  # whether to add a residual term to the weight scheduler

    def __post_init__(self):
        if self.num_input_tokens is None:
            self.num_input_tokens = self.num_tokens

# %%

class Kappa():
    """
    Returns the weight of a conditional probability

    Example:
    >>> kappa = Kappa(config)
    >>> kappa(t) # returns the weight of the conditional probability
    >>> kappa.derivative(t) # returns the derivative of the weight of the conditional probability

    Methods:
    - __call__(t): returns the weight of the conditional probability
    - derivative(t): returns the derivative of the weight of the conditional probability
    """
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, t) -> torch.Tensor:
        pass

    def derivative(self, t) -> torch.Tensor:
        pass

class LinearKappa(Kappa):
    """
    Simple linear Kappa
    """
    def __init__(self, config: Config):
        super().__init__(config)

    def __call__(self, t) -> torch.Tensor:
        return t

    def derivative(self, t) -> torch.Tensor:
        return torch.ones_like(t)

class ResidualKappa(Kappa):
    """
    Returns the weight of the residual term
    """
    def __init__(self, config, *args):
        super().__init__(config)
        self.kappas = args

    def __call__(self, t) -> torch.Tensor:
        return torch.ones_like(t) - sum(kappa(t) for kappa in self.kappas)

    def derivative(self, t) -> torch.Tensor:
        return -1 * sum(kappa.derivative(t) for kappa in self.kappas)

class WeightScheduler():
    """
    List of Kappa schedulers.
    
    Methods:
    - __call__(t): returns the weights of the Kappa schedulers
    - derivative(t): returns the derivatives of the Kappa schedulers
    - b(t): returns the minimum ratio of the weights to the derivatives
    - a(t): returns the difference between the derivatives and the weights multiplied by the minimum ratio

    Example:
    >>> weight_scheduler = WeightScheduler(config, *kappas)
    >>> weight_scheduler(t) # returns the weights of the Kappa schedulers
    >>> weight_scheduler.derivative(t) # returns the derivatives of the Kappa schedulers
    >>> weight_scheduler.b(t) # returns the minimum ratio
    >>> weight_scheduler.a(t) # returns the weights for each conditional posterior
    """
    def __init__(self, config: Config, *args):
        self.config = config
        self.kappas = args # list of m or m-1 Kappa schedulers
        self.add_residual = getattr(config, "add_residual", False)
        if self.add_residual:
            self.residual_kappa = ResidualKappa(config, *self.kappas)
            self.kappas = [*self.kappas, self.residual_kappa]
        self.m = len(self.kappas)

    def __call__(self, t):
        return torch.stack([kappa(t) for kappa in self.kappas], dim=-1) # (bs, m) or (m,)

    def derivative(self, t):    
        return torch.stack([
            kappa.derivative(t) for kappa in self.kappas
            ], dim=-1) # (bs, m) or (m,)

    def b(self, t):
        probs = self(t)
        derivatives = self.derivative(t)
        ratios = derivatives / probs
        return torch.min(ratios, dim=-1).values # (bs,) or ()

    def a(self, t):
        b = self.b(t)
        return self.derivative(t) - self(t) * b.unsqueeze(-1) # (bs, m) or (m,)

    # for backwards velocity calculation
    def b_backward(self, t):
        probs = self(t)
        derivatives = self.derivative(t)
        ratios = derivatives / probs
        return torch.max(ratios, dim=-1).values # (bs,) or ()

    def a_backward(self, t):
        b = self.b_backward(t)
        return self.derivative(t) - self(t) * b.unsqueeze(-1) # (bs, m) or (m,)

# %%

def test_weight_scheduler():
    config = Config(add_residual=True)
    kappas = [LinearKappa(config)]
    weight_scheduler = WeightScheduler(config, *kappas)
    print(weight_scheduler(torch.tensor(0.5)))
    print(weight_scheduler.derivative(torch.tensor(0.5)))
    print(weight_scheduler.b(torch.tensor(0.5)))
    print(weight_scheduler.a(torch.tensor(0.5)))
    for time in torch.linspace(0, 1, 10):
        print("-" * 100)
        print(f"Time: {time}")
        print(weight_scheduler(time))
        print(f"Derivative: {weight_scheduler.derivative(time)}")
        print(f"B: {weight_scheduler.b(time)}")
        print(f"A: {weight_scheduler.a(time)}")

    tensor = torch.rand(10, 3)
    print(weight_scheduler.b(tensor))
    print(weight_scheduler.a(tensor))
    print(weight_scheduler.derivative(tensor))
    print(weight_scheduler(tensor))

# %%

test_weight_scheduler()
# %%

"""
Need to implement conditional samplers
"""

class Sampler():    
    def __init__(self, config: Config):
        self.config = config

    def sample(self, x0, x1):
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        pass

class NoiseSampler(Sampler):

    def __init__(self, config: Config):
        super().__init__(config)

    def sample(self, x0, x1):
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        return x0

class DataSampler(Sampler):
    def __init__(self, config: Config):
        super().__init__(config)

    def sample(self, x0, x1):
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        return x1

# not really sure what other samplers we need

# %%

class CorrectorScheduler():
    """
    function that returnt the weight of the forward velocity
    """
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, t):
        return self.ones_like(t)

class PolynomialCorrectorScheduler(CorrectorScheduler):
    """
    Polynomial corrector scheduler
    """
    def __init__(self, config: Config, alpha = 10, a = 0.25, b = 0.25):
        super().__init__(config)
        self.alpha = alpha
        self.a = a
        self.b = b

    def __call__(self, t):
        return 1 + self.alpha * (t.pow(self.a)) * ((1 - t).pow(self.b))


# %%

class GeneralFlow():
    """
    General flow model
    """
    def __init__(self, config: Config, model: nn.Module, samplers: list[Sampler], weight_scheduler: WeightScheduler, corrector_scheduler: CorrectorScheduler = CorrectorScheduler()):
        self.config = config
        self.weight_scheduler = weight_scheduler
        self.model = model
        self.samplers = samplers
        self.corrector_scheduler = corrector_scheduler
        assert self.config.output_channels == self.weight_scheduler.m, f"output_channels must match the number of kappas"
        assert len(self.samplers) == self.weight_scheduler.m, f"number of samplers must match the number of kappas"

    def get_train_loss(self, x0, x1, t):
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        # t : (bs,) or (1,) or (...)

        # make t have correct shape (bs,)
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x0.shape[0])
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(x0.shape[0])
            else:
                assert t.shape[0] == x0.shape[0], f"t.shape: {t.shape}, x0.shape: {x0.shape}"
        else:
            raise ValueError(f"Invalid t shape: {t.shape}")

        t = t.to(x0.device)

        samples = torch.stack([
            sampler.sample(x0, x1) for sampler in self.samplers
        ], dim=-1) # (bs, c, m)
        weights = self.weight_scheduler(t) # (bs, m)
        xt = torch.einsum("bm,bcm->bc", weights, samples) # (bs, c)
        logits = self.model(xt, t) # (bs, sequence_length, output_channels, num_tokens)
        loss = F.cross_entropy(einops.rearrange(logits, "bs c m s -> bs s c m"), samples, reduction='mean')
        return loss

    def forward_velocity(self, x, t):
        # x : (bs, c)
        # t : (1,)
        # returns (bs, c, s): get the velocity to each token
        logits = self.model(x, t) # (bs, c, m, s)
        probs = F.softmax(logits, dim=-1) # (bs, c, m, s)
        a = self.weight_scheduler.a(t) # (bs, m)
        velocity = torch.einsum("bm,bcms->bcs", a, probs) # (bs, c, s)
        # handle diagonal terms
        b = self.weight_scheduler.b(t) # (bs,)
        # for each token in x, add b to that velocity
        diagonal_term = F.one_hot(x, num_classes=self.config.num_tokens).float() * b.unsqueeze(-1).unsqueeze(-1) # (bs, c, s)
        velocity += diagonal_term
        return velocity

    def backward_velocity(self, x, t):
        # x : (bs, c)
        # t : (1,)
        # returns (bs, c, s): get the velocity to each token
        logits = self.model(x, t) # (bs, c, m, s)
        probs = F.softmax(logits, dim=-1) # (bs, c, m, s)
        a = self.weight_scheduler.a_backward(t) # (bs, m)
        velocity = torch.einsum("bm,bcms->bcs", a, probs) # (bs, c, s)
        # handle diagonal terms
        b = self.weight_scheduler.b_backward(t) # (bs,)
        # for each token in x, add b to that velocity
        diagonal_term = F.one_hot(x, num_classes=self.config.num_tokens).float() * b.unsqueeze(-1).unsqueeze(-1) # (bs, c, s)
        velocity += diagonal_term
        return velocity

    def corrector_sampling_velocity(self, x, t):
        alpha_t = self.corrector_scheduler(t)
        beta_t = alpha_t - 1
        return alpha_t * self.forward_velocity(x, t) - beta_t * self.backward_velocity(x, t)

    def corrector_iteration_velocity(self, x, t):
        alpha_t = self.corrector_scheduler(t)
        return alpha_t * self.forward_velocity(x, t) - alpha_t * self.backward_velocity(x, t)

    def forward_euler_step(self, x, t, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        # x : (bs, c) or (...)
        # t : (1,)
        # dt : (1,)

        logits = self.model(x, t) # (bs, c, m, s)
