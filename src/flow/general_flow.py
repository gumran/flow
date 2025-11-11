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
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from flow.transformer import SmallModel, TimeAwareTransformer
from flow.utils import Config

# %%

class Kappa(ABC):
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

    @abstractmethod
    def __call__(self, t) -> torch.Tensor:
        pass

    @abstractmethod
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

class CubicKappa(Kappa):
    """
    Cubic Kappa as in Gat et al.

    parameters:
    - alpha: derivative at t=0.
    - beta: derivative at t=1.

    Default initialisation corresponds to quadratic.
    """
    def __init__(self, config: Config, alpha = 0, beta = 2):
        super().__init__(config)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, t) -> torch.Tensor:
        approx = -2 * t.pow(3) + 3 * t.pow(2) \
            + self.alpha * (t.pow(3) - 2 * t.pow(2) + t) \
            + self.beta * (t.pow(3) - t.pow(2))
        approx = torch.where(approx.abs() < 1e-6, torch.zeros_like(approx) + 1e-6, approx)
        approx = torch.where(approx.abs() > 1 - 1e-6, torch.ones_like(approx), approx)
        return approx

    def derivative(self, t) -> torch.Tensor:
        return -6 * t.pow(2) + 6 * t \
            + self.alpha * (3 * t.pow(2) - 4 * t + torch.ones_like(t)) \
            + self.beta * (3 * t.pow(2) - 2 * t)

class InverseCubicKappa(Kappa):
    """
    Inverse cubic Kappa, where t becomes 1 - t.

    parameters:
    - alpha: negative derivative at t=1.
    - beta: negative derivative at t=0.

    Default initialisation corresponds to quadratic.
    """
    def __init__(self, config: Config, alpha = 0, beta = 2):
        super().__init__(config)
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, t) -> torch.Tensor:
        approx =  2 * t.pow(3) - 3 * t.pow(2) + torch.ones_like(t) \
            + self.alpha * (t.pow(2) - t.pow(3)) \
            - self.beta * (t.pow(3) - 2 * t.pow(2) + t)
        approx = torch.where(approx.abs() < 1e-6, torch.zeros_like(approx) + 1e-6, approx)
        approx = torch.where(approx.abs() > 1 - 1e-6, torch.ones_like(approx), approx)
        return approx

    def derivative(self, t) -> torch.Tensor:
        return 6 * t.pow(2) - 6 * t \
            + self.alpha * (2 * t - 3 * t.pow(2)) \
            - self.beta * (3 * t.pow(2) - 4 * t + torch.ones_like(t))

class ResidualKappa(Kappa):
    """
    Returns the weight of the residual term
    """
    def __init__(self, config, *args):
        super().__init__(config)
        self.kappas = args

    def __call__(self, t) -> torch.Tensor:
        approx = torch.ones_like(t) - sum(kappa(t) for kappa in self.kappas)
        # round to 0 or 1 if close to 0 or 1 to avoid numerical issues
        approx = torch.where(approx.abs() < 1e-6, torch.zeros_like(approx) + 1e-6, approx)
        approx = torch.where(approx.abs() > 1 - 1e-6, torch.ones_like(approx), approx)
        return approx

    def derivative(self, t) -> torch.Tensor:
        return -1 * sum([kappa.derivative(t) for kappa in self.kappas])

def plot_kappa(kappa: Kappa):
    """
    Plot the Kappa function
    """
    t = torch.linspace(0, 1, 100)
    plt.plot(t, kappa(t))
    plt.title(f"Kappa: {kappa.__class__.__name__}")
    plt.xlabel("t")
    plt.ylabel("Kappa(t)")
    plt.grid(True)
    plt.show()

def test_kappa():
    config = Config()
    kappa1 = CubicKappa(config, alpha = 0, beta = 2)
    kappa2 = InverseCubicKappa(config, alpha = 0, beta = 2)
    plot_kappa(kappa1)
    plot_kappa(kappa2)
    plot_kappa(ResidualKappa(config, kappa1, kappa2))

if __name__ == "__main__":
    test_kappa()

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

if __name__ == "__main__":
    test_weight_scheduler()
# %%

"""
Need to implement conditional samplers
"""

class Sampler(ABC):    
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def sample(self, x0, x1) -> torch.Tensor:
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        pass

class InitialSampler(Sampler):

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

class RandomSampler(Sampler):
    def __init__(self, config: Config):
        super().__init__(config)

    def sample(self, x0, x1):
        # x0 : (bs, c) or (...)
        # x1 : (bs, c) or (...)
        return torch.randint(0, self.config.num_tokens, (x0.shape[0], x0.shape[1]), device=x0.device)

# not really sure what other samplers we need

# %%

class CorrectorScheduler(ABC):
    """
    function that returns the weight of the forward velocity
    """
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, t):
        return torch.ones_like(t)

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
    def __init__(self, config: Config, model: nn.Module, samplers: list[Sampler], weight_scheduler: WeightScheduler, corrector_scheduler: CorrectorScheduler = None, temperature: float = 1.0):
        self.config = config
        self.weight_scheduler = weight_scheduler
        self.model = model
        self.samplers = samplers
        self.corrector_scheduler = corrector_scheduler
        self.temperature = temperature
        if self.corrector_scheduler is None:
            self.corrector_scheduler = CorrectorScheduler(config)
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

        path_samples = torch.stack([
            sampler.sample(x0, x1) for sampler in self.samplers
        ], dim=-1) # (bs, c, m)
        weights = self.weight_scheduler(t).unsqueeze(1).expand(-1, self.config.context_len, -1) # (bs, c, m)
        if weights.min() < 0:
            if self.config.debug:
                print(weights.min())
                print("Negative weights found, clipping to 0 and renormalizing")
            # Clip negative values to 0 (due to numerical rounding errors) and renormalize to sum to 1
            weights = weights.clamp(min=0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize to sum to 1
        which_probability_path = Categorical(weights).sample() # (bs, c)
        # some advanced indexing shenanigans to sample each token of xt from
        # path_samples according to the weights distribution
        xt = path_samples[
            torch.arange(0, path_samples.shape[0]).unsqueeze(-1).expand(-1, path_samples.shape[1]), # (bs, c)
            torch.arange(0, path_samples.shape[1]).unsqueeze(0).expand(path_samples.shape[0], -1), # (bs, c)
            which_probability_path # (bs, c)
            ]
        logits = self.model(xt, t) # (bs, sequence_length, output_channels, num_tokens)
        loss = F.cross_entropy(einops.rearrange(logits, "bs c m s -> bs s c m"), path_samples, reduction='mean')
        return loss

    def forward_velocity(self, x, t):
        # x : (bs, c)
        # t : (1,)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # returns (bs, c, s): get the velocity to each token
        logits = self.model(x, t) # (bs, c, m, s)
        # Apply temperature scaling
        logits = logits / self.temperature
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
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # returns (bs, c, s): get the velocity to each token
        logits = self.model(x, t) # (bs, c, m, s)
        # Apply temperature scaling
        logits = logits / self.temperature
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

    def forward_euler_step(self, xt, t, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        # x : (bs, c) or (...)
        # t : (1,)
        # dt : (1,)
        v = self.forward_velocity(xt, t) # (bs, c, s)
        diag = F.one_hot(xt, self.config.num_tokens) # (bs, c, s)
        probs = (diag + dt * v).clip(min=0.0) # (bs, c, s)
        xtdt = Categorical(probs).sample() # (bs, c)
        return xtdt

    def backward_euler_step(self, xt, t, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        v = self.backward_velocity(xt, t) # (bs, c, s)
        diag = F.one_hot(xt, self.config.num_tokens) # (bs, c, s)
        probs = (diag - dt * v).clip(min=0.0) # (bs, c, s) - negative due to backward velocity definition
        xtdt = Categorical(probs).sample() # (bs, c)
        return xtdt

    def corrector_sampling_euler_step(self, xt, t, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        v = self.corrector_sampling_velocity(xt, t) # (bs, c, s)
        diag = F.one_hot(xt, self.config.num_tokens) # (bs, c, s)
        probs = (diag + dt * v).clip(min=0.0) # (bs, c, s)
        xtdt = Categorical(probs).sample() # (bs, c)
        return xtdt

    def corrector_iteration_euler_step(self, xt, t, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        v = self.corrector_iteration_velocity(xt, t) # (bs, c, s)
        diag = F.one_hot(xt, self.config.num_tokens) # (bs, c, s)
        probs = (diag + dt * v).clip(min=0.0) # (bs, c, s)
        xtdt = Categorical(probs).sample() # (bs, c)
        return xtdt
    
    def forward_sample(self, x0, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        t = torch.tensor(0., device=x0.device)
        x = x0.clone()
        while t < 1:
            x = self.forward_euler_step(x, t, dt)
            t += dt
        return x

    def corrector_sample(self, x0, dt = None):
        if dt is None:
            dt = getattr(self.config, "dt", 1e-3)
        t = torch.tensor(1e-6, device=x0.device) # start away from 0 so backward velocity is defined
        x = x0.clone()
        while t < 1:
            x = self.corrector_sampling_euler_step(x, t, dt)
            t += dt
        return x

    def set_corrector_scheduler(self, corrector_scheduler: CorrectorScheduler):
        self.corrector_scheduler = corrector_scheduler
    
class UsualFlow(GeneralFlow):
    """
    Usual masked flow
    """
    def __init__(self, config: Config, model: nn.Module, temperature: float = 1.0):
        kappa1 = LinearKappa(config)
        assert config.add_residual, "Usual flow requires residual kappa"
        weight_scheduler = WeightScheduler(config, kappa1) # 2 kappas
        samplers = [
            DataSampler(config),
            InitialSampler(config)
        ]
        super().__init__(config, model, samplers, weight_scheduler, temperature=temperature)

class QuadraticRandomFlow(GeneralFlow):
    """
    Quadratic random flow
    """
    def __init__(self, config: Config, model: nn.Module, temperature: float = 1.0):
        kappa1 = CubicKappa(config, alpha = 0, beta = 2)
        kappa2 = InverseCubicKappa(config, alpha = 0, beta = 2)
        assert config.add_residual, "Quadratic random flow requires residual kappa"
        weight_scheduler = WeightScheduler(config, kappa1, kappa2) # 3 kappas
        samplers = [
            DataSampler(config),
            InitialSampler(config),
            RandomSampler(config)
        ]
        super().__init__(config, model, samplers, weight_scheduler, temperature=temperature)

# %%

def basic_uniform_noise_sampler(bs, device='cpu'):
    return torch.randint(0, 3, (bs, 10), device=device)

def basic_mask_sampler(bs, device='cpu', mask_token = 3):
    return torch.full((bs, 10), mask_token, dtype=torch.long, device=device)

def basic_data_sampler(bs, device='cpu'):
    # increasing tensors with 0, 1 and 2
    pos1 = torch.randint(0, 11, (bs,), device = device) # (bs,)
    onehot1 = F.one_hot(pos1, num_classes = 11) # (bs, 11)
    cumsum1 = onehot1.cumsum(dim=-1) # (bs, 11)
    r1 = cumsum1[:, :-1] # (bs, 10)

    pos2 = torch.randint(0, 11, (bs,), device = device) # (bs,)
    onehot2 = F.one_hot(pos2, num_classes = 11) # (bs, 11)
    cumsum2 = onehot2.cumsum(dim=-1) # (bs, 11)
    r2 = cumsum2[:, :-1] # (bs, 10)

    return r1 + r2

# %%

if __name__ == "__main__":
    print(basic_uniform_noise_sampler(20))
    print('-'*100)
    print(basic_mask_sampler(20))
    print('-'*100)
    print(basic_data_sampler(20))

# %%

def test_general_flow():
    config = Config(
        num_tokens = 3,
        context_len = 10,
        add_residual=True,
        output_channels = 2
        )
    kappa1 = LinearKappa(config)
    weight_scheduler = WeightScheduler(config, kappa1) # 2 kappas
    samplers = [
        DataSampler(config),
        InitialSampler(config)
    ]
    model = TimeAwareTransformer(config)
    gf = GeneralFlow(
        config=config,
        model=model,
        samplers=samplers,
        weight_scheduler=weight_scheduler
    )
    x = torch.randint(0, 3, (5, 10))
    t = torch.tensor(0.3)
    fv = gf.forward_velocity(x, t)
    print(fv.shape)
    print(fv.sum(dim = -1).abs().max()) # should be small
    bv = gf.backward_velocity(x, t)
    print(bv.shape)
    print(bv.sum(dim = -1).abs().max()) # should be small
    cv = gf.corrector_sampling_velocity(x, t)
    print(cv.shape)
    print(cv.sum(dim = -1).abs().max()) # should be small
    print(cv)


# %%

if __name__ == "__main__":
    test_general_flow()

# %%

def small_uniform_train():
    config = Config(
        num_tokens = 3,
        context_len = 10,
        add_residual=True,
        output_channels = 2
        )
    kappa1 = LinearKappa(config)
    weight_scheduler = WeightScheduler(config, kappa1) # 2 kappas
    samplers = [
        DataSampler(config),
        InitialSampler(config)
    ]
    model = TimeAwareTransformer(config)
    gf = GeneralFlow(
        config=config,
        model=model,
        samplers=samplers,
        weight_scheduler=weight_scheduler
    )
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0)
    bs = 32
    num_epochs = 5000
    device = config.device
    wrapper = tqdm(range(num_epochs))
    losses = []
    last_100 = []
    for epoch in wrapper:
        x0 = basic_uniform_noise_sampler(bs, device)
        x1 = basic_data_sampler(bs, device)
        t = torch.rand((bs,), device=device)
        loss = gf.get_train_loss(x0, x1, t)
        loss.backward()
        opt.step()
        wrapper.set_postfix_str(f"loss: {loss}")
        # print(loss)
        losses.append(loss.detach().cpu())
        last_100.append(loss)
        if (epoch + 1) % 100 == 0:
            print(f"mean of 100 losses: {sum(last_100) / 100}")
            last_100.clear()

    # plt.plot(losses)
    x0 = basic_uniform_noise_sampler(bs, device)
    print(gf.forward_sample(x0))

# %%
if __name__ == "__main__":
    small_uniform_train()
# %%

def small_mask_train():
    config = Config(
        num_tokens = 4,
        context_len = 10,
        add_residual=True,
        output_channels = 2
        )
    kappa1 = LinearKappa(config)
    weight_scheduler = WeightScheduler(config, kappa1) # 2 kappas
    samplers = [
        DataSampler(config),
        InitialSampler(config)
    ]
    model = TimeAwareTransformer(config)
    gf = GeneralFlow(
        config=config,
        model=model,
        samplers=samplers,
        weight_scheduler=weight_scheduler
    )
    bs = 32
    num_epochs = 10000
    device = config.device
    wrapper = tqdm(range(num_epochs))
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    losses = []
    last_100 = []
    for epoch in wrapper:
        x0 = basic_mask_sampler(bs, device)
        x1 = basic_data_sampler(bs, device)
        t = torch.rand((bs,), device=device)
        loss = gf.get_train_loss(x0, x1, t)
        loss.backward()
        opt.step()
        scheduler.step()
        wrapper.set_postfix_str(f"loss: {loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
        # print(loss)
        losses.append(loss.detach().cpu())
        last_100.append(loss)
        if (epoch + 1) % 100 == 0:
            print(f"mean of 100 losses: {sum(last_100) / 100:.4f}")
            last_100.clear()

    plt.plot(losses, label='loss')
    plt.show()
    x0 = basic_mask_sampler(bs, device)
    print(gf.forward_sample(x0))

    return model, gf

# %%

def small_quadratic_random_train():
    config = Config(
        num_tokens = 4,
        context_len = 10,
        add_residual=True,
        output_channels = 3,
        )
    model = TimeAwareTransformer(config)
    print(config.device)
    model.to(config.device)
    gf = QuadraticRandomFlow(config=config, model=model)
    bs = 32
    num_epochs = 10000
    device = config.device
    wrapper = tqdm(range(num_epochs))
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    losses = []
    last_100 = []
    for epoch in wrapper:
        x0 = basic_mask_sampler(bs, device)
        x1 = basic_data_sampler(bs, device)
        t = torch.rand((bs,), device=device)
        loss = gf.get_train_loss(x0, x1, t)
        loss.backward()
        opt.step()
        scheduler.step()
        wrapper.set_postfix_str(f"loss: {loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
        # print(loss)
        losses.append(loss.detach().cpu())
        last_100.append(loss)
        if (epoch + 1) % 100 == 0:
            print(f"mean of 100 losses: {sum(last_100) / 100:.4f}")
            last_100.clear()

    plt.plot(losses, label='loss')
    plt.show()
    x0 = basic_mask_sampler(bs, device)
    print(gf.forward_sample(x0))

    return model, gf

# %%
if __name__ == "__main__":
    model, gf = small_mask_train()
    # %%

    x = basic_mask_sampler(32, 'cpu')
    for _ in range(10000):
        x = gf.corrector_iteration_euler_step(x, torch.tensor(0.5), 1e-4)

    print(x)

    # %%

if __name__ == "__main__":
    x = basic_mask_sampler(32, 'cpu')
    x = gf.corrector_sample(x, dt = 1e-4)

    print(x)

# %%
if __name__ == "__main__":
    model, gf = small_quadratic_random_train()

    x = basic_mask_sampler(32, 'cuda')
    for _ in range(10000):
        x = gf.corrector_iteration_euler_step(x, torch.tensor(0.5, device=x.device), 1e-4)

    print(x)

# %%
