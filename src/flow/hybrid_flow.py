"""
Hybrid flow matching implementations that combine empirical and trained models.

This module provides samplers that use empirical DFM for early stages of generation
and transition to trained models for the final stages.
"""

import torch
import torch.nn as nn

from flow.utils import Config


class HybridMaskedSampler(nn.Module):
    """
    A hybrid sampler that uses empirical DFM until time tau, then switches to a trained model.
    
    This allows leveraging the empirical distribution early in the flow and transitioning
    to learned model predictions for the final stages of generation.
    """
    def __init__(self, config: Config, empirical_dfm, trained_model):
        """
        Args:
            config: Config object
            empirical_dfm: EmpiricalDFM instance (must have initial_type="mask")
            trained_model: MaskedFMModel instance
        """
        super().__init__()
        self.config = config
        self.empirical_dfm = empirical_dfm
        self.trained_model = trained_model
        
        # Ensure empirical DFM is using masked initial type
        if empirical_dfm.initial_type != "mask":
            raise ValueError(f"empirical_dfm must have initial_type='mask', got '{empirical_dfm.initial_type}'")
    
    def sample(self, bs, tau, eta=None, dt=None, temperature=None, top_k=None):
        """
        Sample using hybrid approach: empirical DFM from t=0 to t=tau, then trained model from t=tau to t=1.
        
        Args:
            bs: Batch size (int)
            tau: Time at which to switch from empirical to trained model (0 < tau < 1)
            eta: Stochasticity parameter (optional, uses config default if None)
            dt: Time step size (optional, uses config default if None)
            temperature: Sampling temperature for trained model (optional, uses config default if None)
            top_k: Top-k sampling for trained model (optional, uses config default if None)
        
        Returns:
            Final samples at t=1, shape (bs, context_len)
        """
        if tau <= 0 or tau >= 1:
            raise ValueError(f"tau must be between 0 and 1, got {tau}")
        
        # Sample using empirical DFM from t=0 to t=tau
        x_tau = self.empirical_dfm.sample(bs, eta=eta, dt=dt, tau=tau)
        
        # Continue sampling using trained model from t=tau to t=1
        x_final = self.trained_model.sample(x_tau, eta=eta, dt=dt, temperature=temperature, top_k=top_k, t=tau)
        
        return x_final

