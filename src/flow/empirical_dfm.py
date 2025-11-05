"""
Implement empirical DFM flows for testing and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

