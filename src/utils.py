import os, random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_frame(frame_type="Haar"):
    if frame_type == "Haar":
        return [lambda x: np.cos(x / 2), lambda x: np.sin(x / 2)]
    elif frame_type == "Linear":
        return [lambda x: np.cos(x / 2) ** 2,
                lambda x: np.sin(x) / np.sqrt(2),
                lambda x: np.sin(x / 2) ** 2]
    raise ValueError(f"Unsupported frame type: {frame_type}")

def cheb_approx(func, n=2):
    quad_points = 500
    c = np.zeros(n, dtype=np.float32)
    a = np.pi / 2
    for k in range(1, n + 1):
        x = np.linspace(0, np.pi, quad_points)
        integrand = np.cos((k - 1) * x) * func(a * (np.cos(x) + 1))
        c[k - 1] = 2 / np.pi * np.trapz(integrand, x)
    return c

def sinusoidal_posemb(x, dim, theta=10000):
    half = dim // 2
    freq = (np.log(theta) / max(half - 1, 1)) if half > 1 else 0.0
    freq = np.exp(np.arange(half) * -freq)
    outer = np.outer(x, freq)
    return np.concatenate([np.sin(outer), np.cos(outer)], axis=-1).astype(np.float32)

class CSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
 
    def forward(self, x):
        return self.act(x.real) + 1j * self.act(x.imag)
 
def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat)).astype(np.float64)
 
def RMSE(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2)).astype(np.float64)
 
def MAPE(y, y_hat):
    raw = np.abs(y - y_hat) / (np.abs(y) + 1e-5)
    raw = np.where(raw > 5, 5.0, raw)
    return np.mean(raw).astype(np.float64)

def evaluate(y, y_hat):
    return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)

