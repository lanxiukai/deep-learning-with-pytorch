import os

import torch
from torch import nn

from ._ebm_types import DBMFineTuneMetrics


class DBM(nn.Module):
    """Two-layer Bernoulli Deep Boltzmann Machine with manual optimizer state."""

    def __init__(self, n_visible: int, n_hidden1: int, n_hidden2: int, device: torch.device):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.device = device
        self.W1 = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden1, device=device))
        self.bv = nn.Parameter(torch.zeros(n_visible, device=device))
        self.bh1 = nn.Parameter(torch.zeros(n_hidden1, device=device))
        self.W2 = nn.Parameter(0.01 * torch.randn(n_hidden1, n_hidden2, device=device))
        self.bh2 = nn.Parameter(torch.zeros(n_hidden2, device=device))
        for name, parameter in (("W1_vel", self.W1), ("bv_vel", self.bv), ("bh1_vel", self.bh1), ("W2_vel", self.W2), ("bh2_vel", self.bh2)):
            self.register_buffer(name, torch.zeros_like(parameter.data))
        self.W1_vel = self.get_buffer("W1_vel")
        self.bv_vel = self.get_buffer("bv_vel")
        self.bh1_vel = self.get_buffer("bh1_vel")
        self.W2_vel = self.get_buffer("W2_vel")
        self.bh2_vel = self.get_buffer("bh2_vel")
        self.pcd_v: torch.Tensor | None = None
        self.pcd_h1: torch.Tensor | None = None
        self.pcd_h2: torch.Tensor | None = None
        self.adam_step = 0
        self.adam_m_W1: torch.Tensor | None = None
        self.adam_v_W1: torch.Tensor | None = None
        self.adam_m_W2: torch.Tensor | None = None
        self.adam_v_W2: torch.Tensor | None = None
        self.adam_m_bv: torch.Tensor | None = None
        self.adam_v_bv: torch.Tensor | None = None
        self.adam_m_bh1: torch.Tensor | None = None
        self.adam_v_bh1: torch.Tensor | None = None
        self.adam_m_bh2: torch.Tensor | None = None
        self.adam_v_bh2: torch.Tensor | None = None

    @torch.no_grad()
    def prob_h1_given_v_h2(self, v: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bh1 + v @ self.W1 + h2 @ self.W2.t())

    @torch.no_grad()
    def prob_h2_given_h1(self, h1: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bh2 + h1 @ self.W2)

    @torch.no_grad()
    def prob_v_given_h1(self, h1: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bv + h1 @ self.W1.t())

    @torch.no_grad()
    def mean_field(self, v: torch.Tensor, n_steps: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        mu2 = 0.5 * torch.ones(v.size(0), self.n_hidden2, device=self.device)
        mu1 = self.prob_h1_given_v_h2(v, mu2)
        mu2 = self.prob_h2_given_h1(mu1)
        for _ in range(n_steps - 1):
            mu1 = self.prob_h1_given_v_h2(v, mu2)
            mu2 = self.prob_h2_given_h1(mu1)
        return mu1, mu2

    @torch.no_grad()
    def gibbs_step(self, v: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p_h2 = self.prob_h2_given_h1(h1)
        h2 = torch.bernoulli(p_h2)
        p_h1 = self.prob_h1_given_v_h2(v, h2)
        h1 = torch.bernoulli(p_h1)
        p_v = self.prob_v_given_h1(h1)
        v = torch.bernoulli(p_v)
        return v, h1, h2, p_v, p_h1, p_h2

    @torch.no_grad()
    def finetune_step(self, v0: torch.Tensor, lr: float, momentum: float, weight_decay: float, mf_steps: int = 5, *, pcd_steps: int = 1, use_adam: bool = False, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_eps: float = 1e-8, max_grad_norm: float = 0.0, max_w_norm: float = 0.0, use_pcd: bool = True) -> DBMFineTuneMetrics:
        batch_size = v0.size(0)
        mu1, mu2 = self.mean_field(v0, n_steps=mf_steps)
        pos_W1, pos_W2 = v0.t() @ mu1 / batch_size, mu1.t() @ mu2 / batch_size
        pos_bv, pos_bh1, pos_bh2 = v0.mean(0), mu1.mean(0), mu2.mean(0)
        if use_pcd:
            if self.pcd_v is None or self.pcd_h1 is None or self.pcd_h2 is None or self.pcd_v.size(0) != batch_size:
                v_neg = torch.bernoulli(0.5 * torch.ones(batch_size, self.n_visible, device=self.device))
                h1_neg = torch.bernoulli(0.5 * torch.ones(batch_size, self.n_hidden1, device=self.device))
                h2_neg = torch.bernoulli(0.5 * torch.ones(batch_size, self.n_hidden2, device=self.device))
            else:
                v_neg, h1_neg, h2_neg = self.pcd_v, self.pcd_h1, self.pcd_h2
        else:
            mu1_init, mu2_init = self.mean_field(v0, n_steps=mf_steps)
            v_neg, h1_neg, h2_neg = v0.clone(), torch.bernoulli(mu1_init), torch.bernoulli(mu2_init)
        v_neg, h1_neg, h2_neg, p_v_neg, p_h1_neg, p_h2_neg = self.gibbs_step(v_neg, h1_neg, h2_neg)
        for _ in range(max(1, pcd_steps) - 1):
            v_neg, h1_neg, h2_neg, p_v_neg, p_h1_neg, p_h2_neg = self.gibbs_step(v_neg, h1_neg, h2_neg)
        if use_pcd:
            self.pcd_v, self.pcd_h1, self.pcd_h2 = v_neg, h1_neg, h2_neg
        dW1 = pos_W1 - v_neg.t() @ p_h1_neg / batch_size - weight_decay * self.W1.data
        dW2 = pos_W2 - h1_neg.t() @ p_h2_neg / batch_size - weight_decay * self.W2.data
        dbv, dbh1, dbh2 = pos_bv - p_v_neg.mean(0), pos_bh1 - p_h1_neg.mean(0), pos_bh2 - p_h2_neg.mean(0)
        if max_grad_norm > 0:
            total_norm = torch.sqrt(dW1.square().sum() + dW2.square().sum() + dbv.square().sum() + dbh1.square().sum() + dbh2.square().sum())
            clip_coef = max_grad_norm / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for gradient in (dW1, dW2, dbv, dbh1, dbh2):
                    gradient.mul_(clip_coef)
        if use_adam:
            self.adam_step += 1
            states = (("adam_m_W1", "adam_v_W1", self.W1, dW1), ("adam_m_W2", "adam_v_W2", self.W2, dW2), ("adam_m_bv", "adam_v_bv", self.bv, dbv), ("adam_m_bh1", "adam_v_bh1", self.bh1, dbh1), ("adam_m_bh2", "adam_v_bh2", self.bh2, dbh2))
            for mean_name, variance_name, parameter, gradient in states:
                mean = getattr(self, mean_name)
                variance = getattr(self, variance_name)
                if mean is None:
                    mean = torch.zeros_like(parameter.data)
                if variance is None:
                    variance = torch.zeros_like(parameter.data)
                mean.mul_(adam_beta1).add_(gradient, alpha=1 - adam_beta1)
                variance.mul_(adam_beta2).addcmul_(gradient, gradient, value=1 - adam_beta2)
                parameter.data.addcdiv_(mean / (1 - adam_beta1**self.adam_step), (variance / (1 - adam_beta2**self.adam_step)).sqrt().add_(adam_eps), value=lr)
                setattr(self, mean_name, mean)
                setattr(self, variance_name, variance)
        else:
            for velocity, gradient, parameter in ((self.W1_vel, dW1, self.W1), (self.W2_vel, dW2, self.W2), (self.bv_vel, dbv, self.bv), (self.bh1_vel, dbh1, self.bh1), (self.bh2_vel, dbh2, self.bh2)):
                velocity.mul_(momentum).add_(lr * gradient)
                parameter.data.add_(velocity)
        if max_w_norm > 0:
            for weight in (self.W1, self.W2):
                weight.data.mul_(torch.clamp(max_w_norm / (torch.linalg.norm(weight.data, dim=0) + 1e-8), max=1.0).unsqueeze(0))
        mse = ((v0 - self.prob_v_given_h1(torch.bernoulli(mu1))) ** 2).mean().item()
        return {"mse": mse}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"W1": self.W1.data.cpu(), "bv": self.bv.data.cpu(), "bh1": self.bh1.data.cpu(), "W2": self.W2.data.cpu(), "bh2": self.bh2.data.cpu(), "n_visible": self.n_visible, "n_hidden1": self.n_hidden1, "n_hidden2": self.n_hidden2}, path)
        print(f"Saved DBM checkpoint → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "DBM":
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["n_visible"], ckpt["n_hidden1"], ckpt["n_hidden2"], device)
        model.W1.data.copy_(ckpt["W1"].to(device))
        model.bv.data.copy_(ckpt["bv"].to(device))
        model.bh1.data.copy_(ckpt["bh1"].to(device))
        model.W2.data.copy_(ckpt["W2"].to(device))
        model.bh2.data.copy_(ckpt["bh2"].to(device))
        return model
