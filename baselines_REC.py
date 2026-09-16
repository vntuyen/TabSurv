#!/usr/bin/env python3
"""Treatment-recommendation baselines: 8 predictive models + SurvITE + BITES.

All methods use the same outer split seeds and held-out test patients. SurvITE
and BITES are Python causal-survival / heterogeneous treatment-effect baselines.
Treatment coding is fixed to 1=CHEMOTHERAPY and 0=RADIO_THERAPY.

SurvITE and BITES are implemented directly in this script; no separate causal-model
Python modules are required. BITES still requires the external geomloss package.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest

from datasets import load_datafile_treatment, preprocess_dataset, get_target
from experiment_config import (
    SEEDS, REC_TEST_SIZE, REC_MODEL_RANDOM_STATE,
    REC_ALL_BASELINE_MODELS, REC_PREDICTIVE_BASELINE_MODELS,
    REC_SCENARIOS, BASELINES_REC_RESULTS_DIR, selected_rec_scenarios,
    REC_CAUSAL_HORIZON,
    REC_SURVITE_Z_DIM, REC_SURVITE_REP_HIDDEN, REC_SURVITE_HEAD_HIDDEN,
    REC_SURVITE_REP_LAYERS, REC_SURVITE_HEAD_LAYERS, REC_SURVITE_ACTIVATION,
    REC_SURVITE_DROPOUT, REC_SURVITE_BETA, REC_SURVITE_GAMMA, REC_SURVITE_LR,
    REC_SURVITE_WEIGHT_DECAY, REC_SURVITE_BATCH_SIZE, REC_SURVITE_EPOCHS,
    REC_SURVITE_PATIENCE, REC_SURVITE_CHECK_EVERY, REC_SURVITE_DEVICE,
    REC_BITES_SHARED_LAYERS, REC_BITES_INDIVIDUAL_LAYERS,
    REC_BITES_ALPHA, REC_BITES_BLUR, REC_BITES_LR, REC_BITES_WEIGHT_DECAY,
    REC_BITES_DROPOUT, REC_BITES_EPOCHS, REC_BITES_PATIENCE, REC_BITES_DEVICE,
    compact_prediction_frame, ensure_rec_output_dirs, format_mean_std,
)
from models import get_model, model_dict, fit_coxnet_survival
from utils import get_labtrans, evaluate_model_sksurv, survival_curves, mean_survival_time_km

# ============================================================================
# Embedded causal-survival baseline implementations
# SurvITE and BITES are intentionally defined in this file so baselines_REC.py
# is self-contained and requires no local survite_rec.py / bites_rec.py modules.
# ============================================================================


try:
    from geomloss import SamplesLoss
except Exception as exc:  # BITES is optional unless selected
    SamplesLoss = None
    _GEOMLOSS_IMPORT_ERROR = exc
else:
    _GEOMLOSS_IMPORT_ERROR = None

# ----------------------------- SurvITE ------------------------------------

@dataclass
class SurvITEFitResult:
    best_val_loss: float
    epochs_trained: int
    device: str
    n_time_points: int


def _resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SurvITE requested CUDA but torch.cuda.is_available() is False.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("SurvITE device must be 'auto', 'cpu', or 'cuda'.")
    return requested


def _activation(name: str):
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class _RepresentationNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, n_layers, activation, dropout):
        super().__init__()
        layers = []
        d = in_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.out(self.body(x))


class _HazardHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_time_points, n_layers, activation, dropout):
        super().__init__()
        layers = []
        d = in_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_time_points)

    def forward(self, z):
        # Clamp probabilities away from 0/1 for stable survival likelihood.
        return torch.sigmoid(self.out(self.body(z))).clamp(1e-6, 1.0 - 1e-6)


class _SurvITENet(nn.Module):
    def __init__(self, in_features, z_dim, rep_hidden, head_hidden, rep_layers, head_layers,
                 n_time_points, activation, dropout):
        super().__init__()
        self.phi = _RepresentationNet(
            in_features, rep_hidden, z_dim, rep_layers, activation, dropout
        )
        self.h0 = _HazardHead(
            z_dim, head_hidden, n_time_points, head_layers, activation, dropout
        )
        self.h1 = _HazardHead(
            z_dim, head_hidden, n_time_points, head_layers, activation, dropout
        )

    def forward(self, x):
        z = self.phi(x)
        return z, self.h0(z), self.h1(z)


class SurvITERecommender:
    """SurvITE-style heterogeneous treatment-effect estimator for survival data."""

    def __init__(
        self,
        in_features: int,
        horizon: float = 20.0,
        z_dim: int = 100,
        rep_hidden: int = 100,
        head_hidden: int = 100,
        rep_layers: int = 3,
        head_layers: int = 2,
        activation: str = "elu",
        dropout: float = 0.3,
        beta: float = 1e-3,
        gamma: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        epochs: int = 3000,
        patience: int = 20,
        check_every: int = 25,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.in_features = int(in_features)
        self.horizon = float(horizon)
        self.z_dim = int(z_dim)
        self.rep_hidden = int(rep_hidden)
        self.head_hidden = int(head_hidden)
        self.rep_layers = int(rep_layers)
        self.head_layers = int(head_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.check_every = int(check_every)
        self.device = _resolve_device(device)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.model = None
        self.time_grid = None
        self.fit_result = None

    def _seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _make_time_grid(self, observed_times):
        t = np.asarray(observed_times, dtype=float)
        t = t[np.isfinite(t)]
        if t.size == 0:
            raise ValueError("SurvITE received no finite training times.")
        # Include all distinct observed training times up to the RMST horizon.
        grid = np.unique(t[t <= self.horizon])
        if grid.size == 0 or grid[0] > 0.0:
            grid = np.concatenate([[0.0], grid])
        if grid[-1] < self.horizon:
            grid = np.concatenate([grid, [self.horizon]])
        return np.unique(grid.astype(np.float32))

    def _time_index(self, times):
        idx = np.searchsorted(self.time_grid, np.asarray(times, dtype=float), side="right") - 1
        return np.clip(idx, 0, len(self.time_grid) - 1).astype(np.int64)

    @staticmethod
    def _wasserstein_ipm(z1, z0, lam=10.0, iterations=10):
        """Differentiable Sinkhorn-style Wasserstein discrepancy."""
        if z1.shape[0] < 2 or z0.shape[0] < 2:
            return z1.new_tensor(0.0)
        # Pairwise squared Euclidean distances.
        M = torch.cdist(z1, z0, p=2).pow(2)
        mean_M = M.mean().detach().clamp_min(1e-6)
        eff_lam = float(lam) / mean_M
        K = torch.exp(-eff_lam * M).clamp_min(1e-8)
        a = torch.full((z1.shape[0],), 1.0 / z1.shape[0], device=z1.device, dtype=z1.dtype)
        b = torch.full((z0.shape[0],), 1.0 / z0.shape[0], device=z0.device, dtype=z0.dtype)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(iterations):
            v = b / (K.T @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        transport = u[:, None] * K * v[None, :]
        return torch.sum(transport * M)

    def _loss(self, x, time_idx, event, treatment):
        z, h0, h1 = self.model(x)
        h = torch.where(treatment[:, None].bool(), h1, h0)
        log_h = torch.log(h)
        log_1mh = torch.log1p(-h)

        gather_idx = time_idx[:, None]
        log_h_t = torch.gather(log_h, 1, gather_idx).squeeze(1)
        log_1mh_t = torch.gather(log_1mh, 1, gather_idx).squeeze(1)
        cum_log_surv = torch.cumsum(log_1mh, dim=1)
        cum_t = torch.gather(cum_log_surv, 1, gather_idx).squeeze(1)
        before_t = cum_t - log_1mh_t

        # Event: survive previous bins then fail in observed bin.
        # Censored: survive through the observed censoring bin.
        ll_event = before_t + log_h_t
        ll_censor = cum_t
        nll = -torch.where(event > 0.5, ll_event, ll_censor).mean()

        z1 = z[treatment > 0.5]
        z0 = z[treatment <= 0.5]
        ipm = self._wasserstein_ipm(z1, z0)

        smooth = h.new_tensor(0.0)
        if self.gamma > 0 and h.shape[1] > 1:
            smooth = (h[:, 1:] - h[:, :-1]).pow(2).mean()

        total = nll + self.beta * ipm + self.gamma * smooth
        return total, nll.detach(), ipm.detach(), smooth.detach()

    def _eval_loss(self, X, time_idx, event, treatment):
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            ti = torch.as_tensor(time_idx, dtype=torch.long, device=self.device)
            ev = torch.as_tensor(event, dtype=torch.float32, device=self.device)
            tr = torch.as_tensor(treatment, dtype=torch.float32, device=self.device)
            total, nll, ipm, smooth = self._loss(x, ti, ev, tr)
        return float(total.item()), float(nll.item()), float(ipm.item()), float(smooth.item())

    def fit(self, X_train, time_train, event_train, treatment_train,
            X_val, time_val, event_val, treatment_val):
        self._seed()
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        time_train = np.asarray(time_train, dtype=float)
        time_val = np.asarray(time_val, dtype=float)
        event_train = np.asarray(event_train, dtype=np.float32)
        event_val = np.asarray(event_val, dtype=np.float32)
        treatment_train = np.asarray(treatment_train, dtype=np.float32)
        treatment_val = np.asarray(treatment_val, dtype=np.float32)

        if set(np.unique(treatment_train).astype(int)) != {0, 1}:
            raise ValueError("SurvITE requires both treatment groups in the training split.")

        self.time_grid = self._make_time_grid(time_train)
        train_idx = self._time_index(time_train)
        val_idx = self._time_index(time_val)

        self.model = _SurvITENet(
            self.in_features, self.z_dim, self.rep_hidden, self.head_hidden,
            self.rep_layers, self.head_layers, len(self.time_grid),
            self.activation, self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        n = len(X_train)
        rng = np.random.default_rng(self.random_state)
        best_loss = math.inf
        best_state = None
        stale = 0
        epochs_trained = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            if self.batch_size >= n:
                ids = np.arange(n)
            else:
                ids = rng.choice(n, size=self.batch_size, replace=False)

            # Keep both arms represented in the batch whenever possible.
            if np.unique(treatment_train[ids]).size < 2:
                ids0 = np.flatnonzero(treatment_train == 0)
                ids1 = np.flatnonzero(treatment_train == 1)
                k = min(max(2, self.batch_size // 2), len(ids0), len(ids1))
                if k > 0:
                    ids = np.concatenate([
                        rng.choice(ids0, size=k, replace=False),
                        rng.choice(ids1, size=k, replace=False),
                    ])

            x = torch.as_tensor(X_train[ids], dtype=torch.float32, device=self.device)
            ti = torch.as_tensor(train_idx[ids], dtype=torch.long, device=self.device)
            ev = torch.as_tensor(event_train[ids], dtype=torch.float32, device=self.device)
            tr = torch.as_tensor(treatment_train[ids], dtype=torch.float32, device=self.device)

            optimizer.zero_grad(set_to_none=True)
            total, _, _, _ = self._loss(x, ti, ev, tr)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            epochs_trained = epoch

            if epoch == 1 or epoch % self.check_every == 0 or epoch == self.epochs:
                val_total, val_nll, val_ipm, _ = self._eval_loss(
                    X_val, val_idx, event_val, treatment_val
                )
                if self.verbose and (epoch == 1 or epoch % (self.check_every * 4) == 0):
                    print(
                        f"[SurvITE] epoch={epoch} val={val_total:.5f} "
                        f"nll={val_nll:.5f} ipm={val_ipm:.5f}"
                    )
                if val_total < best_loss - 1e-6:
                    best_loss = val_total
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        if self.verbose:
                            print(f"[SurvITE] early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fit_result = SurvITEFitResult(
            best_val_loss=float(best_loss),
            epochs_trained=int(epochs_trained),
            device=self.device,
            n_time_points=int(len(self.time_grid)),
        )
        return self

    def _survival_curves(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before SurvITE prediction.")
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            _, h0, h1 = self.model(x)
            s0 = torch.cumprod(1.0 - h0, dim=1).cpu().numpy()
            s1 = torch.cumprod(1.0 - h1, dim=1).cpu().numpy()
        return s0, s1

    def _rmst(self, survival):
        grid = np.asarray(self.time_grid, dtype=float)
        # Survival is 1 at time zero before any failure; prepend explicitly.
        if grid[0] == 0.0:
            eval_grid = grid
            surv = survival.copy()
            surv[:, 0] = 1.0
        else:
            eval_grid = np.concatenate([[0.0], grid])
            surv = np.concatenate([np.ones((len(survival), 1)), survival], axis=1)
        trapezoid = getattr(np, "trapezoid", np.trapz)
        return trapezoid(surv, eval_grid, axis=1)

    def recommend(self, X):
        """Return recommendation and counterfactual RMSTs.

        Returns
        -------
        rec_w : int array
            1=CHEMOTHERAPY, 0=RADIO_THERAPY.
        rmst0, rmst1 : arrays
            Predicted RMST under radiotherapy and chemotherapy.
        ite_rmst : array
            rmst1 - rmst0.
        """
        s0, s1 = self._survival_curves(X)
        rmst0 = self._rmst(s0)
        rmst1 = self._rmst(s1)
        ite = rmst1 - rmst0
        rec_w = (ite > 0.0).astype(int)
        return rec_w, rmst0, rmst1, ite

# ------------------------------- BITES ------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class BitesNet(nn.Module):
    def __init__(self, in_features, shared_layers=(15, 10), individual_layers=(10, 5), dropout=0.1):
        super().__init__()
        self.shared = _MLP(in_features, shared_layers, dropout)
        self.head0_body = _MLP(self.shared.out_dim, individual_layers, dropout)
        self.head1_body = _MLP(self.shared.out_dim, individual_layers, dropout)
        self.head0 = nn.Linear(self.head0_body.out_dim, 1)
        self.head1 = nn.Linear(self.head1_body.out_dim, 1)

    def forward(self, x):
        z = self.shared(x)
        r0 = self.head0(self.head0_body(z)).squeeze(-1)
        r1 = self.head1(self.head1_body(z)).squeeze(-1)
        return r0, r1, z


def _cox_ph_loss(log_risk, times, events):
    """Negative Cox partial log-likelihood, Breslow handling of ties."""
    if log_risk.numel() == 0:
        return log_risk.sum() * 0.0
    order = torch.argsort(times, descending=True)
    r = log_risk[order]
    e = events[order]
    log_cum_risk = torch.logcumsumexp(r, dim=0)
    n_events = torch.clamp(e.sum(), min=1.0)
    return -torch.sum((r - log_cum_risk) * e) / n_events


def _breslow_baseline(times, events, log_risk):
    """Return event-time grid and Breslow cumulative baseline hazard."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    log_risk = np.asarray(log_risk, dtype=float)
    event_times = np.sort(np.unique(times[events == 1]))
    if event_times.size == 0:
        return np.array([0.0]), np.array([0.0])
    exp_risk = np.exp(np.clip(log_risk, -50, 50))
    increments = []
    for t in event_times:
        d = np.sum((times == t) & (events == 1))
        denom = np.sum(exp_risk[times >= t])
        increments.append(float(d / denom) if denom > 0 else 0.0)
    return event_times, np.cumsum(np.asarray(increments, dtype=float))


def _median_survival(log_risk, event_times, cum_h0, horizon):
    log_risk = np.asarray(log_risk, dtype=float).reshape(-1)
    out = np.full(log_risk.shape[0], float(horizon), dtype=float)
    if len(event_times) == 0:
        return out
    hr = np.exp(np.clip(log_risk, -50, 50))
    for i, h in enumerate(hr):
        surv = np.exp(-cum_h0 * h)
        idx = np.flatnonzero(surv <= 0.5)
        if idx.size:
            out[i] = min(float(event_times[idx[0]]), float(horizon))
    return out


@dataclass
class BitesFitResult:
    model: BitesNet
    baseline0: tuple[np.ndarray, np.ndarray]
    baseline1: tuple[np.ndarray, np.ndarray]
    best_val_loss: float
    epochs_trained: int
    device: str


class BitesRecommender:
    def __init__(
        self,
        in_features,
        shared_layers=(15, 10),
        individual_layers=(10, 5),
        alpha=0.1,
        blur=0.05,
        lr=1e-3,
        weight_decay=0.2,
        dropout=0.1,
        epochs=1000,
        patience=50,
        horizon=20.0,
        device="auto",
        random_state=42,
    ):
        if SamplesLoss is None:
            raise ImportError(
                "BITES requires geomloss. Install it in the Gadi environment with "
                "`pip install geomloss`. Original import error: " + str(_GEOMLOSS_IMPORT_ERROR)
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BITES requested CUDA but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.random_state = int(random_state)
        self.horizon = float(horizon)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.model = BitesNet(in_features, shared_layers, individual_layers, dropout).to(self.device)
        self.alpha = float(alpha)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        self.sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur))
        self.fit_result = None

    @staticmethod
    def _tensor(x, dtype=torch.float32, device=None):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    def _loss(self, X, time, event, treatment):
        r0, r1, z = self.model(X)
        m0 = treatment < 0.5
        m1 = treatment >= 0.5
        if not torch.any(m0) or not torch.any(m1):
            raise ValueError("BITES requires both treatment groups in the training split.")
        loss0 = _cox_ph_loss(r0[m0], time[m0], event[m0])
        loss1 = _cox_ph_loss(r1[m1], time[m1], event[m1])
        balance = self.sinkhorn(z[m0], z[m1])
        return loss0 + loss1 + self.alpha * balance

    def fit(self, X_train, time_train, event_train, treatment_train,
            X_val, time_val, event_val, treatment_val):
        Xtr = self._tensor(X_train, device=self.device)
        Ttr = self._tensor(time_train, device=self.device)
        Etr = self._tensor(event_train, device=self.device)
        Wtr = self._tensor(treatment_train, device=self.device)
        Xva = self._tensor(X_val, device=self.device)
        Tva = self._tensor(time_val, device=self.device)
        Eva = self._tensor(event_val, device=self.device)
        Wva = self._tensor(treatment_val, device=self.device)

        best_state = None
        best_val = float("inf")
        stale = 0
        epochs_trained = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(Xtr, Ttr, Etr, Wtr)
            if not torch.isfinite(loss):
                raise RuntimeError(f"BITES training loss became non-finite at epoch {epoch}")
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val = float(self._loss(Xva, Tva, Eva, Wva).detach().cpu())
            epochs_trained = epoch
            if val < best_val - 1e-6:
                best_val = val
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        # Treatment-specific Breslow baseline hazards from factual training heads.
        with torch.no_grad():
            r0, r1, _ = self.model(Xtr)
        r0 = r0.cpu().numpy(); r1 = r1.cpu().numpy()
        w = np.asarray(treatment_train).astype(int)
        t = np.asarray(time_train, dtype=float)
        e = np.asarray(event_train).astype(int)
        b0 = _breslow_baseline(t[w == 0], e[w == 0], r0[w == 0])
        b1 = _breslow_baseline(t[w == 1], e[w == 1], r1[w == 1])
        self.fit_result = BitesFitResult(
            self.model, b0, b1, best_val, epochs_trained, str(self.device)
        )
        return self

    def predict_counterfactual_median(self, X):
        if self.fit_result is None:
            raise RuntimeError("BITES model is not fitted")
        Xt = self._tensor(X, device=self.device)
        self.model.eval()
        with torch.no_grad():
            r0, r1, _ = self.model(Xt)
        r0 = r0.cpu().numpy(); r1 = r1.cpu().numpy()
        med0 = _median_survival(r0, *self.fit_result.baseline0, self.horizon)
        med1 = _median_survival(r1, *self.fit_result.baseline1, self.horizon)
        return med0, med1, r0, r1

    def recommend(self, X):
        med0, med1, r0, r1 = self.predict_counterfactual_median(X)
        rec = (med1 > med0).astype(int)
        # Ties default to control (0), deterministic and conservative.
        return rec, med0, med1, r0, r1

COXNET_MODEL_NAME = "ENCox"
L1_RATIOS = (0.5,)
CV_FOLDS = 3


def _as_float(value):
    try:
        return float(value) if value is not None and np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _treatment_vector(df, treatments):
    if set(treatments) != {"CHEMOTHERAPY", "RADIO_THERAPY"}:
        raise ValueError(f"Expected CHEMOTHERAPY/RADIO_THERAPY, got {treatments}")
    chemo = pd.to_numeric(df["CHEMOTHERAPY"], errors="coerce").fillna(0).to_numpy()
    radio = pd.to_numeric(df["RADIO_THERAPY"], errors="coerce").fillna(0).to_numpy()
    valid = ((chemo == 1) & (radio == 0)) | ((chemo == 0) & (radio == 1))
    if not np.all(valid):
        bad = int((~valid).sum())
        raise ValueError(
            f"Treatment-recommendation cohort contains {bad} rows that are not exactly one of "
            "CHEMOTHERAPY or RADIO_THERAPY. Filter these rows before running causal baselines."
        )
    return chemo.astype(int)  # 1=chemotherapy, 0=radiotherapy


def _current_tp_from_w(w):
    w = np.asarray(w).astype(int)
    return np.where(w == 1, "CHEMOTHERAPY", "RADIO_THERAPY")


def _finalize_policy_result(dataset_name, model_name, split_seed, df_test, durations_test,
                            events_test, current_tp, rec_tp, policy_score,
                            prediction_dir, plots_dir, extra_cols=None, c_index=np.nan):
    pred_df = compact_prediction_frame(
        durations_test, events_test, policy_score, "risk_score",
        patient_ids=df_test.index.to_numpy(),
    )
    pred_df["REC_TP"] = np.asarray(rec_tp, dtype=object)
    pred_df["CURRENT_TP"] = np.asarray(current_tp, dtype=object)
    pred_df["FOLLOW_REC"] = (pred_df["REC_TP"] == pred_df["CURRENT_TP"]).astype(int)
    if extra_cols:
        for k, v in extra_cols.items():
            pred_df[k] = np.asarray(v)
    pred_path = prediction_dir / f"{dataset_name}_{model_name}_seed{split_seed}_predict.csv"
    pred_df.to_csv(pred_path, index=False)

    combined = pred_df.copy()
    plot_path = plots_dir / f"{dataset_name}_{model_name}_seed{split_seed}_risk_KM_plot.png"
    p_value = survival_curves(combined, "time", "event", f"{model_name} (seed={split_seed})", str(plot_path))
    follow_df = combined[combined["FOLLOW_REC"] == 1]
    not_follow_df = combined[combined["FOLLOW_REC"] == 0]
    mean_follow = mean_survival_time_km(follow_df["time"], follow_df["event"]) if len(follow_df) else np.nan
    mean_not = mean_survival_time_km(not_follow_df["time"], not_follow_df["event"]) if len(not_follow_df) else np.nan

    return {
        "seed": split_seed, "dataset": dataset_name, "model": model_name,
        "c_index": _as_float(c_index), "p_value": _as_float(p_value),
        "mean_survival_followed": _as_float(mean_follow),
        "mean_survival_not_followed": _as_float(mean_not),
        "delta_mean_survival": _as_float(mean_follow - mean_not),
        "n_followed": int(len(follow_df)), "n_not_followed": int(len(not_follow_df)),
        "prediction_file": str(pred_path),
    }


def recommend_treatment_baseline(model, X_test, treatment_plans, model_name, train_features):
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=train_features)
    predicted = pd.DataFrame(index=X_test.index)
    for tp in treatment_plans:
        X_cf = X_test.copy()
        for col in treatment_plans:
            if col in X_cf.columns:
                X_cf[col] = 0
        if tp in X_cf.columns:
            X_cf[tp] = 1
        X_cf = X_cf[train_features].astype(np.float32)
        if model_name in ("RSF", COXNET_MODEL_NAME):
            pred = model.predict(X_cf.values)
        else:
            surv = model.predict_surv_df(X_cf.values)
            t = surv.index.values
            exp_surv = np.trapz(surv.to_numpy().T, t, axis=1)
            pred = -exp_surv
        predicted[tp] = pred
    predicted["REC_TP"] = predicted.idxmin(axis=1)
    return predicted


def _run_predictive(dataset_name, model_name, split_seed, prediction_dir, results_dir,
                    plots_dir, test_size, full_reseed):
    model_seed = split_seed if full_reseed else REC_MODEL_RANDOM_STATE
    np.random.seed(split_seed); torch.manual_seed(model_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(model_seed)

    df, cols_standardize, cols_leave, duration_col, event_col, feature_names, treatments = load_datafile_treatment(dataset_name)
    df_train, df_val, df_test, x_train, x_val, x_test, _ = preprocess_dataset(
        df, cols_standardize, cols_leave, duration_col, event_col, test_size, split_seed
    )
    durations_test, events_test = get_target(df_test, duration_col, event_col)

    if model_name in ("RSF", COXNET_MODEL_NAME):
        durations_train, events_train = get_target(df_train, duration_col, event_col)
    else:
        model_class = model_dict[model_name]
        labtrans = get_labtrans(model_class, 10) if model_name in ("LH", "PMF", "DeepHS", "PCHazard", "MTLR") else None
        if labtrans is not None:
            y_train = labtrans.fit_transform(*get_target(df_train, duration_col, event_col))
            y_val = labtrans.transform(*get_target(df_val, duration_col, event_col))
            model = get_model(model_name, x_train.shape[1], labtrans.out_features, labtrans)
        else:
            y_train = get_target(df_train, duration_col, event_col)
            y_val = get_target(df_val, duration_col, event_col)
            model = get_model(model_name, x_train.shape[1])

    if model_name == "RSF":
        model = RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=15,
                                     max_features="sqrt", n_jobs=-1, random_state=model_seed)
        y = np.array([(bool(e), t) for e, t in zip(df_train[event_col], df_train[duration_col])],
                     dtype=[(event_col, "bool"), (duration_col, "float")])
        model.fit(x_train, y)
    elif model_name == COXNET_MODEL_NAME:
        model, selection = fit_coxnet_survival(x_train, durations_train, events_train,
                                                l1_ratios=L1_RATIOS, cv_folds=CV_FOLDS,
                                                random_state=model_seed)
        # pd.DataFrame([{"dataset": dataset_name, "seed": split_seed, **selection}]).to_csv(
        #     results_dir / f"{dataset_name}_{COXNET_MODEL_NAME}_seed{split_seed}_hyperparameters.csv", index=False)
    else:
        model.fit(x_train, y_train, batch_size=256, epochs=100,
                  callbacks=[tt.cb.EarlyStopping()], val_data=(x_val, y_val))

    metrics = evaluate_model_sksurv(model, x_test, durations_test, events_test, model_name, feature_names)
    c_index = float(metrics["c_index"][0])
    risk_scores = metrics["df_results"]["risk_score"].to_numpy(dtype=float)

    train_features = df_train.drop(columns=[duration_col, event_col]).columns.tolist()
    x_test_raw = df_test[train_features].values.astype(np.float32)
    rec_df = recommend_treatment_baseline(model, x_test_raw, treatments, model_name, train_features)
    current_tp = df_test[treatments].idxmax(axis=1).to_numpy(dtype=object)
    return _finalize_policy_result(dataset_name, model_name, split_seed, df_test, durations_test,
                                   events_test, current_tp, rec_df["REC_TP"].to_numpy(), risk_scores,
                                   prediction_dir, plots_dir, c_index=c_index)


def _causal_split(dataset_name, split_seed, test_size):
    # Reuse preprocess_dataset solely to guarantee identical row partitions to existing baselines.
    df, cols_standardize, cols_leave, duration_col, event_col, feature_names, treatments = load_datafile_treatment(dataset_name)
    df_train, df_val, df_test, *_ = preprocess_dataset(
        df, cols_standardize, cols_leave, duration_col, event_col, test_size, split_seed
    )
    return df_train, df_val, df_test, duration_col, event_col, treatments


def _run_survite(dataset_name, split_seed, prediction_dir, results_dir, plots_dir, test_size, full_reseed):
    model_seed = split_seed if full_reseed else REC_MODEL_RANDOM_STATE
    df_train, df_val, df_test, time_col, event_col, treatments = _causal_split(
        dataset_name, split_seed, test_size
    )
    covariates = [c for c in df_train.columns if c not in [time_col, event_col, *treatments]]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(df_train[covariates]).astype(np.float32)
    Xva = scaler.transform(df_val[covariates]).astype(np.float32)
    Xte = scaler.transform(df_test[covariates]).astype(np.float32)
    wtr = _treatment_vector(df_train, treatments)
    wva = _treatment_vector(df_val, treatments)
    wte = _treatment_vector(df_test, treatments)

    model = SurvITERecommender(
        in_features=Xtr.shape[1],
        horizon=REC_CAUSAL_HORIZON,
        z_dim=REC_SURVITE_Z_DIM,
        rep_hidden=REC_SURVITE_REP_HIDDEN,
        head_hidden=REC_SURVITE_HEAD_HIDDEN,
        rep_layers=REC_SURVITE_REP_LAYERS,
        head_layers=REC_SURVITE_HEAD_LAYERS,
        activation=REC_SURVITE_ACTIVATION,
        dropout=REC_SURVITE_DROPOUT,
        beta=REC_SURVITE_BETA,
        gamma=REC_SURVITE_GAMMA,
        lr=REC_SURVITE_LR,
        weight_decay=REC_SURVITE_WEIGHT_DECAY,
        batch_size=REC_SURVITE_BATCH_SIZE,
        epochs=REC_SURVITE_EPOCHS,
        patience=REC_SURVITE_PATIENCE,
        check_every=REC_SURVITE_CHECK_EVERY,
        device=REC_SURVITE_DEVICE,
        random_state=model_seed,
        verbose=True,
    ).fit(
        Xtr, df_train[time_col], df_train[event_col], wtr,
        Xva, df_val[time_col], df_val[event_col], wva,
    )

    rec_w, rmst0, rmst1, ite_rmst = model.recommend(Xte)
    rec_tp = _current_tp_from_w(rec_w)
    current_tp = _current_tp_from_w(wte)

    # Factual predicted RMST provides a survival-oriented risk score for the
    # descriptive C-index; recommendation itself uses counterfactual RMST.
    factual_rmst = np.where(wte == 1, rmst1, rmst0)
    risk_score = -factual_rmst
    from sksurv.metrics import concordance_index_censored
    cidx = concordance_index_censored(
        df_test[event_col].astype(bool).to_numpy(),
        df_test[time_col].astype(float).to_numpy(),
        risk_score,
    )[0]

    result = _finalize_policy_result(
        dataset_name, "SurvITE", split_seed, df_test,
        df_test[time_col].to_numpy(), df_test[event_col].to_numpy(),
        current_tp, rec_tp, risk_score, prediction_dir, plots_dir,
        extra_cols={
            "rmst_radio": rmst0,
            "rmst_chemo": rmst1,
            "ITE_RMST": ite_rmst,
        },
        c_index=cidx,
    )
    result["survite_best_val_loss"] = model.fit_result.best_val_loss
    result["survite_epochs_trained"] = model.fit_result.epochs_trained
    result["survite_device"] = model.fit_result.device
    result["survite_n_time_points"] = model.fit_result.n_time_points
    return result


def _run_bites(dataset_name, split_seed, prediction_dir, results_dir, plots_dir, test_size, full_reseed):
    model_seed = split_seed if full_reseed else REC_MODEL_RANDOM_STATE
    df_train, df_val, df_test, time_col, event_col, treatments = _causal_split(dataset_name, split_seed, test_size)
    covariates = [c for c in df_train.columns if c not in [time_col, event_col, *treatments]]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(df_train[covariates]).astype(np.float32)
    Xva = scaler.transform(df_val[covariates]).astype(np.float32)
    Xte = scaler.transform(df_test[covariates]).astype(np.float32)
    wtr = _treatment_vector(df_train, treatments); wva = _treatment_vector(df_val, treatments); wte = _treatment_vector(df_test, treatments)

    model = BitesRecommender(
        in_features=Xtr.shape[1], shared_layers=REC_BITES_SHARED_LAYERS,
        individual_layers=REC_BITES_INDIVIDUAL_LAYERS, alpha=REC_BITES_ALPHA,
        blur=REC_BITES_BLUR, lr=REC_BITES_LR, weight_decay=REC_BITES_WEIGHT_DECAY,
        dropout=REC_BITES_DROPOUT, epochs=REC_BITES_EPOCHS, patience=REC_BITES_PATIENCE,
        horizon=REC_CAUSAL_HORIZON, device=REC_BITES_DEVICE, random_state=model_seed,
    ).fit(Xtr, df_train[time_col], df_train[event_col], wtr,
          Xva, df_val[time_col], df_val[event_col], wva)

    rec_w, med0, med1, r0, r1 = model.recommend(Xte)
    rec_tp = _current_tp_from_w(rec_w); current_tp = _current_tp_from_w(wte)
    factual_median = np.where(wte == 1, med1, med0)
    risk_score = -factual_median
    from sksurv.metrics import concordance_index_censored
    cidx = concordance_index_censored(df_test[event_col].astype(bool).to_numpy(),
                                      df_test[time_col].astype(float).to_numpy(), risk_score)[0]
    ite_median = med1 - med0
    result = _finalize_policy_result(
        dataset_name, "BITES", split_seed, df_test, df_test[time_col].to_numpy(), df_test[event_col].to_numpy(),
        current_tp, rec_tp, risk_score, prediction_dir, plots_dir,
        extra_cols={"median_survival_radio": med0, "median_survival_chemo": med1,
                    "ITE_median_survival": ite_median}, c_index=cidx,
    )
    result["bites_best_val_loss"] = model.fit_result.best_val_loss
    result["bites_epochs_trained"] = model.fit_result.epochs_trained
    result["bites_device"] = model.fit_result.device
    return result


def run_one_seed_one_model(dataset_name, model_name, split_seed, prediction_dir, results_dir,
                           plots_dir, test_size, full_reseed=False):
    if model_name == "SurvITE":
        row = _run_survite(dataset_name, split_seed, prediction_dir, results_dir, plots_dir, test_size, full_reseed)
    elif model_name == "BITES":
        row = _run_bites(dataset_name, split_seed, prediction_dir, results_dir, plots_dir, test_size, full_reseed)
    elif model_name in REC_PREDICTIVE_BASELINE_MODELS:
        row = _run_predictive(dataset_name, model_name, split_seed, prediction_dir, results_dir, plots_dir, test_size, full_reseed)
    else:
        raise ValueError(f"Unknown REC model: {model_name}")
    row["model_seed"] = split_seed if full_reseed else REC_MODEL_RANDOM_STATE
    row["test_size"] = test_size
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        choices=["72genes", "allgenes", "both"],
        default="both",
        help="REC gene-set scenario. Default: both.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional custom dataset override for a single REC scenario. Normally omit this "
            "and use --scenario so 72genes/allgenes come from experiment_config.py."
        ),
    )
    p.add_argument("--models", nargs="+", choices=REC_ALL_BASELINE_MODELS, default=None)
    p.add_argument("--test-size", type=float, default=REC_TEST_SIZE)
    p.add_argument("--full-reseed", action="store_true")
    return p.parse_args()


def _run_rec_scenario(rec_scenario, seeds, models, test_size, full_reseed, datasets_override=None):
    dirs = ensure_rec_output_dirs(rec_scenario, "baselines_REC")
    prediction_dir = dirs["prediction_dir"]
    results_dir = dirs["results_dir"]
    plots_dir = dirs["plots_dir"]
    datasets = list(datasets_override) if datasets_override is not None else list(dirs["datasets"])

    metadata = {
        "method": "baselines_REC",
        "rec_scenario": rec_scenario,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "test_size": test_size,
        "model_random_state": REC_MODEL_RANDOM_STATE,
        "full_reseed": full_reseed,
        "treatment_1": "CHEMOTHERAPY",
        "treatment_0": "RADIO_THERAPY",
        "causal_RMST_horizon": REC_CAUSAL_HORIZON,
        "SurvITE_epochs": REC_SURVITE_EPOCHS,
        "SurvITE_beta": REC_SURVITE_BETA,
    }
    (results_dir / "baselines_REC_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rows = []
    print("\n" + "=" * 78)
    print(f"baselines_REC | scenario={rec_scenario}")
    print(f"datasets  : {datasets}")
    print(f"models    : {models}")
    print(f"seeds     : {seeds}")
    print(f"test_size : {test_size}")
    print("=" * 78)

    for dataset in datasets:
        for model in models:
            for seed in seeds:
                print(f"\n=== baselines_REC | scenario={rec_scenario} | {dataset} | {model} | seed={seed} ===")
                try:
                    row = run_one_seed_one_model(
                        dataset, model, seed, prediction_dir, results_dir, plots_dir,
                        test_size, full_reseed,
                    )
                    row["rec_scenario"] = rec_scenario
                    print(
                        f"C-index={row['c_index'] if np.isfinite(row['c_index']) else 'N/A'}, "
                        f"Delta mean survival={row['delta_mean_survival']:.3f}"
                    )
                except Exception as exc:
                    print(f"[ERROR] {dataset}/{model}/seed={seed}: {type(exc).__name__}: {exc}")
                    row = {
                        "rec_scenario": rec_scenario,
                        "seed": seed,
                        "dataset": dataset,
                        "model": model,
                        "model_seed": seed if full_reseed else REC_MODEL_RANDOM_STATE,
                        "test_size": test_size,
                        "c_index": np.nan,
                        "p_value": np.nan,
                        "mean_survival_followed": np.nan,
                        "mean_survival_not_followed": np.nan,
                        "delta_mean_survival": np.nan,
                        "n_followed": np.nan,
                        "n_not_followed": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)

    raw = pd.DataFrame(rows)
    raw_path = results_dir / "baselines_REC_all_seeds.csv"
    raw.to_csv(raw_path, index=False)

    summary = []
    for (dataset, model), g in raw.groupby(["dataset", "model"], sort=False):
        d = pd.to_numeric(g["delta_mean_survival"], errors="coerce")
        d = d[np.isfinite(d)]
        c = pd.to_numeric(g["c_index"], errors="coerce")
        c = c[np.isfinite(c)]
        dm = float(d.mean()) if len(d) else np.nan
        ds = float(d.std(ddof=1)) if len(d) > 1 else (0.0 if len(d) == 1 else np.nan)
        cm = float(c.mean()) if len(c) else np.nan
        cs = float(c.std(ddof=1)) if len(c) > 1 else (0.0 if len(c) == 1 else np.nan)
        summary.append({
            "rec_scenario": rec_scenario,
            "dataset": dataset,
            "model": model,
            "n_seeds": int(len(d)),
            "mean_C-index": cm,
            "std_C-index": cs,
            "C-index (mean ± std)": format_mean_std(cm, cs),
            "mean_delta_mean_survival": dm,
            "std_delta_mean_survival": ds,
            "Delta mean survival (mean ± std)": format_mean_std(dm, ds, ndigits=3),
        })

    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df["model"] = pd.Categorical(summary_df["model"], categories=models, ordered=True)
        summary_df = summary_df.sort_values(["dataset", "model"]).reset_index(drop=True)
        summary_df["model"] = summary_df["model"].astype(str)

    summary_path = results_dir / "baselines_REC_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {raw_path}\nSaved: {summary_path}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    return raw, summary_df


def main():
    args = parse_args()
    seeds = list(SEEDS if args.seeds is None else args.seeds)
    models = list(REC_ALL_BASELINE_MODELS if args.models is None else args.models)
    scenarios = selected_rec_scenarios(args.scenario)

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1")
    if not seeds:
        raise SystemExit("At least one seed is required")
    if args.datasets is not None and len(scenarios) != 1:
        raise SystemExit("--datasets can only be used with a single --scenario (72genes or allgenes).")

    all_raw = []
    all_summary = []
    for rec_scenario in scenarios:
        raw, summary = _run_rec_scenario(
            rec_scenario,
            seeds,
            models,
            args.test_size,
            args.full_reseed,
            datasets_override=args.datasets,
        )
        all_raw.append(raw)
        all_summary.append(summary)

    if len(scenarios) > 1:
        BASELINES_REC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_raw = pd.concat(all_raw, ignore_index=True, sort=False)
        combined_summary = pd.concat(all_summary, ignore_index=True, sort=False)
        combined_raw_path = BASELINES_REC_RESULTS_DIR / "baselines_REC_all_scenarios_all_seeds.csv"
        combined_summary_path = BASELINES_REC_RESULTS_DIR / "baselines_REC_all_scenarios_summary.csv"
        combined_raw.to_csv(combined_raw_path, index=False)
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"\nSaved combined REC baseline results: {combined_raw_path}")
        print(f"Saved combined REC baseline summary: {combined_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
