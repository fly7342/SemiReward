"""Utilities for building a semi-supervised regression pipeline with semilearn.

The helpers in this module follow the workflow illustrated in the semilearn
notebooks and documentation.  They expose a convenient API to assemble
configurations, algorithms, datasets, and data loaders directly from the
semilearn package instead of re-implementing the training loop from scratch.

The module provides

* :class:`RegressionLabelScaler` – encodes continuous regression targets into a
  bounded range so that semilearn's regression losses work out-of-the box.
* :func:`build_semilearn_regression_pipeline` – prepares a configurable
  pipeline (config, algorithm, data loaders, scaler) for any tabular regression
  dataset and model builder.
* :class:`RegressionTrainer` – a light-weight trainer that calls the selected
  semilearn algorithm's ``train_step`` method while handling optimization and
  evaluation on CPU or GPU environments.
* :func:`run_polynomial_regression_example` – an executable example that
  generates a random polynomial regression dataset and trains a small MLP with
  pseudo-labeling.

The example defaults to the ``pseudolabel`` algorithm for broad device
compatibility.  To use the SemiReward variant ``srpseudolabel`` simply pass the
algorithm name to :func:`build_semilearn_regression_pipeline` (a CUDA capable
GPU is required because the reference implementation relies on CUDA ops).
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "The semilearn regression pipeline requires numpy. Please install numpy to run the example."
    ) from exc

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from semilearn import get_algorithm, get_config
from semilearn.core.utils import get_data_loader
from semilearn.datasets import split_ssl_data


@dataclass
class NetBuilderSpec:
    """Specification for constructing a semilearn-compatible model."""

    builder: Callable[[int, bool, Optional[str]], nn.Module]
    feature_dim: int


@dataclass
class RegressionPipelineArtifacts:
    """Return type for :func:`build_semilearn_regression_pipeline`."""

    config: object
    algorithm: object
    train_lb_loader: torch.utils.data.DataLoader
    train_ulb_loader: torch.utils.data.DataLoader
    eval_loader: torch.utils.data.DataLoader
    scaler: "RegressionLabelScaler"


@dataclass
class RegressionTrainingHistory:
    """Simple container for logging loss curves and evaluation metrics."""

    train_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, float]] = field(default_factory=list)


class RegressionLabelScaler:
    """Encode and decode regression targets for semilearn algorithms.

    Semi-supervised algorithms in semilearn expect bounded targets when the
    regression-specific losses are used.  This helper scales arbitrary
    real-valued targets into ``[0, num_bins - 1]`` while preserving ordering so
    that pseudo-labels can be safely generated and evaluated.
    """

    def __init__(self, min_value: float, max_value: float, num_bins: int = 256):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.num_bins = int(max(1, num_bins))
        if math.isclose(self.max_value, self.min_value):
            self.scale = 0.0
        else:
            self.scale = (self.num_bins - 1) / (self.max_value - self.min_value)

    def encode(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.scale == 0.0:
            return np.zeros_like(values, dtype=np.float32)
        scaled = (values - self.min_value) * self.scale
        return np.clip(scaled, 0.0, self.num_bins - 1).astype(np.float32)

    def decode(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.scale == 0.0:
            return np.full_like(values, fill_value=self.min_value, dtype=np.float32)
        return values / self.scale + self.min_value


class RegressionLabeledDataset(Dataset):
    """Torch dataset returning semilearn-style labeled batches."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        if len(features) != len(targets):
            raise ValueError("Features and targets must contain the same number of samples")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        targets = torch.as_tensor(targets, dtype=torch.float32)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x_lb": self.features[idx], "y_lb": self.targets[idx]}


class RegressionUnlabeledDataset(Dataset):
    """Torch dataset returning semilearn-style unlabeled batches."""

    def __init__(self, features: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x_ulb_w": self.features[idx]}


class RegressionMLP(nn.Module):
    """A minimal MLP that matches semilearn's network interface."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        feature_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden
        if feature_dim != last_dim:
            layers.append(nn.Linear(last_dim, feature_dim))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.feature_dim = feature_dim if layers else input_dim
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        logits = self.head(feat)
        return {"logits": logits, "feat": feat}


def build_mlp_regressor(
    input_dim: int,
    hidden_dims: Sequence[int] = (256, 128),
    feature_dim: Optional[int] = None,
) -> NetBuilderSpec:
    """Create a ``NetBuilderSpec`` for a feed-forward MLP regressor."""

    hidden_dims = tuple(hidden_dims)
    feature_dim = feature_dim or (hidden_dims[-1] if hidden_dims else input_dim)

    def builder(num_classes: int, pretrained: bool = False, pretrained_path: Optional[str] = None) -> nn.Module:
        del pretrained, pretrained_path
        return RegressionMLP(input_dim=input_dim, hidden_dims=hidden_dims, feature_dim=feature_dim, num_classes=num_classes)

    return NetBuilderSpec(builder=builder, feature_dim=feature_dim)


def generate_polynomial_regression_data(
    num_samples: int = 4096,
    input_dim: int = 8,
    degree: int = 3,
    noise: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic polynomial regression dataset."""

    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=(num_samples, input_dim)).astype(np.float32)
    bias = rng.normal(scale=0.5)
    targets = np.full(num_samples, bias, dtype=np.float32)
    for power in range(1, degree + 1):
        weights = rng.normal(scale=1.0 / power, size=input_dim)
        targets += (features ** power @ weights.astype(np.float32))
    targets += rng.normal(scale=noise, size=num_samples).astype(np.float32)

    indices = rng.permutation(num_samples)
    val_size = int(num_samples * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return features[train_idx], targets[train_idx], features[val_idx], targets[val_idx]


def build_semilearn_regression_pipeline(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
    net_spec: NetBuilderSpec,
    algorithm: str = "pseudolabel",
    num_labels: int = 512,
    batch_size: int = 64,
    uratio: int = 1,
    epochs: int = 10,
    label_range: int = 256,
    config_overrides: Optional[Dict[str, object]] = None,
    seed: int = 0,
) -> RegressionPipelineArtifacts:
    """Create a reusable semilearn regression pipeline for arbitrary datasets."""

    if algorithm == "srpseudolabel" and not torch.cuda.is_available():
        raise RuntimeError("srpseudolabel requires CUDA because its rewarder uses CUDA-specific operations."
                           " Please switch to 'pseudolabel' or run on a GPU-enabled environment.")

    scaler = RegressionLabelScaler(np.min(train_targets), np.max(train_targets), num_bins=label_range)
    encoded_train = scaler.encode(train_targets)
    encoded_eval = scaler.encode(eval_targets)

    steps_per_epoch = max(1, math.ceil(num_labels / batch_size))
    eval_steps = max(1, math.ceil(len(eval_features) / max(1, batch_size)))

    base_config: Dict[str, object] = {
        "algorithm": algorithm,
        "net": "custom_mlp",
        "net_from_name": False,
        "use_pretrain": False,
        "num_classes": 1,
        "loss_type": "l2_loss",
        "task_type": "reg",
        "feature_dim": net_spec.feature_dim,
        "num_labels": num_labels,
        "batch_size": batch_size,
        "uratio": uratio,
        "eval_batch_size": batch_size,
        "epoch": epochs,
        "num_train_iter": epochs * steps_per_epoch,
        "num_eval_iter": eval_steps,
        "num_log_iter": max(1, steps_per_epoch // 2),
        "ema_m": 0.0,
        "ulb_loss_ratio": 1.0,
        "optim": "AdamW",
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "amp": False,
        "clip_grad": 0.0,
        "data_dir": "./data",
        "dataset": "poly_regression_demo",
        "save_dir": "./saved_models",
        "save_name": f"{algorithm}_poly_regression",
        "gpu": 0 if torch.cuda.is_available() else None,
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "multiprocessing_distributed": False,
        "num_workers": 0,
        "seed": seed,
    }
    if algorithm == "srpseudolabel":
        base_config.update({
            "range": label_range,
            "feature_dim": net_spec.feature_dim,
            "start_timing": 100,
            "sr_lr": 5e-4,
            "N_k": 5,
            "sr_ema": False,
            "sr_ema_m": 0.99,
        })
        total_iters = base_config["num_train_iter"]
        warmup_target = max(base_config["start_timing"], int(0.6 * total_iters))
        if warmup_target >= total_iters:
            warmup_target = max(total_iters - max(1, total_iters // 10), 1)
        base_config["start_timing"] = warmup_target

    if config_overrides:
        base_config.update(config_overrides)

    config = get_config(base_config)

    # ``get_config`` mirrors the command line interface used in the reference
    # training scripts where distributed attributes are populated after
    # argument parsing.  When building a configuration programmatically we need
    # to ensure those attributes exist so that algorithm constructors relying
    # on them (e.g. ``AlgorithmBase``) behave as expected.
    if not hasattr(config, "distributed"):
        config.distributed = False
    if not hasattr(config, "rank"):
        config.rank = 0
    if not hasattr(config, "world_size"):
        config.world_size = 1
    if not hasattr(config, "multiprocessing_distributed"):
        config.multiprocessing_distributed = False

    lb_features, lb_targets, ulb_features, _ = split_ssl_data(
        config,
        train_features,
        encoded_train,
        num_classes=config.num_classes,
        lb_num_labels=config.num_labels,
        include_lb_to_ulb=True,
    )

    train_lb_dataset = RegressionLabeledDataset(lb_features, lb_targets)
    train_ulb_dataset = RegressionUnlabeledDataset(ulb_features)
    eval_dataset = RegressionLabeledDataset(eval_features, encoded_eval)

    train_lb_loader = get_data_loader(
        config,
        train_lb_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        num_epochs=config.epoch,
        num_iters=config.num_train_iter,
    )
    train_ulb_loader = get_data_loader(
        config,
        train_ulb_dataset,
        batch_size=int(config.batch_size * config.uratio),
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        num_epochs=config.epoch,
        num_iters=config.num_train_iter,
    )
    eval_loader = get_data_loader(
        config,
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    algorithm_instance = get_algorithm(config, net_spec.builder, tb_log=None, logger=None)
    if getattr(algorithm_instance, "task_type", None) == "reg" and not hasattr(algorithm_instance, "range"):
        setattr(algorithm_instance, "range", label_range)

    return RegressionPipelineArtifacts(
        config=config,
        algorithm=algorithm_instance,
        train_lb_loader=train_lb_loader,
        train_ulb_loader=train_ulb_loader,
        eval_loader=eval_loader,
        scaler=scaler,
    )


class RegressionTrainer:
    """Utility trainer that leverages a semilearn algorithm for regression tasks."""

    def __init__(
        self,
        algorithm: object,
        config: object,
        scaler: RegressionLabelScaler,
        device: Optional[torch.device] = None,
    ) -> None:
        self.algorithm = algorithm
        self.config = config
        self.scaler = scaler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algorithm.model.to(self.device)
        if hasattr(self.algorithm, "ema_model") and self.algorithm.ema_model is not None:
            self.algorithm.ema_model.to(self.device)
        self.algorithm.optimizer.zero_grad(set_to_none=True)
        self.algorithm.it = 0
        self.algorithm.epoch = 0

    def fit(
        self,
        train_lb_loader: Iterable[Dict[str, torch.Tensor]],
        train_ulb_loader: Iterable[Dict[str, torch.Tensor]],
        eval_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    ) -> RegressionTrainingHistory:
        history = RegressionTrainingHistory()
        ulb_cycle = itertools.cycle(train_ulb_loader)

        for epoch in range(self.config.epoch):
            self.algorithm.model.train()
            epoch_losses: List[float] = []
            for batch_lb in train_lb_loader:
                batch_ulb = next(ulb_cycle)
                x_lb = batch_lb["x_lb"].to(self.device)
                y_lb = batch_lb["y_lb"].to(self.device)
                x_ulb = batch_ulb["x_ulb_w"].to(self.device)

                self.algorithm.optimizer.zero_grad(set_to_none=True)
                out_dict, _ = self.algorithm.train_step(x_lb=x_lb, y_lb=y_lb, x_ulb_w=x_ulb)
                loss = out_dict["loss"]
                loss.backward()
                if getattr(self.algorithm, "clip_grad", 0.0) and self.algorithm.clip_grad > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.algorithm.model.parameters(), self.algorithm.clip_grad)
                self.algorithm.optimizer.step()
                if getattr(self.algorithm, "scheduler", None) is not None:
                    self.algorithm.scheduler.step()

                epoch_losses.append(float(loss.item()))
                self.algorithm.it += 1

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history.train_losses.append(mean_loss)

            if eval_loader is not None:
                metrics = self.evaluate(eval_loader)
                history.eval_metrics.append(metrics)
                print(
                    f"Epoch {epoch + 1}/{self.config.epoch} - "
                    f"Train Loss: {mean_loss:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{self.config.epoch} - Train Loss: {mean_loss:.4f}")

            self.algorithm.epoch += 1

        return history

    def evaluate(self, data_loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.algorithm.model.eval()
        preds: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch["x_lb"].to(self.device)
                y = batch["y_lb"].to(self.device)
                outputs = self.algorithm.model(x)["logits"]
                preds.append(outputs.cpu().numpy())
                targets.append(y.cpu().numpy())
        self.algorithm.model.train()

        pred_encoded = np.concatenate(preds, axis=0).squeeze(-1)
        target_encoded = np.concatenate(targets, axis=0).squeeze(-1)
        pred_real = self.scaler.decode(pred_encoded)
        target_real = self.scaler.decode(target_encoded)

        diff = pred_real - target_real
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(diff)))
        denom = np.maximum(np.abs(target_real), 1e-6)
        mape = float(np.mean(np.abs(diff) / denom))
        target_centered = target_real - np.mean(target_real)
        denominator = float(np.sum(target_centered ** 2)) or 1e-6
        r2 = float(1.0 - np.sum(diff ** 2) / denominator)
        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def run_polynomial_regression_example() -> Dict[str, Dict[str, float]]:
    """Train a pseudo-label regressor on synthetic polynomial data."""

    train_x, train_y, val_x, val_y = generate_polynomial_regression_data()
    net_spec = build_mlp_regressor(input_dim=train_x.shape[1], hidden_dims=(128, 64), feature_dim=64)

    algorithm_name = "srpseudolabel" if torch.cuda.is_available() else "pseudolabel"
    pipeline = build_semilearn_regression_pipeline(
        train_features=train_x,
        train_targets=train_y,
        eval_features=val_x,
        eval_targets=val_y,
        net_spec=net_spec,
        algorithm=algorithm_name,
        num_labels=512,
        batch_size=64,
        uratio=2,
        epochs=5,
        label_range=256,
        seed=42,
    )

    trainer = RegressionTrainer(pipeline.algorithm, pipeline.config, pipeline.scaler)
    trainer.fit(pipeline.train_lb_loader, pipeline.train_ulb_loader, pipeline.eval_loader)
    metrics = trainer.evaluate(pipeline.eval_loader)
    print("Final evaluation:", metrics)
    return {"metrics": metrics}


if __name__ == "__main__":
    run_polynomial_regression_example()
