"""
Configuration loader for FNO-EVIO (YAML + optional OmegaConf).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from fno_evio.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig


def _load_yaml_as_dict(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(p))
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("YAML config requested but neither omegaconf nor PyYAML is available.") from e

    with open(str(p), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict")
    return data


def experiment_config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    dataset = DatasetConfig(**(d.get("dataset") or {}))
    model = ModelConfig(**(d.get("model") or {}))
    training = TrainingConfig(**(d.get("training") or {}))
    return ExperimentConfig(
        dataset=dataset,
        model=model,
        training=training,
        seed=int(d.get("seed", 0)),
        device=str(d.get("device", "cuda")),
        output_dir=d.get("output_dir"),
    )


def load_experiment_config(path: str) -> ExperimentConfig:
    d = _load_yaml_as_dict(path)
    return experiment_config_from_dict(d)


def merge_cli_overrides(cfg: ExperimentConfig, *, overrides: Dict[str, Any]) -> ExperimentConfig:
    """
    Apply CLI overrides to an ExperimentConfig.
    """
    data = asdict(cfg)
    for k, v in overrides.items():
        if v is None:
            continue
        if k in ("device", "seed", "output_dir"):
            data[k] = v
            continue
        if isinstance(data.get("dataset"), dict) and k in data["dataset"]:
            data["dataset"][k] = v
            continue
        if isinstance(data.get("model"), dict) and k in data["model"]:
            data["model"][k] = v
            continue
        if isinstance(data.get("training"), dict) and k in data["training"]:
            data["training"][k] = v
    return experiment_config_from_dict(data)
