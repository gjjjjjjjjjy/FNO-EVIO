from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from fno_evio.app import build_dataloaders
from fno_evio.common.constants import compute_adaptive_sequence_length
from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
from fno_evio.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from fno_evio.eval.evaluate import evaluate
from fno_evio.models.vio import HybridVIONet


def _resolve_device(device_str: str) -> torch.device:
    s = str(device_str).strip().lower()
    if s in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _strip_prefixes(state: Dict[str, Any]) -> Dict[str, Any]:
    prefixes = ("module.", "model.", "orig_mod.")
    out = state
    for pfx in prefixes:
        if out and all(str(k).startswith(pfx) for k in out.keys()):
            out = {str(k)[len(pfx) :]: v for k, v in out.items()}
            break
    return out


def _infer_model_config_from_state_dict(sd: Dict[str, torch.Tensor], *, dt: float) -> ModelConfig:
    k_channels = int(sd["stem.0.weight"].shape[1]) if "stem.0.weight" in sd else 5
    win_k = max(k_channels // 5, 1)

    if "stem.4.weight" in sd:
        stem_channels = int(sd["stem.4.weight"].shape[0])
    elif "stem.0.weight" in sd:
        stem_channels = int(sd["stem.0.weight"].shape[0])
    else:
        stem_channels = 64

    modes = 10
    if "fno_block.unit.spec1.weight" in sd:
        modes = int(sd["fno_block.unit.spec1.weight"].shape[2])
    elif "fno_block.unit1.weights1" in sd:
        modes = int(sd["fno_block.unit1.weights1"].shape[2])

    use_mr = any(k.startswith("fno_block.unit_low") for k in sd.keys())
    modes_low, modes_high = 16, 32
    if use_mr:
        for k, v in sd.items():
            if "unit_low.weights1" in k:
                modes_low = int(v.shape[2])
                break
        for k, v in sd.items():
            if "unit_high.weights1" in k:
                modes_high = int(v.shape[2])
                break

    lstm_hidden = 128
    if "imu_encoder.lstm.weight_ih_l0" in sd:
        lstm_hidden = int(sd["imu_encoder.lstm.weight_ih_l0"].shape[0]) // 4
    elif "imu_encoder.lstm.cells.0.weight_ih" in sd:
        lstm_hidden = int(sd["imu_encoder.lstm.cells.0.weight_ih"].shape[0]) // 4

    lstm_layers = 1
    if "imu_encoder.lstm.weight_ih_l2" in sd or "imu_encoder.lstm.cells.2.weight_ih" in sd:
        lstm_layers = 3
    elif "imu_encoder.lstm.weight_ih_l1" in sd or "imu_encoder.lstm.cells.1.weight_ih" in sd:
        lstm_layers = 2

    use_cross = any((k.startswith("v_proj") or k.startswith("cross.")) for k in sd.keys())
    fusion_dim = int(sd["v_proj.weight"].shape[0]) if use_cross and "v_proj.weight" in sd else None
    fusion_heads = 4

    imu_embed_dim = 64
    if "imu_encoder.fc.weight" in sd:
        imu_embed_dim = int(sd["imu_encoder.fc.weight"].shape[0])
    elif "scale_head.0.weight" in sd:
        in_features = int(sd["scale_head.0.weight"].shape[1])
        imu_embed_dim = max(int(in_features - 4), 1)

    use_cudnn_lstm = any(("imu_encoder.lstm.weight_ih_l0" in k) for k in sd.keys())
    norm_mode = "gn"
    if any(k.startswith("imu_norm.") for k in sd.keys()):
        if any("running_mean" in k for k in sd.keys()):
            norm_mode = "bn"
        else:
            norm_mode = "gn"

    seq_len = compute_adaptive_sequence_length(float(dt))

    cfg = ModelConfig(
        modes=int(modes),
        stem_channels=int(stem_channels),
        imu_embed_dim=int(imu_embed_dim),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        sequence_length=int(seq_len),
        norm_mode=str(norm_mode),
        use_mr_fno=bool(use_mr),
        modes_low=int(modes_low),
        modes_high=int(modes_high),
        window_stack_K=int(win_k),
        voxel_stack_mode="abs",
        use_cross_attn=bool(use_cross),
        fusion_dim=fusion_dim,
        fusion_heads=int(fusion_heads),
        use_cudnn_lstm=bool(use_cudnn_lstm),
    )
    if cfg.fusion_dim is None and cfg.use_cross_attn:
        cfg.fusion_dim = int(cfg.stem_channels)
    return cfg


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")
    sd = _strip_prefixes(sd)
    return {str(k): v for k, v in sd.items()}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser("FNO-EVIO evaluation (ported from fno-FAST/test_fno_vio.py)")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--calib_yaml", type=str, default="")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dt", type=float, default=0.00833)
    p.add_argument("--rpe_dt", type=float, default=None)
    p.add_argument("--sample_stride", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resolution", type=int, nargs=2, default=(180, 320))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--eval_all_roots", action="store_true")
    p.add_argument("--eval_sim3_mode", type=str, default="diagnose", choices=["diagnose", "use_learned", "fix_learned", "off"])
    args = p.parse_args(argv)

    device = _resolve_device(str(args.device))
    dt = float(args.dt)
    rpe_dt = float(args.rpe_dt) if args.rpe_dt is not None else float(dt)

    calib_obj: Optional[Dict[str, Any]] = None
    calib_yaml = str(args.calib_yaml or "").strip()
    if calib_yaml:
        calib_yaml_p = Path(calib_yaml).expanduser().resolve()
        if not calib_yaml_p.exists():
            raise FileNotFoundError(str(calib_yaml_p))
        calib_obj = load_calibration(str(calib_yaml_p))
        if calib_obj is None:
            raise RuntimeError(f"Failed to parse calib_yaml: {str(calib_yaml_p)} (needs PyYAML)")

    dataset_root = str(args.dataset_root or "").strip()
    roots = []
    if dataset_root:
        roots = [dataset_root]
    elif isinstance(calib_obj, dict):
        mr = calib_obj.get("multi_root")
        if isinstance(mr, list) and mr:
            roots = [str(r) for r in mr]
        else:
            inferred = infer_dataset_root_from_calib(calib_obj, calib_yaml)
            if inferred:
                roots = [str(inferred)]
    if not roots:
        raise ValueError("Missing dataset_root. Provide --dataset_root or --calib_yaml with inferable root/multi_root")
    if not bool(args.eval_all_roots):
        roots = [roots[0]]

    ckpt_path = str(Path(args.checkpoint).expanduser().resolve())
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(ckpt_path)
    sd = _load_checkpoint(ckpt_path, device=device)

    model_cfg = _infer_model_config_from_state_dict(sd, dt=dt)
    train_cfg = TrainingConfig(
        eval_interval=0,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        persistent_workers=(int(args.num_workers) > 0),
        pin_memory=False,
        sequence_len=200,
        sequence_stride=200,
    )

    model = HybridVIONet(model_cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    results = []
    for r in roots:
        stride = max(int(args.sample_stride), 1)
        print(
            f"[EVAL CFG] root={str(Path(r).expanduser().resolve())} | dt={float(dt):.6f} | "
            f"resolution=({int(args.resolution[0])},{int(args.resolution[1])}) | sample_stride={stride}"
        )
        ds_cfg = DatasetConfig(
            root=str(Path(r).expanduser().resolve()),
            dt=float(dt),
            resolution=(int(args.resolution[0]), int(args.resolution[1])),
            sample_stride=stride,
            windowing_mode="imu",
            window_dt=float(dt),
            train_split=0.0,
            calib=calib_obj,
        )
        exp = ExperimentConfig(dataset=ds_cfg, model=model_cfg, training=train_cfg, seed=0, device=str(device.type))
        _ = exp
        _, val_loader = build_dataloaders(ds_cfg, training=train_cfg, model=model_cfg, device=device)
        ate, rpe_t, rpe_r, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            rpe_dt=float(rpe_dt),
            dt=float(dt),
            eval_sim3_mode=str(args.eval_sim3_mode),
        )
        results.append(
            {
                "root": str(r),
                "ate": float(ate),
                "rpe_t": float(rpe_t),
                "rpe_r": float(rpe_r),
            }
        )

    out_dir = str(args.output_dir or "").strip()
    if out_dir:
        out_p = Path(out_dir).expanduser().resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        for item in results:
            name = Path(item["root"]).name
            sub = out_p / name
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "test_results.txt").write_text(
                "\n".join([f"ATE: {item['ate']:.6f}", f"RPE_t: {item['rpe_t']:.6f}", f"RPE_r: {item['rpe_r']:.6f}"]) + "\n",
                encoding="utf-8",
            )
            (sub / "results.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        (out_p / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (out_p / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["root", "ate", "rpe_t", "rpe_r"])
            w.writeheader()
            for item in results:
                w.writerow({k: item[k] for k in ["root", "ate", "rpe_t", "rpe_r"]})

        lines = []
        for item in results:
            lines.extend(
                [
                    "----------------------------------------",
                    f"root: {item['root']}",
                    f"ATE: {item['ate']:.6f}",
                    f"RPE_t: {item['rpe_t']:.6f}",
                    f"RPE_r: {item['rpe_r']:.6f}",
                    "",
                ]
            )
        (out_p / "summary_results.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    for item in results:
        print(f"[{item['root']}] ATE: {item['ate']:.6f} | RPE_t: {item['rpe_t']:.6f} | RPE_r: {item['rpe_r']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
