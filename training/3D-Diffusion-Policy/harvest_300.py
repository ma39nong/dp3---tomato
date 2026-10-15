"""Isolated, resumable harvest-tomato experiment. Never used by legacy train.py."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.common.harvest_lr_scheduler import get_harvest_two_stage_scheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_bimanual import evaluate_split, training_action_stats  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_json(path: Path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_metrics(output: Path, rows: list[dict]):
    atomic_json(output / "metrics.json", rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    tmp = output / "metrics.csv.tmp"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    os.replace(tmp, output / "metrics.csv")


def write_curves(output: Path, rows: list[dict]):
    import plotly.graph_objects as go

    def chart(filename, title, fields):
        fig = go.Figure()
        for key, label in fields:
            x = [r["epoch"] for r in rows if r.get(key) is not None]
            y = [r[key] for r in rows if r.get(key) is not None]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label))
        fig.update_layout(title=title, xaxis_title="epoch", hovermode="x unified")
        tmp = output / (filename + ".tmp")
        fig.write_html(str(tmp), include_plotlyjs=True, full_html=True)
        os.replace(tmp, output / filename)

    chart("loss_curve.html", "Diffusion losses", [
        ("train_loss", "raw augmented train"), ("ema_train_eval_loss", "EMA fixed train"),
        ("ema_val_loss", "EMA validation")])
    chart("lr_curve.html", "Learning rate", [("lr", "learning rate")])
    chart("gradient_norm_curve.html", "Gradient norm (sampled)", [
        ("grad_norm_mean", "mean"), ("grad_norm_max", "max")])
    chart("ema_gap_curve.html", "EMA/raw parameter difference", [
        ("ema_raw_rms", "parameter RMS difference")])
    chart("action_error_curve.html", "Validation action error", [
        ("action_mae", "MAE"), ("action_rmse", "RMSE")])
    chart("component_error_curve.html", "Validation component MAE", [
        (f"{name}_mae", name) for name in ("left_arm", "left_hand", "right_arm", "right_hand")])
    chart("ab_breakdown_curve.html", "Validation source RMSE", [
        (f"{source}_rmse", source) for source in ("A", "B")])
    fig = go.Figure()
    for row in rows:
        if row.get("horizon_mae") is not None:
            fig.add_trace(go.Scatter(x=list(range(1, len(row["horizon_mae"]) + 1)),
                                     y=row["horizon_mae"], name=f"epoch {row['epoch']}"))
    fig.update_layout(title="Action horizon MAE", xaxis_title="executed action step")
    tmp = output / "horizon_error.html.tmp"
    fig.write_html(str(tmp), include_plotlyjs=True)
    os.replace(tmp, output / "horizon_error.html")


class CheckpointRegistry:
    def __init__(self, output: Path, top_k: int, milestones: set[int], rows=None):
        self.output = output
        self.directory = output / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.milestones = milestones
        self.rows = list(rows or [])

    def record(self, payload: dict, epoch: int, step: int, val_loss: float):
        previous_best = min((r["val_loss"] for r in self.rows), default=float("inf"))
        row = {"epoch": epoch, "global_step": step, "checkpoint_path": None,
               "val_loss": val_loss, "is_topk": False,
               "is_milestone": epoch in self.milestones,
               "is_best_so_far": val_loss < previous_best}
        self.rows.append(row)
        top_epochs = {r["epoch"] for r in sorted(
            self.rows, key=lambda r: (r["val_loss"], r["epoch"]))[:self.top_k]}
        for item in self.rows:
            item["is_topk"] = item["epoch"] in top_epochs
            if item["is_topk"] or item["is_milestone"]:
                item["checkpoint_path"] = str(self.directory / f"epoch={item['epoch']:04d}.ckpt")
            elif item["epoch"] == epoch:
                item["checkpoint_path"] = str(self.directory / "last.ckpt")
            else:
                item["checkpoint_path"] = None
        payload["registry"] = copy.deepcopy(self.rows)
        temporary = self.directory / "last.ckpt.tmp"
        torch.save(payload, temporary, pickle_module=dill)
        os.replace(temporary, self.directory / "last.ckpt")
        self.reconcile()
        return row

    def reconcile(self):
        """Finish registry/link updates if interrupted after last.ckpt commit."""
        latest = self.rows[-1]
        if latest["is_topk"] or latest["is_milestone"]:
            path = self.directory / f"epoch={latest['epoch']:04d}.ckpt"
            if not path.exists():
                os.link(self.directory / "last.ckpt", path)
        keep = {r["checkpoint_path"] for r in self.rows if r["checkpoint_path"]}
        for path in self.directory.glob("epoch=*.ckpt"):
            if str(path) not in keep:
                path.unlink()
        atomic_json(self.output / "checkpoint_registry.json", self.rows)


@torch.no_grad()
def parameter_diagnostics(raw, ema):
    raw_sumsq = diff_sumsq = count = 0.0
    finite = True
    for a, b in zip(raw.parameters(), ema.parameters()):
        finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        raw_sumsq += float(torch.sum(a.float().square()))
        diff_sumsq += float(torch.sum((a.float() - b.float()).square()))
        count += a.numel()
    return math.sqrt(raw_sumsq), math.sqrt(diff_sumsq / count), finite


def warning_status(rows: list[dict], patience: int, min_delta: float):
    """Report stale validation improvement without controlling the run."""
    if patience <= 0 or not rows:
        return None
    effective_best, last_effective = float("inf"), 0
    for row in rows:
        if row["ema_val_loss"] < effective_best - min_delta:
            effective_best = row["ema_val_loss"]
            last_effective = row["epoch"]
    current = rows[-1]
    stale = current["epoch"] - last_effective
    if stale < patience:
        return None
    absolute_best = min(rows, key=lambda row: row["ema_val_loss"])
    return {
        "warning": "EARLY_STOPPING_WARNING",
        "warning_current_epoch": current["epoch"],
        "warning_current_val_loss": current["ema_val_loss"],
        "warning_best_val_loss": absolute_best["ema_val_loss"],
        "warning_best_epoch": absolute_best["epoch"],
        "warning_epochs_since_effective_improvement": stale,
    }


@torch.no_grad()
def diffusion_eval(policy, loader, device, seed, max_batches=None):
    state = rng_state()
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        policy.eval()
        total = batches = 0
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            loss, _ = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            total += float(loss)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
        if not batches:
            raise ValueError("Empty evaluation split")
        return total / batches
    finally:
        restore_rng(state)


class Harvest300Workspace(TrainDP3Workspace):
    def run(self):
        cfg = self.cfg
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        last_path = output / "checkpoints" / "last.ckpt"
        if last_path.exists() and not cfg.training.resume:
            raise FileExistsError(f"Existing experiment output: {last_path}")
        if cfg.training.resume and not last_path.exists():
            raise FileNotFoundError(f"No resumable last checkpoint: {last_path}")
        if cfg.training.gradient_accumulate_every != 1:
            raise ValueError("This experiment requires gradient_accumulate_every=1")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        torch.cuda.manual_seed_all(cfg.training.seed)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        dataset_path = Path(cfg.task.dataset.zarr_path)
        manifest_path = dataset_path.with_name(dataset_path.name + ".manifest.json")
        dataset_fingerprint = hashlib.sha256(
            manifest_path.read_bytes() if manifest_path.exists()
            else (dataset_path / ".zattrs").read_bytes()).hexdigest()
        val_dataset = dataset.get_validation_dataset()
        fixed_dataset, fixed_bags = dataset.get_fixed_train_eval_dataset(
            bags_per_source=cfg.experiment.fixed_train_bags_per_source,
            seed=cfg.training.seed)
        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)
        fixed_loader = DataLoader(fixed_dataset, **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        if cfg.training.lr_scheduler == "harvest_two_stage_cosine":
            scheduler = get_harvest_two_stage_scheduler(
                self.optimizer, steps_per_epoch=len(train_loader),
                total_epochs=cfg.training.num_epochs,
                first_stage_epochs=cfg.experiment.lr_first_stage_end_epoch,
                warmup_steps=cfg.training.lr_warmup_steps,
                middle_lr=cfg.experiment.lr_middle,
                final_lr=cfg.experiment.lr_final)
        else:
            scheduler = get_scheduler(
                cfg.training.lr_scheduler, self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=len(train_loader) * cfg.training.num_epochs,
            )
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        registry = CheckpointRegistry(output, cfg.checkpoint.topk.k,
                                      set(cfg.experiment.milestones))
        history = []
        if cfg.training.resume:
            payload = torch.load(last_path, map_location="cpu", mmap=True,
                                 weights_only=False, pickle_module=dill)
            if payload.get("format_version") != 2:
                raise ValueError("Legacy checkpoint is not strictly resumable")
            saved_cfg = OmegaConf.to_container(payload["cfg"], resolve=True)
            current_cfg = OmegaConf.to_container(cfg, resolve=True)
            saved_cfg["training"]["resume"] = current_cfg["training"]["resume"]
            saved_cfg["training"]["stop_after_epoch"] = current_cfg["training"]["stop_after_epoch"]
            if saved_cfg != current_cfg:
                raise ValueError("Resume config differs from saved experiment config")
            self.model.load_state_dict(payload["model"])
            self.ema_model.load_state_dict(payload["ema_model"])
            # LinearNormalizer registers its ParameterDict lazily while loading;
            # those newly created tensors are CPU tensors even if the model was
            # already moved to CUDA before load_state_dict.
            self.model.to(device)
            self.ema_model.to(device)
            self.optimizer.load_state_dict(payload["optimizer"])
            optimizer_to(self.optimizer, device)
            scheduler.load_state_dict(payload["lr_scheduler"])
            ema.optimization_step = payload["ema_updater"]["optimization_step"]
            ema.decay = payload["ema_updater"]["decay"]
            self.epoch = payload["next_epoch"]
            self.global_step = payload["global_step"]
            registry.rows = payload["registry"]
            history = payload["metrics"]
            if payload["fixed_train_bags"] != fixed_bags:
                raise ValueError("Fixed train subset changed during resume")
            if payload["dataset_fingerprint"] != dataset_fingerprint:
                raise ValueError("Dataset manifest changed during resume")
            if payload["validation_bags"] != sorted({
                    name for name, keep in zip(dataset.source_episode_names, dataset.train_mask)
                    if not keep}):
                raise ValueError("Validation bag split changed during resume")
            restore_rng(payload["random_state"])
            del payload
            registry.reconcile()
            write_metrics(output, history)
            write_curves(output, history)
        else:
            if (output / "metrics.json").exists():
                raise FileExistsError(f"Experiment metrics already exist in {output}")
            atomic_json(output / "fixed_split.json", {
                "seed": cfg.training.seed, "validation_bags": sorted({
                    name for name, keep in zip(dataset.source_episode_names, dataset.train_mask)
                    if not keep}), "fixed_train_eval_bags": fixed_bags})

        train_mean, train_min, train_max = training_action_stats(dataset)
        end_epoch = cfg.training.num_epochs
        if cfg.training.stop_after_epoch is not None:
            end_epoch = min(end_epoch, cfg.training.stop_after_epoch)
        for index in range(self.epoch, end_epoch):
            def nonfinite_failure(message):
                atomic_json(output / "health_failure.json", {
                    "epoch": index + 1, "global_step": self.global_step,
                    "error": message, "nan_inf_detected": True})
                raise FloatingPointError(message)

            self.model.train()
            losses, gradient_norms = [], []
            for batch_index, batch in enumerate(train_loader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss, _ = self.model.compute_loss(batch)
                if not torch.isfinite(loss):
                    nonfinite_failure("NaN/Inf train loss")
                loss.backward()
                if batch_index % cfg.experiment.grad_norm_every_steps == 0 or batch_index == len(train_loader) - 1:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
                    if not torch.isfinite(norm):
                        nonfinite_failure("NaN/Inf gradient")
                    gradient_norms.append(float(norm))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.step(self.model)
                losses.append(float(loss.detach()))
                self.global_step += 1
                if cfg.experiment.smoke_mode and cfg.training.max_train_steps is not None \
                        and batch_index + 1 >= cfg.training.max_train_steps:
                    break
            self.epoch = index + 1
            parameter_norm, ema_raw_rms, finite = parameter_diagnostics(self.model, self.ema_model)
            if not finite:
                nonfinite_failure("NaN/Inf model parameter")
            try:
                max_eval = cfg.training.max_val_steps if cfg.experiment.smoke_mode else None
                val_loss = diffusion_eval(self.ema_model, val_loader, device,
                                          cfg.training.seed, max_eval)
                fixed_loss = diffusion_eval(self.ema_model, fixed_loader, device,
                                            cfg.training.seed, max_eval)
            except FloatingPointError as error:
                nonfinite_failure(str(error))
            best_before = min((r["ema_val_loss"] for r in history), default=float("inf"))
            is_best = val_loss < best_before
            row = {
                "epoch": self.epoch, "global_step": self.global_step,
                "train_loss": float(np.mean(losses)), "ema_val_loss": val_loss,
                "val_loss": val_loss, "ema_train_eval_loss": fixed_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm_mean": float(np.mean(gradient_norms)),
                "grad_norm_max": max(gradient_norms),
                "parameter_norm": parameter_norm, "ema_raw_rms": ema_raw_rms,
                "nan_inf_detected": False, "is_best_so_far": is_best,
            }
            if not cfg.experiment.smoke_mode and (
                    self.epoch % cfg.experiment.action_eval_every == 0
                    or is_best or self.epoch in cfg.experiment.milestones):
                state = rng_state()
                try:
                    action = evaluate_split(
                        self.ema_model, val_dataset, device,
                        cfg.val_dataloader.batch_size, cfg.training.seed + 100000,
                        train_mean, train_min, train_max, ablation_batches=0)
                finally:
                    restore_rng(state)
                action_dir = output / "action_eval"
                action_dir.mkdir(exist_ok=True)
                action_path = action_dir / f"epoch={self.epoch:04d}.json"
                atomic_json(action_path, action)
                row.update(action_report=str(action_path), action_mae=action["model"]["mae"],
                           action_rmse=action["model"]["rmse"],
                           horizon_mae=[item["mae"] for item in action["per_horizon_step"]])
                for name, values in action["model"]["groups"].items():
                    row[f"{name}_mae"] = values["mae"]
                    row[f"{name}_rmse"] = values["rmse"]
                for source, values in action["by_source"].items():
                    row[f"{source}_mae"] = values["mae"]
                    row[f"{source}_rmse"] = values["rmse"]
                row["mean_baseline_rmse"] = action["train_mean_baseline"]["rmse"]
                row["hold_baseline_rmse"] = action["hold_current_state_baseline"]["rmse"]
            warning = warning_status(history + [row],
                                     int(cfg.experiment.warning_patience),
                                     float(cfg.experiment.warning_min_delta))
            if warning:
                row.update(warning)
                print("EARLY_STOPPING_WARNING "
                      f"epoch={self.epoch} val_loss={val_loss:.8f} "
                      f"best_val_loss={warning['warning_best_val_loss']:.8f} "
                      f"best_epoch={warning['warning_best_epoch']} "
                      f"epochs_since_effective_improvement="
                      f"{warning['warning_epochs_since_effective_improvement']}",
                      flush=True)
            history.append(row)
            payload = {
                "format_version": 2, "cfg": cfg,
                "model": self.model.state_dict(), "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                "ema_updater": {"optimization_step": ema.optimization_step, "decay": ema.decay},
                "next_epoch": self.epoch, "global_step": self.global_step,
                "random_state": rng_state(), "fixed_train_bags": fixed_bags,
                "dataset_fingerprint": dataset_fingerprint,
                "validation_bags": sorted({
                    name for name, keep in zip(dataset.source_episode_names, dataset.train_mask)
                    if not keep}),
                "metrics": history,
            }
            registry.record(payload, self.epoch, self.global_step, val_loss)
            write_metrics(output, history)
            write_curves(output, history)
            print(f"epoch={self.epoch} train={row['train_loss']:.6f} "
                  f"ema_train={fixed_loss:.6f} val={val_loss:.6f} lr={row['lr']:.8g}", flush=True)


@hydra.main(version_base=None, config_path="diffusion_policy_3d/config",
            config_name="harvest_tomato_300")
def main(cfg):
    if cfg.training.num_epochs != 300:
        raise ValueError("harvest_300 requires training.num_epochs=300")
    if not cfg.training.run_validation or not cfg.training.use_ema:
        raise ValueError("harvest_300 requires validation and EMA")
    Harvest300Workspace(cfg).run()


if __name__ == "__main__":
    main()
