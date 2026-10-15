"""Small CPU checks; never starts the 300-epoch trainer."""
from pathlib import Path
import copy
import dill
import json
import sys
import tempfile
import unittest

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(ROOT / "scripts"))
from diffusion_policy_3d.common.sampler import get_val_mask  # noqa: E402
from diffusion_policy_3d.dataset.generic_dp3_dataset import GenericDP3Dataset  # noqa: E402
from evaluate_bimanual import evaluate_split, training_action_stats  # noqa: E402
from harvest_300 import CheckpointRegistry, restore_rng, rng_state, warning_status  # noqa: E402
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy_3d.model.common.harvest_lr_scheduler import (  # noqa: E402
    get_harvest_two_stage_scheduler,
)
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel  # noqa: E402


class ZeroPolicy:
    n_obs_steps = 2
    n_action_steps = 1

    def predict_action(self, obs):
        batch = len(obs["agent_pos"])
        return {"action": torch.zeros((batch, 1, 54), device=obs["agent_pos"].device)}


class InfrastructureTest(unittest.TestCase):
    def test_two_stage_lr_boundaries_and_resume(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = get_harvest_two_stage_scheduler(
            optimizer, steps_per_epoch=1962, total_epochs=300,
            first_stage_epochs=50, warmup_steps=500,
            middle_lr=1e-5, final_lr=1e-6)
        factor = scheduler.lr_lambdas[0]
        self.assertAlmostEqual(1e-4 * factor(500), 1e-4)
        self.assertAlmostEqual(1e-4 * factor(50 * 1962), 1e-5)
        self.assertAlmostEqual(1e-4 * factor(300 * 1962), 1e-6)
        for _ in range(5):
            optimizer.step(); scheduler.step()
        restored_param = torch.nn.Parameter(torch.zeros(()))
        restored_optimizer = torch.optim.AdamW([restored_param], lr=1e-4)
        restored = get_harvest_two_stage_scheduler(
            restored_optimizer, steps_per_epoch=1962, total_epochs=300,
            first_stage_epochs=50, warmup_steps=500,
            middle_lr=1e-5, final_lr=1e-6)
        restored_optimizer.load_state_dict(optimizer.state_dict())
        restored.load_state_dict(scheduler.state_dict())
        optimizer.step(); scheduler.step()
        restored_optimizer.step(); restored.step()
        self.assertEqual(restored.get_last_lr(), scheduler.get_last_lr())

    def test_warning_only_after_twenty_stale_epochs(self):
        rows = [{"epoch": e, "ema_val_loss": 0.5 - min(e, 10) * 0.01}
                for e in range(1, 31)]
        self.assertIsNone(warning_status(rows[:-1], 20, 1e-5))
        warning = warning_status(rows, 20, 1e-5)
        self.assertEqual(warning["warning_current_epoch"], 30)
        self.assertEqual(warning["warning_best_epoch"], 10)
        self.assertEqual(warning["warning_epochs_since_effective_improvement"], 20)

    def test_train_only_normalizer_and_grouped_action_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = zarr.open_group(str(Path(directory) / "tiny.zarr"), mode="w")
            data = root.create_group("data")
            meta = root.create_group("meta")
            meta.create_dataset("episode_ends", data=np.array([2, 4, 6, 8]))
            selected = get_val_mask(4, 0.5, 42)
            val_frames = np.repeat(selected, 2)
            values = np.where(val_frames, 1000.0, 1.0).astype("f4")
            data.create_dataset("action", data=np.repeat(values[:, None], 54, axis=1))
            data.create_dataset("state", data=np.repeat(values[:, None], 54, axis=1))
            data.create_dataset("point_cloud", data=np.broadcast_to(
                values[:, None, None], (8, 4, 3)).copy())
            root.attrs["source_episode_names"] = ["A/a", "A/b", "B/c", "B/d"]
            dataset = GenericDP3Dataset(
                zarr_path=str(Path(directory) / "tiny.zarr"),
                expected_num_points=4, expected_dim=54, horizon=2,
                val_ratio=0.5, grouped_validation=True, seed=42)
            stats = dataset.get_normalizer().get_input_stats()
            for field in ("action", "agent_pos", "point_cloud"):
                self.assertEqual(float(stats[field]["max"].max()), 1.0)
            subset, bags = dataset.get_fixed_train_eval_dataset(1, 42)
            self.assertEqual(subset.point_cloud_noise_std, 0)
            self.assertTrue(all(bag in ("A/a", "A/b", "B/c", "B/d") for bag in bags))
            mean, minimum, maximum = training_action_stats(dataset)
            report = evaluate_split(ZeroPolicy(), dataset.get_validation_dataset(),
                                    torch.device("cpu"), 2, 42, mean, minimum, maximum, 0)
            self.assertEqual(len(report["per_horizon_step"]), 1)
            self.assertEqual(set(report["by_source"]), {"A", "B"})
            self.assertEqual(len(report["by_bag"]), 2)

    def test_registry_keeps_topk_milestone_and_last_without_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            store = CheckpointRegistry(output, top_k=2, milestones={2})
            for epoch, loss in enumerate((0.4, 0.3, 0.2, 0.5), 1):
                store.record({"model": {"weight": torch.tensor([epoch])}},
                             epoch, epoch * 10, loss)
            paths = list((output / "checkpoints").glob("*.ckpt"))
            self.assertEqual({p.name for p in paths},
                             {"epoch=0002.ckpt", "epoch=0003.ckpt", "last.ckpt"})
            rows = json.loads((output / "checkpoint_registry.json").read_text())
            self.assertTrue(rows[1]["is_milestone"])
            self.assertTrue(rows[2]["is_best_so_far"])
            self.assertIsNone(rows[0]["checkpoint_path"])
            self.assertEqual(torch.load(output / "checkpoints" / "last.ckpt",
                                        weights_only=False)["model"]["weight"].item(), 4)

    def test_random_state_restores(self):
        state = rng_state()
        expected = torch.rand(3)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.rand(3)))

    def test_full_state_restores_lr_optimizer_ema_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            shadow = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=1,
                                      num_training_steps=20)
            updater = EMAModel(shadow)
            for _ in range(4):
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step(); optimizer.zero_grad(); scheduler.step()
                updater.step(model)
            payload = {
                "format_version": 2, "model": model.state_dict(),
                "ema_model": shadow.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "ema_updater": {"optimization_step": updater.optimization_step,
                                "decay": updater.decay},
                "next_epoch": 4, "global_step": 4, "random_state": rng_state(),
            }
            store = CheckpointRegistry(Path(directory), 2, {4})
            store.record(payload, 4, 4, 0.1)
            loaded = torch.load(Path(directory) / "checkpoints" / "last.ckpt",
                                weights_only=False, pickle_module=dill)
            restored = torch.nn.Linear(2, 1)
            restored_shadow = copy.deepcopy(restored)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored_scheduler = get_scheduler("cosine", restored_optimizer,
                                               num_warmup_steps=1, num_training_steps=20)
            restored_updater = EMAModel(restored_shadow)
            restored.load_state_dict(loaded["model"])
            restored_shadow.load_state_dict(loaded["ema_model"])
            restored_optimizer.load_state_dict(loaded["optimizer"])
            restored_scheduler.load_state_dict(loaded["lr_scheduler"])
            restored_updater.optimization_step = loaded["ema_updater"]["optimization_step"]
            restored_updater.decay = loaded["ema_updater"]["decay"]
            self.assertEqual(loaded["next_epoch"], 4)
            self.assertEqual(restored_updater.optimization_step, 4)
            self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
            restore_rng(loaded["random_state"])
            reference_random = torch.rand(2)
            restore_rng(loaded["random_state"])
            self.assertTrue(torch.equal(torch.rand(2), reference_random))
            for net, opt, sch, ema in ((model, optimizer, scheduler, updater),
                                       (restored, restored_optimizer, restored_scheduler,
                                        restored_updater)):
                net(torch.ones(1, 2)).sum().backward()
                opt.step(); opt.zero_grad(); sch.step(); ema.step(net)
            self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(model.parameters(), restored.parameters())))
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(shadow.parameters(), restored_shadow.parameters())))


if __name__ == "__main__":
    unittest.main()
