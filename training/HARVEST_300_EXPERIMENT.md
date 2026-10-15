# Harvest tomato A/B: proposed 300-epoch experiment

This is a **new run from random initialization** using the existing, verified
`AB_committed_4096.zarr`. Running this file's command starts training; the
preparation work has not started that run.

## Review points

- Hydra configuration: `3D-Diffusion-Policy/diffusion_policy_3d/config/harvest_tomato_300.yaml`.
- Fixed seed 42 and grouped split by source bag. `fixed_split.json` records the
  validation and diagnostic train-evaluation bags at start.
- Action, state and XYZ point-cloud normalizers fit training frames only.
- The chosen plan uses 500 warmup steps, cosine decay to 1e-5 at epoch 50,
  then a second cosine decay to 1e-6 at epoch 300. See
  `lr_schedule_comparison.csv` and `.html` in the new output directory.
- `warning_patience: 20` and `warning_min_delta: 1e-5` emit warnings only.
  They never stop training or change checkpoint ranking/saves.
- `last.ckpt` is an atomic, full-state checkpoint written after each completed
  epoch. It saves raw/EMA weights, optimizer, scheduler, EMA updater counters,
  Python/NumPy/Torch/CUDA RNG, config, next epoch, global step, fixed split and
  experiment metrics. A mid-epoch interruption restarts that epoch.
- Each epoch: raw augmented train loss; EMA validation and fixed train subset
  losses; LR; sampled gradient norm; raw parameter norm; raw/EMA RMS difference;
  non-finite checks. Every fifth, milestone, and newly best epoch: validation
  action metrics and baselines. `action_eval/epoch=XXXX.json` contains per-joint,
  per-executed-step, A/B and per-bag results.
- Permanent checkpoints: global validation-loss Top 5, the ten configured
  milestones, and rolling `last.ckpt`. Milestone/Top 5 overlap uses one entity;
  permanent files are hard links to the completed `last.ckpt` of that epoch.
  `checkpoint_registry.json` records every epoch's selection and current path.
- Outputs include `metrics.csv`, `metrics.json`, separate HTML curves for
  losses, LR, gradient norm, EMA/raw distance, action errors, components,
  horizon, and A/B breakdown. The trajectory cosine score is omitted.

## Commands (run only after approval)

```bash
cd /home/user/dp3/DP3_ROSBAG_ZARR_TRAINING_BUNDLE
bash training/scripts/run_harvest_tomato_ab_300.sh start
```

If interrupted after a completed epoch, run the same script with `resume`.
`start` refuses an existing `last.ckpt`; `resume` requires it and checks the
config, split and dataset manifest before continuing. It does not use any
checkpoint from the previous 50-epoch experiment.
