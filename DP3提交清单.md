# DP3 提交清单与取舍建议

这份清单用于把当前工作区交给项目负责人审核。它区分“主项目必须保留”和“研究过程资料”，不把所有工作区文件都视为交付代码。

## A. 建议纳入主项目

这些文件直接改变训练/转换行为，或构成可复核的部署最小链路：

- `conversion/src/convest/dp3_pipeline.py`
- `conversion/src/convest/recipes/dp3.py`
- `conversion/scripts/dp3`
- `conversion/scripts/build_episode_list.py`
- `dp3.sh`
- `training/scripts/bimanual_zarr.py`
- `training/scripts/merge_bimanual_zarr.py`
- `training/3D-Diffusion-Policy/diffusion_policy_3d/dataset/generic_zarr_dataset.py`
- `training/3D-Diffusion-Policy/diffusion_policy_3d/config/harvest_tomato_300.yaml`
- `training/3D-Diffusion-Policy/diffusion_policy_3d/model/common/harvest_lr_scheduler.py`
- `training/3D-Diffusion-Policy/harvest_300.py`
- `training/scripts/train_bimanual.sh`
- `training/scripts/evaluate_bimanual.py`
- `training/scripts/deployment_offline_replay.py`
- `training/scripts/deployment_dry_run.py`
- `dp3_deploy/README.md`
- `dp3_deploy/config/deploy.json`
- `dp3_deploy/config/robot_source_contract.json`
- `dp3_deploy/config/ros_to_dp3_joint_mapping.csv`
- `dp3_deploy/action_mapping.py`
- `dp3_deploy/inference.py`
- `dp3_deploy/preprocess.py`
- `dp3_deploy/ros_prediction_only.py`
- `dp3_deploy/deploy.py`
- `dp3_deploy/build_bundle.py`
- `dp3_deploy/diffusion_policy_3d/`（部署运行所需的最小模型实现）
- `configs/harvest_tomato_AB_40_40.json`
- `training/deployment/robot_source_contract.json`
- `training/deployment/ros_to_dp3_joint_mapping.csv`

## B. 建议保留为审核/研究记录

这些文件有证据价值，但不应和主运行代码混在一个“部署提交”里：

- `DP3仓库修改汇总.md`
- `DP3_PREDICTION_BIAS_OFFLINE_REVIEW.md`
- `DP3_REAL_ROBOT_DEPLOYMENT_REVIEW.md`
- `DP3_STAGE4_INITIAL_DELTA_ANALYSIS.md`
- `training/DEPLOYMENT_READINESS.md`
- `training/HARVEST_300_EXPERIMENT.md`
- `training/deployment/REAL_ROBOT_OBSERVATION_PLAN.md`
- `training/deployment/epoch20_joint_delta_diagnostics.json`
- `training/deployment/epoch20_joint_delta_diagnostics.csv`
- `training/scripts/diagnose_dp3_prediction_bias.py`
- `training/scripts/extract_rollout_rosbag_events.py`
- `training/scripts/real_rollout_report.py`
- `training/tests/test_harvest_300_infrastructure.py`
- `dp3_deploy/DP3_EPOCH20_FRAMEWORK_ADAPTER.md`

## C. 建议移入 archive/experiments，不进入主交付

这些文件不是一定无效，而是实验性、重复性或兼容性较强，负责人不应把它们当成当前正式入口：

- `configs/harvest_tomato_AB_trial_40_40.json`（探索性 trial 配置）
- `training/scripts/run_harvest_tomato_ab_trial.sh`（兼容旧说明的入口）
- `training/scripts/run_harvest_tomato_ab_full.sh`（旧 full 入口，需确认是否仍使用）
- `training/scripts/compare_harvest_lr_options.py`
- `training/scripts/preview_harvest_lr.py`
- `training/scripts/build_depth_crop_guide.py`
- `training/scripts/benchmark_bimanual_parallel.py`
- `dp3_deploy/PI05_REFERENCE_REVIEW.md`（与 DP3 主交付无关的参考记录）

## D. 可直接排除或不应提交

这些是运行缓存、编译产物或临时结果，不属于源代码管理：

- 所有 `__pycache__/` 和 `*.pyc`
- `conversion/.runtime/`、`conversion/.cache/`、`conversion/.pytest_cache/`
- `outputs/holiday_pre722/` 下的 `.partial`、`lock`、`cron.log`
- `outputs/*_UNAPPROVED*` 和临时 crop selector/preview
- 旧的 smoke Zarr、progress/report 临时文件，除非负责人明确要求保留复现实验证据
- `dp3_deploy/checkpoints/epoch20_ema.pt` 是否提交需单独决定；通常放模型制品仓库或发布包，不放源码提交

## 建议的负责人交付形态

把交付拆成三层：

1. **源码层**：A 类代码和配置。
2. **审核层**：B 类报告、指标和现场验证计划。
3. **实验归档层**：C 类脚本和临时配置；D 类完全排除。

当前不要把整个工作区直接 `git add .`。先按 A/B/C/D 分类，再由负责人决定是否把部署 checkpoint 作为独立模型制品发布。
