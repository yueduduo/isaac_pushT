# Isaac-Pusht-v0 

## Overview

This repository contains the code for the Isaac-Pusht-v0 task.
<div style="display: flex; justify-content: center;">
  <img src="source/isaac_pusht/docs/display.png" alt="Isaac-Pusht-v0" width="400">
</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/isaac_pusht
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        python scripts/list_envs.py
        ```

    - Running with keyboard teleoperation:

        ```bash
        python scripts/teleop_se3_agent.py --task Isaac-Pusht-v0 
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            python scripts/zero_agent.py --task=Isaac-Pusht-v0 
            ```
        - Random-action agent

            ```bash
            python scripts/random_agent.py --task=Isaac-Pusht-v0 
            ```

## Data Collection and Visualization

This project supports data collection and visualization using the LeRobot v3.0 format.

### 1. Data Collection

Run the following command to start collecting data (including camera observations):

```bash
python scripts/record_lerobot.py --task Isaac-Pusht-v0 --num_episodes 200 --enable_cameras
```

### 2. Data Visualization

After collection is complete, you can use the `lerobot` visualization tool to view the data:

```bash
python -m lerobot.scripts.lerobot_dataset_viz --repo-id isaac_pusht --root data/isaac_pusht --episode-index 0
```

## Policy Training (Diffusion / Flow Matching)

This project includes two Transformer-based policy learning algorithms:

- **`diffusion`**: conditional diffusion policy (DDPM-style action noise prediction).
- **`flow_matching`**: flow-matching (velocity-field) policy. Python package: `algo.flow_matching` (`FlowMatchingPolicy`, `FlowMatchingTrainer`, …). Checkpoints: `best_flow_matching.pt` / `last_flow_matching.pt`.

Both algorithms use:

- multimodal observations: `front image + back image + state`
- a pretrained `ResNet18` image encoder
- horizon-based action sequence modeling

### 1. Train Diffusion Policy

Training budget is **`--train-steps`** (number of `optimizer.step` calls), not epochs. Comparable “data passes” = `train_steps / steps_per_epoch`, where `steps_per_epoch = ceil(train_windows / batch_size)` (printed at startup).

```bash
python scripts/train.py --algo diffusion --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --train-steps 5000 --batch-size 32 --device cuda
```

### 2. Train Flow Matching Policy

```bash
python scripts/train.py --algo flow_matching --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --train-steps 5000 --batch-size 32 --device cuda
```

### 3. JSON training & model config (`scripts/configs`)

Training hyperparameters and Transformer architecture (including the action head) can be loaded from **JSON** (`stdlib` `json`). The `train` section uses `train_steps`, `freeze_steps`, etc. (see template).

- Example template: [`scripts/configs/train_default.json`](scripts/configs/train_default.json)
- Loader and dataclasses: `utils/train_config.py`
- Sections: `train`, `diffusion_model`, `flow_matching_model`, `diffusion_trainer`, `flow_matching_trainer`

```bash
python scripts/train.py --config scripts/configs/train_default.json --algo diffusion
```

Command-line arguments override the `train` section of the JSON.

### 4. Freeze ResNet for Warmup (Optional)

Freeze the ResNet18 backbone for the first **`freeze_steps`** optimizer steps, then unfreeze:

```bash
python scripts/train.py --algo diffusion --horizon 16 --freeze-resnet --freeze-steps 500
```

### 5. Resume training (`--resume`)

Use the **same** `--config` / CLI as a normal run (architecture and trainer hyperparameters must match the checkpoint). Checkpoints store **`step`** (completed global optimizer steps) and **`best_metric`**.

```bash
python scripts/train.py --config scripts/configs/train_default.json --algo diffusion --resume checkpoints/last_diffusion.pt --train-steps 2000
```

`--train-steps` is always **how many optimizer steps to run in this invocation**, starting from `saved_step + 1`. Legacy checkpoints with only `epoch` are converted with `completed_steps ≈ epoch * steps_per_epoch` using the **current** dataloader (approximate if batch or data changed).

When a sidecar `*.norm.json` exists next to the resume checkpoint, normalization stats are loaded from it; otherwise they are recomputed from the current dataset.

## Policy Evaluation

`scripts/eval.py` expects the repository root on `PYTHONPATH` (same as `scripts/train.py`).

### 1. Evaluate Diffusion Checkpoint

```bash
python scripts/eval.py --algo diffusion --ckpt checkpoints/best_diffusion.pt --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --device cuda
```

### 2. Evaluate Flow Matching Checkpoint

```bash
python scripts/eval.py --algo flow_matching --ckpt checkpoints/best_flow_matching.pt --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --device cuda
```

**Migrating old runs:** checkpoints named `best_flow_mapping.pt` are from the previous naming; rename to `best_flow_matching.pt` or pass the old path as `--ckpt` (filename is arbitrary). The **state dict** must match a model built with the current `algo.flow_matching` architecture.

## TensorBoard Visualization

### 1. Training Curves

TensorBoard is enabled by default for `scripts/train.py`. To turn it off or set a run name:

```bash
python scripts/train.py --algo diffusion --horizon 16 --tb-logdir runs --tb-run-name diffusion_h16_exp1
python scripts/train.py --algo diffusion --no-tensorboard
```

### 2. Evaluation Metrics

Log evaluation metric (`eval/action_mse`) to TensorBoard:

```bash
python scripts/eval.py --algo diffusion --ckpt checkpoints/best_diffusion.pt --horizon 16 --tensorboard --tb-logdir runs --tb-run-name diffusion_h16_exp1 --tb-step 50
```

### 3. Start TensorBoard

```bash
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006).

