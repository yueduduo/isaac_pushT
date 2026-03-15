# Isaac-Pusht-v0 

## Overview

This repository contains the code for the Isaac-Pusht-v0 task.
<div style="display: flex; justify-content: center;">
  <img src="source/docs/display.png" alt="Isaac-Pusht-v0" width="400">
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

## Policy Training (Diffusion / Flow Mapping)

This project includes two Transformer-based policy learning algorithms:

- `diffusion`: conditional diffusion policy
- `flow_mapping`: flow matching policy

Both algorithms use:

- multimodal observations: `front image + back image + state`
- a pretrained `ResNet18` image encoder
- horizon-based action sequence modeling

### 1. Train Diffusion Policy

```bash
python scripts/train.py --algo diffusion --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --epochs 500 --batch-size 32 --device cuda
```

### 2. Train Flow Mapping Policy

```bash
python scripts/train.py --algo flow_mapping --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --epochs 500 --batch-size 32 --device cuda
```

### 3. Freeze ResNet for Warmup (Optional)

Freeze the ResNet18 backbone for early epochs, then unfreeze for fine-tuning:

```bash
python scripts/train.py --algo diffusion --horizon 16 --freeze-resnet --freeze-epochs 5
```

## Policy Evaluation

### 1. Evaluate Diffusion Checkpoint

```bash
python scripts/eval.py --algo diffusion --ckpt checkpoints/best_diffusion.pt --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --device cuda
```

### 2. Evaluate Flow Mapping Checkpoint

```bash
python scripts/eval.py --algo flow_mapping --ckpt checkpoints/best_flow_mapping.pt --repo-id isaac_pusht --root data/isaac_pusht --horizon 16 --device cuda
```

## TensorBoard Visualization

### 1. Training Curves

Enable TensorBoard logging during training:

```bash
python scripts/train.py --algo diffusion --horizon 16 --tensorboard --tb-logdir runs --tb-run-name diffusion_h16_exp1
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

