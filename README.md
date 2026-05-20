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

## Policy Training (LeRobot Diffusion)

Training uses **`lerobot.policies.diffusion.DiffusionPolicy`** (1D conditional U-Net, SpatialSoftmax vision, `diffusers` DDPM schedule). There is **no** local `algo/` package; old Transformer+flatten checkpoints are **not** compatible.

Training budget is **`--train-steps`** (number of `optimizer.step` calls). `horizon` must be divisible by `2 ** len(down_dims)` (default down_dims length 3 → multiple of 8).

```bash
python scripts/train.py --repo-id isaac_pusht --root data/isaac_pusht --horizon 32 --train-steps 5000 --batch-size 32 --device cuda
```

### JSON config (`scripts/configs`)

- Example: [`scripts/configs/train_default.json`](scripts/configs/train_default.json)
- Loader: `utils/train_config.py`
- Sections: **`train`**, **`diffusion_trainer`** (maps into `DiffusionConfig` where keys match), optional **`lerobot_diffusion`** for further overrides.

```bash
python scripts/train.py --config scripts/configs/train_default.json
```

### Freeze RGB backbone (optional)

```bash
python scripts/train.py --horizon 32 --freeze-resnet --freeze-steps 500
```

### Resume (`--resume`)

Checkpoints are **`lerobot_diffusion`** format (`policy_type` + pickled `DiffusionConfig` + weights). Old `best_diffusion.pt` from the removed custom algo cannot be loaded.

```bash
python scripts/train.py --config scripts/configs/train_default.json --resume checkpoints/last_diffusion.pt --train-steps 2000
```

Sidecar `*.norm.json` behavior is unchanged.

## Policy Evaluation

`scripts/eval.py` expects the repository root on `PYTHONPATH` (same as `scripts/train.py`).

```bash
python scripts/eval.py --ckpt checkpoints/best_diffusion.pt --repo-id isaac_pusht --root data/isaac_pusht --horizon 32 --device cuda
```

`--horizon` must match the checkpoint training horizon.

### Isaac Lab simulation (`eval_model.py`, `record_model_replay.py`)

These scripts load the trained diffusion policy and step the **Isaac** task (not the offline dataloader in `eval.py`).

- **`scripts/eval_model.py`** — roll out `Isaac-Pusht-v0` with the policy for multiple episodes (uses `AppLauncher`; run with the same Isaac Lab Python you use for training data collection).
- **`scripts/record_model_replay.py`** — replay a single dataset episode: feed dataset images + `observation.state` into the policy and step the sim (see the script docstring for consistency caveats).

By default both load **`--checkpoint`** and `*.norm.json` on the **same machine** as the simulator.

### Remote policy over WebSocket (save VRAM)

If Isaac Sim and the diffusion model do not fit in one GPU, run **inference in a separate process** (or on another host with a GPU) and keep only the simulator on the Isaac machine.

**Implementation** lives under [`utils/remote_policy/`](utils/remote_policy/): msgpack + NumPy wire format (same idea as `pi0_fast_deploy` / `websocket_policy_server.py`), async server + sync client.

1. **Inference server** (no Isaac; needs PyTorch + project `utils` + `lerobot` on `PYTHONPATH`):

   ```bash
   python scripts/serve_push_diffusion_ws.py --checkpoint checkpoints/best_diffusion.pt --horizon 32 --device cuda:0 --host 0.0.0.0 --port 8765
   ```

   Requires the sidecar **`checkpoints/best_diffusion.norm.json`** (or whatever matches your checkpoint path). `--horizon` must match training / the client scripts.

2. **Simulator client** — point **`--policy-host`** at the server IP and set **`--policy-port`** (default `8765`):

   ```bash
   python scripts/eval_model.py --task Isaac-Pusht-v0 --policy-host 192.168.1.10 --policy-port 8765 --horizon 32 ...
   python scripts/record_model_replay.py --policy-host 192.168.1.10 --policy-port 8765 --horizon 32 ...
   ```

   When `--policy-host` is set, the client does **not** load local weights; observations are sent as raw images + 21-D state, and the server returns a **denormalized** flat action vector `(horizon * 8,)`.

**Dependencies** for the server/client path: `websockets`, `msgpack` (install in the environment that runs `serve_push_diffusion_ws.py` and in the Isaac env if not already present).

**Health check:** `GET http://<host>:<port>/healthz` → `OK` (handled by the WebSocket server process).

## TensorBoard Visualization

### 1. Training Curves

TensorBoard is enabled by default for `scripts/train.py`. To turn it off or set a run name:

```bash
python scripts/train.py --horizon 32 --tb-logdir runs --tb-run-name lerobot_dp_h32_exp1
python scripts/train.py --no-tensorboard
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

