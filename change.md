
下面按**对 PushT 类任务上「训练效果 / 成功率」的预期影响**排序，并标明**建议优先级**和**可暂缓项**。依据是：Diffusion Policy 论文与 LeRobot 实现里反复验证有效的设计（多帧观测、时间结构上的去噪、调度与动作空间处理），以及你们当前实现与它们的差距。

---

## 最应该先动（高优先级）

### 1. 去噪网络：从「整段轨迹一大向量 + 单 action token」改为「保留时间维的结构化去噪」

**现状**：`action_dim = 单步维度 × horizon`，`TransformerActionHead` 里只有一个 action token，整条 future 被压成一条向量做扩散。  
**问题**：时间步之间强相关（平滑轨迹、相位一致），一维大向量 + 浅层耦合很难像 **1D U-Net** 那样沿时间做局部与多尺度建模。  
**建议**：优先对齐 LeRobot：**`(B, T, action_dim)` 输入**，用 **Conditional U-Net 1d**（或至少 **沿 T 的 Conv1d + 与 obs 的全局 FiLM 条件**），与 LeRobot 的 `DiffusionConditionalUnet1d` 同构。  
**预期**：对 **长 horizon、多步动作相关性** 的任务通常提升最大。  
**工作量**：大（模型 + `train_step`/`sample` 张量形状 + eval）。

---

### 2. 观测：从「单帧」改为「多帧历史 `n_obs_steps`」

**现状**：`collate_fn` 里图像/state 只取窗口起点。  
**问题**：Diffusion Policy / LeRobot 默认 **`n_obs_steps=2`**（可更多），用速度、接触等 **短时动态** 补单帧视觉的歧义。  
**建议**：数据管道堆叠 `n_obs_steps` 帧（与 LeRobot 的 `observation_delta_indices` 一致思想），编码后 **拼进全局条件**（或时间维显式建模）。  
**预期**：对 **接触、滑动、遮挡** 敏感的操作常明显受益。  
**工作量**：中（dataset/collate、模型条件维、内存）。

---

### 3. 噪声调度：从「线性 beta」改为与 LeRobot 一致的 **diffusers 调度**

**现状**：`torch.linspace` + 手写采样。  
**建议**：使用 **`DDPMScheduler`**，**`beta_schedule="squaredcos_cap_v2"`**，`beta_start`/`beta_end`/`num_train_timesteps` 与 LeRobot 对齐；训练用 `add_noise`，推理用 `step`。可选 **DDIM** 做更快推理。  
**预期**：单独一项未必最大，但**成本低、与大量已发布配置一致**，利于复现和调参。  
**工作量**：小–中（依赖 `diffusers`，改 `trainer`）。

---

### 4. 动作与推理：若采用 LeRobot 式 **MIN_MAX 归一化到约 [-1,1]**，则加上 **`clip_sample`**

**现状**：state/action 用 **全局 mean/std**；扩散过程无 clip。  
**问题**：LeRobot 默认对 action 用 **MIN_MAX**，推理 **`clip_sample_range=1.0`**，避免反推时越界、与训练分布不一致。  
**建议**：二选一或组合：  
- 与 LeRobot 一致：**ACTION（及 STATE）用 min-max 到 [-1,1]**，并打开 **clip**；或  
- 保留 z-score，但需在验证集上看 **动作范数是否爆炸**，必要时 **裁剪或 tighter 归一化**。  
**预期**：减少 **推理发散、饱和**，对稳定性有帮助。  
**工作量**：小–中（改 `normalization` 与训练脚本、保存 stats 格式）。

---

## 建议第二阶段（中优先级，性价比高）

### 5. 视觉编码：**Spatial Softmax（关键点）** 替代「GAP + 单 token」

**现状**：ResNet18 GAP → 每图一个 token。  
**建议**：对齐 LeRobot：`layer4` 特征图 → **SpatialSoftmax**（默认 32 keypoints），再 MLP；可选 **resize/crop** 与 **GroupNorm 骨干**（注意与预训练权重冲突时的取舍）。  
**预期**：对 **平面内精细位姿 / 接触位置** 往往优于单全局向量。  
**工作量**：中。

---

### 6. 优化与调度：**Adam + betas + cosine + warmup**

**现状**：AdamW，`weight_decay=1e-4`，无 LR 调度。  
**LeRobot**：`betas=(0.95, 0.999)`，`weight_decay=1e-6`，**cosine + warmup 500**。  
**建议**：至少加 **warmup + cosine**；betas/weight decay 可向 LeRobot 靠拢做对比实验。  
**预期**：训练曲线更稳、略提最终表现的情况常见，但通常 **不如 1–2 项结构改动**。  
**工作量**：小。

---

### 7. 数据与 loss：**`action_is_pad` mask**（若以后用滑动窗口 + 填充）

**现状**：有效窗口训练，可能暂无 padding。  
**建议**：若改为 LeRobot 式 **horizon 与帧采样**（含尾部丢弃 `drop_n_last_frames` 等），需 **mask 掉 pad 段 loss**。  
**预期**：仅在有 pad 时重要。  
**工作量**：视数据管道而定。

---

## 可先不动或低优先级（除非实验表明是瓶颈）

| 项 | 说明 |
|----|------|
| **每相机独立 ResNet** | LeRobot 可选；共享编码器常已够用，优先保证多帧 + U-Net。 |
| **`torch.compile`** | LeRobot 可选；提速为主，对精度影响次要。 |
| **预训练 ResNet** | 你们已有 ImageNet；与 LeRobot「GN + 无预训练」是不同路线，不必盲目改掉，可后验对比。 |
| **`n_action_steps` 执行队列** | 主要影响 **在线执行 / 控制频率**，对 **离线拟合 loss** 不是第一因素；结构对齐后再做执行策略更顺。 |
| **`embed_dim` / `obs_layers` 等** | 在架构与调度未对齐前，单纯加宽加深 **性价比低**。 |

---

## 推荐落地顺序（最小实验路径）

1. **调度器**（diffusers + squaredcos）→ 快速对齐基线。  
2. **`n_obs_steps` 堆叠** → 数据与条件维改动相对可控。  
3. **U-Net 1d 时间建模** → 最大结构红利，建议单独分支开发。  
4. **归一化 + clip** 与 **Spatial Softmax** → 与 3 并行或紧随其后做消融。  
5. **LR scheduler / Adam 细节** → 长期默认配置微调。

如果你希望下一步落到代码上，可以从 **「仅换调度器 + 多帧 obs」** 或 **「直接上 U-Net 1d」** 两条线里选一条作为主 PR，我可以按你当前 `train.py` / `dataset.py` 结构拆具体改哪些文件、哪些张量形状。



下面用**符号 + 你们仓库里的真实默认**把两条链路从「数据进模型」到「噪声预测输出」逐张量写清楚。符号约定：

- \(B\)：batch size  
- \(D_s\)：`observation.state` 最后一维（由数据集决定，记为 state 维度）  
- \(D_a\)：单步 `action` 维度（数据集 `all_actions.shape[-1]`）  
- \(H\)：horizon（`DatasetConfig.horizon` / `args.horizon`）  
- \(H_i, W_i\)：单张图高宽（你们数据里是 `CHW`，常见如 96×96 或 224×224，以你数据为准）  
- \(E\)：`embed_dim`（默认 **256**，见 `train_default.json` 的 `diffusion_model`）

---

# 一、你们当前模型（`DiffusionTransformerModel`）

## 1. 数据 batch（`utils/dataset.py` 的 `collate_fn`）

| 字段 | 形状 | 含义 |
|------|------|------|
| `obs["observation.front_wrist_camera_image"]` | \((B, 3, H_i, W_i)\) | 窗口**起始帧**前腕相机，`float32`，已归一化到约 \([0,1]\)（若原为 uint8 则 `/255`） |
| `obs["observation.back_wrist_camera_image"]` | \((B, 3, H_i, W_i)\) | 同上，后腕相机 |
| `obs["observation.state"]` | \((B, D_s)\) | 与动作窗口起点对齐的**单帧** proprio |
| `action` | \((B, H, D_a)\) | 从起点往后 \(H\) 步的动作序列 |

训练时 `DiffusionTrainer._flatten_action` 把动作变成：

| 张量 | 形状 |
|------|------|
| `action`（flatten 后） | \((B, H \cdot D_a)\) |

你们 `train.py` 里 **`action_dim = D_a * H`**，与上式一致。

---

## 2. 前向扩散（训练，`train_step`）

| 变量 | 形状 |
|------|------|
| `action` | \((B, H \cdot D_a)\) |
| `t` | \((B,)\)，整数，取值 \(0 \ldots T_{\text{diff}}-1\)，默认 \(T_{\text{diff}}=100\) |
| `noise` \(\epsilon\) | \((B, H \cdot D_a)\) |
| \(\bar\alpha_t\)（`alpha_bar_t`） | \((B, 1)\) broadcast 到动作维 |
| `noisy_action` \(x_t\) | \((B, H \cdot D_a)\) |
| 模型预测 `pred_noise` | \((B, H \cdot D_a)\) |
| `loss` | 标量，对整向量做 MSE |

也就是说：**扩散变量是一条长度为 \(H \cdot D_a\) 的向量**，没有在建模里保留 \((H, D_a)\) 两个轴。

---

## 3. 观测编码 `MultimodalTransformerEncoder`

输入 `obs` 同上。

**ResNet18 图像支路**（每张图一套权重，代码里是**同一个** `ResNet18ImageEncoder` 实例依次调 front/back —— 即两相机**共享**同一 ResNet+proj）：

| 步骤 | 张量 | 形状 |
|------|------|------|
| 输入图像 | `front` / `back` | \((B, 3, H_i, W_i)\) |
| ImageNet 归一化后 | 同上 | \((B, 3, H_i, W_i)\) |
| backbone 输出 flatten | `feat` | \((B, 512)\) |
| `proj` | 每图一个 token | \((B, 1, E)\)，默认 \(E=256\) |

**State 支路**：

| 步骤 | 形状 |
|------|------|
| `state` | \((B, D_s)\) |
| `Linear(D_s, E)` + `unsqueeze(1)` | \((B, 1, E)\) |

**Token 拼接**（加 `pos` / `type` 后）：

| 张量 | 形状 |
|------|------|
| `tokens` | \((B, 3, E)\) —— 顺序：**front \| back \| state** |

**Obs TransformerEncoder**（`obs_layers` 默认 4，`num_heads` 8，`d_ff` 默认 \(4E=1024\)）：

| 张量 | 形状 |
|------|------|
| `encoded` | \((B, 3, E)\) |
| `pooled`（返回值之一，扩散头未用） | \((B, E)\) |

**输出给扩散头**：`obs_tokens`，形状 **\((B, 3, E)\)**。

---

## 4. 扩散头 `TransformerActionHead`

| 输入 | 形状 |
|------|------|
| `x_action`（noisy 动作向量） | \((B, H \cdot D_a)\) |
| `t` | \((B,)\) |
| `obs_tokens` | \((B, 3, E)\) |

**时间嵌入**（`SinusoidalTimeEmbedding(E)`）：

| 步骤 | 形状 |
|------|------|
| `time_embed(t)` | \((B, E)\) |
| `time_proj`（两线性+GELU） | \((B, E)\) → `unsqueeze(1)` → **\((B,1,E)\)** |

**动作 token**：

| 步骤 | 形状 |
|------|------|
| `Linear(H·D_a, E)` + `unsqueeze(1)` | **\((B, 1, E)\)** |

**拼接序列**：

| 张量 | 形状 |
|------|------|
| `[action_token, time_token, obs_tokens]` | \((B, 5, E)\) —— **1+1+3=5 个 token** |

**Head TransformerEncoder**（`head_layers` 默认 3）：

| 张量 | 形状 |
|------|------|
| `encoded` | \((B, 5, E)\) |

**读出**：

| 步骤 | 形状 |
|------|------|
| 取 **`encoded[:, 0]`**（仅第一个 token） | \((B, E)\) |
| `out = Linear(E, H·D_a)` | **\((B, H \cdot D_a)\)** |

整段 \(H\) 步轨迹的噪声预测，都经过**同一个「动作 token」**与 5-token 自注意力汇总，再一次性线性映射回 \(H \cdot D_a\)。

---

## 5. 推理 `sample_actions`

| 变量 | 形状 |
|------|------|
| 初始 \(x_T\) | \((B, H \cdot D_a)\) |
| 每步 `pred_noise` | \((B, H \cdot D_a)\) |
| 输出 | \((B, H \cdot D_a)\) |

`DiffusionPolicy.act` 里若单条环境 obs 会 `unsqueeze(0)`，得到 \(B=1\)。

---

## 6. 参数量级小结（你们）

- 与 **\(H \cdot D_a\)** 线性相关的层：  
  - `TransformerActionHead.action_proj`：**\((H·D_a) \times E\)**  
  - `TransformerActionHead.out`：**\(E \times (H·D_a)\)**  
  当 horizon 大时，这两层很大。  
- 与 **\(D_s\)** 相关：`state_embed` 为 **\(D_s \times E\)**。  
- 图像侧：标准 ResNet18 + `512→E`。

---

# 二、LeRobot `DiffusionModel`（默认配置思路）

下面用 LeRobot `configuration_diffusion.py` 的**默认值** + `modeling_diffusion.py` 的逻辑。记：

- \(N_o =\) `n_obs_steps`（默认 **2**）  
- \(T =\) `horizon`（默认 **16**）  
- \(D_a^{(lr)}\)：`action_feature.shape[0]`（由数据集特征定义，PushT 上常见为 2）  
- \(N_c\)：相机个数（`len(image_features)`）  
- `feature_dim`（每相机一条向量）= `spatial_softmax_num_keypoints * 2` = **32×2 = 64**（经 `Linear(64,64)+ReLU` 后仍是 64 维输出）  
- `down_dims` = **(512, 1024, 2048)**，故下采样 **3 次**，时间长度需被 \(2^3=8\) 整除（默认 horizon=16 满足）

---

## 1. 数据 batch（训练 `compute_loss` 期望）

| 字段 | 形状 |
|------|------|
| `observation.state` | \((B, N_o, D_s^{(lr)})\) —— \(D_s^{(lr)}\) 为配置里的 robot state 维 |
| `observation.images`（多相机堆叠后） | \((B, N_o, N_c, 3, H^{(lr)}, W^{(lr)})\) |
| `action` | \((B, T, D_a^{(lr)})\) |
| `action_is_pad` | \((B, T)\) bool（可选 mask） |

**注意**：动作始终保持 **\((B, T, D_a)\)**，**不** flatten 成一条长向量进网络。

---

## 2. 全局条件 `_prepare_global_conditioning`

对每个时间步 \(s=1\ldots N_o\) 拼特征（最后一维 concat）：

- 状态：\((B, N_o, D_s^{(lr)})\)  
- 图像：每步 \(N_c\) 路相机，每路 encoder 输出 64 维 → \((B, N_o, N_c \cdot 64)\)  
- 若有 `environment_state` 再拼上

`torch.cat(..., dim=-1)` 得到 \((B, N_o, C_{\text{step}})\)，其中  

\[
C_{\text{step}} = D_s^{(lr)} + N_c \times 64 + D_{\text{env}} \ (\text{若有})
\]

再 `flatten(start_dim=1)`：

| 张量 | 形状 |
|------|------|
| `global_cond` | \((B, N_o \cdot C_{\text{step}})\) |

该向量整块送入 U-Net 的 FiLM 条件（与扩散步嵌入拼接）。

**示例**（单相机 PushT、无 env state）：\(N_o=2, N_c=1, D_s=?\)  
\(C_{\text{step}} = D_s + 64\)，`global_cond` 维数 = \(2(D_s+64)\)。

---

## 3. 视觉 `DiffusionRgbEncoder`（每路相机）

| 步骤 | 形状 |
|------|------|
| 输入（单张） | \((B', 3, H, W)\)（\(B'\) 可为 \(B\cdot N_o\cdot N_c\) 展平 batch） |
| backbone（ResNet 到 layer4） | 特征图 \((B', C_{\text{map}}, h, w)\) |
| SpatialSoftmax | \((B', K_{\text{kp}}, 2)\)，\(K_{\text{kp}}=32\) → flatten **\((B', 64)\)** |
| `Linear(64,64)+ReLU` | **\((B', 64)\)** |

默认 **`use_separate_rgb_encoder_per_camera=False`** 时，多相机共享同一 encoder，在 batch 维拼好再喂。

---

## 4. 去噪网络 `DiffusionConditionalUnet1d`

**输入动作轨迹**：

| 张量 | 形状 |
|------|------|
| `sample` / `noisy_trajectory` | \((B, T, D_a^{(lr)})\) |

**Conv1d 前重排**（`einops`）：

| 张量 | 形状 |
|------|------|
| `x` | \((B, D_a^{(lr)}, T)\) —— **通道 = 单步动作维**，**长度 = 时间 horizon**

**扩散步编码**（`diffusion_step_embed_dim` = **128**）：

| 步骤 | 形状 |
|------|------|
| `DiffusionSinusoidalPosEmb(128)(t)` | \((B, 128)\) |
| MLP | \((B, 128)\) |

**FiLM 条件维度**：

| 张量 | 形状 |
|------|------|
| `timesteps_embed` | \((B, 128)\) |
| `global_cond` | \((B, G)\)，\(G = N_o \cdot C_{\text{step}}\) |
| `global_feature = cat(..., dim=-1)` | \((B, 128 + G)\) |

**U-Net 通道（默认 `down_dims`）**  
encoder 路径（概念上）：

- 块 0：\(D_a \rightarrow 512\)，再 Res×2，Conv stride 2 下采样  
- 块 1：\(512 \rightarrow 1024\)，下采样  
- 块 2：\(1024 \rightarrow 2048\)，**最后一级不下采样**  

时间维长度：\(T \rightarrow T/2 \rightarrow T/4 \rightarrow T/8\)（当 \(T=16\) 得 \(16\to8\to4\to2\)）。

**输出**：

| 张量 | 形状 |
|------|------|
| 经 `final_conv` | \((B, D_a^{(lr)}, T)\) |
| rearrange 回 | **\((B, T, D_a^{(lr)})\)** |

预测目标：默认 `prediction_type="epsilon"` 时与噪声 \(\epsilon\) 同形 **\((B, T, D_a)\)**。

---

## 5. 扩散调度（与维度无关但需对齐）

- `num_train_timesteps` 默认 **100**  
- `beta_schedule="squaredcos_cap_v2"`，`beta_start/end` 与你们数值可相同但**曲线形状不同**  
- 可加噪声 / 反推用 `diffusers` API；可选推理步数 `num_inference_steps`、**`clip_sample`**

---

# 三、并排对照（抓住「维数」差异）

| 项目 | 你们 | LeRobot |
|------|------|---------|
| 动作张量进扩散核心 | \((B, H\cdot D_a)\) 向量 | \((B, T, D_a)\) 三维 |
| 时间轴 \(H\) / \(T\) | 压进向量维，**无**沿时间的卷积 | **Conv1d** 沿 \(T\) 多尺度 |
| 单步动作维在 U-Net 里 | N/A（与 horizon 乘在一起） | **通道维** \(D_a\) |
| 观测时间 | **1 帧** \((B,D_s)\)、\((B,3,H,W)\) | **\(N_o\) 帧** \((B,N_o,D_s)\)、\((B,N_o,N_c,3,H,W)\) |
| 条件进入去噪器 | 3 个 obs token 与 action/time **自注意力** | **扁平向量** \((B, G)\) + 步嵌入 → **FiLM** 进各卷积层 |
| 图像语义 | GAP → 每图 **1×E** token | 特征图 → SpatialSoftmax → **64 维/相机** |
| 扩散头输入 token 数 | **5**（1 动作 + 1 时间 + 3 obs） | U-Net 无 token，为 **\((D_a, T)\)** 特征图 |
| 噪声预测形状 | \((B, H\cdot D_a)\) | \((B, T, D_a)\) |

---

# 四、代入一个具体数字例子（便于脑补）

假设：**\(D_s=21\)**，**\(D_a=2\)**，**\(H=16\)**，**\(E=256\)**，图像 **\(3\times224\times224\)**（仅用于说明，以你实际 `all_states` / `all_actions` 为准）。

**你们**：

- `action` flatten → \((B, 32)\)  
- `obs_tokens` → \((B, 3, 256)\)  
- 拼接序列 → \((B, 5, 256)\)  
- 输出噪声 → \((B, 32)\)

**LeRobot**（默认 \(N_o=2, T=16, N_c=2\) 时，仅示例）：

- `action` → \((B, 16, 2)\)  
- U-Net 内部 → \((B, 2, 16)\) 再下采样到时间长度 2  
- 若 \(D_s=21\)，每步 \(C_{\text{step}}=21+64+64=149\)，`global_cond` → \((B, 2\times149)=(B,298)\)  
- FiLM 条件维 → \(128+298=426\)

---

如果你愿意，我可以根据你**当前数据集里真实的** `all_states.shape`、`all_actions.shape`、`front` 图像 shape 打一条「只属于你的」维度表（把上面符号全部换成数字）。只要把 `train.py` 启动时打印的那行 `[Init] Dimensions: state=..., action=...` 和图像 tensor 的 shape 发我即可。