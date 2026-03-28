
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