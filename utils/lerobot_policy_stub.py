"""为 LeRobot 的 `lerobot.policies` 注入占位包，避免执行官方 `policies/__init__.py`。

背景
----
`from lerobot.policies.diffusion.xxx import ...` 会先加载父包 `lerobot.policies`。
vendored 的 `lerobot/policies/__init__.py` 在导入时就会执行顶层语句，链式 import
pi0、xvla 等策略，进而依赖较新的 `transformers`（例如 `transformers.masking_utils`）。
若环境版本偏旧，会出现「只想用 diffusion 却整包报错」的情况。

做法
----
在**任何** `import lerobot.policies...` 之前，把占位模块写入 `sys.modules`：

- `lerobot.policies`：空包，仅设置 `__path__` 指向磁盘上的 `.../policies/`。
  Python 认为该包已加载，**不会再执行** 真实的 `policies/__init__.py`。
- `lerobot.policies.diffusion`：同理，指向 `.../policies/diffusion/`，便于子模块按路径解析。

之后加载 `modeling_diffusion` 等文件时，只走 diffusion 子树，不触发 pi0 那条依赖链。

注意
----
- 若代码依赖 `from lerobot.policies import SomeConfig`（依赖 `__init__.py` 里 re-export 的名字），
  在 stub 之后可能不可用；本项目只从 `lerobot.policies.diffusion.*` 等子路径导入，故可接受。
- 必须在首次 import `lerobot.policies` 之前调用 `stub_lerobot_policies_packages`。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def stub_lerobot_policies_packages(project_root: Path | None = None) -> None:
    """若尚未加载，则注册占位包，使子模块仍能从 `__path__` 在磁盘上找到。"""
    root = (project_root or Path(__file__).resolve().parents[1]) / "lerobot" / "src" / "lerobot"
    p_policies = root / "policies"

    if "lerobot.policies" not in sys.modules:
        m = types.ModuleType("lerobot.policies")
        m.__path__ = [str(p_policies)]  # 命名空间包路径，供 importlib 解析子模块
        sys.modules["lerobot.policies"] = m

    p_diff = p_policies / "diffusion"
    if "lerobot.policies.diffusion" not in sys.modules:
        m = types.ModuleType("lerobot.policies.diffusion")
        m.__path__ = [str(p_diff)]
        sys.modules["lerobot.policies.diffusion"] = m
