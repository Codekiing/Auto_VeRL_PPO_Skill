---
name: verl-agent-training
description: VERL 双卡 A100(80G) 自定义 Agent 训练链路（Qwen2.5-0.5B）
---

# Skill: VERL 双卡 A100(80G) 自定义 Agent 训练链路（Qwen2.5-0.5B）

## 固定参数
- 框架：`VERL`
- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 数据集：`自定义`（通过联网检索）
- Agent 类型：`用户交互输入`
- 训练算法：`PPO（强制）`

---

## 强制约束（禁止偏离）
1. **只允许 VERL PPO**：入口命令必须是 `python3 -m verl.trainer.main_ppo`。  
2. **禁止 SFT**：不得调用任何 SFT 训练入口（如 `main_sft`、`trl sft`、`supervised fine-tuning`）。  
3. **若出现 SFT 相关配置/命令，立即停止并回滚到 PPO 脚本。**  

---

## 目标链路（必须按顺序执行）
1. 读取用户提供的**空白目录路径**
2. 与用户交互，确认 `AGENT_TYPE`
3. 联网搜索对应数据集并让用户确认
4. 在该路径配置基础环境
5. 下载数据集并转换为 `train.parquet/test.parquet`
6. 下载模型并启动 PPO 训练
7. 若报错，按“即时修复清单”处理后重试

---

## 一键脚本（复制到空白目录后执行）
保存为 `run_verl_skill.sh`：

> **执行纪律（强制）**  
> Claude Code 必须将下方脚本原样保存为 `run_verl_skill.sh`，随后只执行：  
> `chmod +x run_verl_skill.sh && bash run_verl_skill.sh`  
> 除了用户显式要求，不允许在运行前/运行中私自改脚本、换训练入口或改算法。
> 本脚本默认非交互执行，不使用 `source` 激活环境，避免出现 “Do you want to proceed?” 交互确认。

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "========== [0/9] Input =========="
WORKDIR="${WORKDIR:-$PWD}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

AGENT_TYPE="${AGENT_TYPE:-general}"
export AGENT_TYPE

echo "========== [1/9] System deps =========="
apt update
apt install python3.10-venv -y
apt install -y git curl build-essential python3-pip

echo "========== [2/9] Create venv =========="
python3 -m venv .venv
VENV_PY="$WORKDIR/.venv/bin/python"
VENV_PIP="$WORKDIR/.venv/bin/pip"
"$VENV_PY" -m pip install --upgrade pip==25.0.1 setuptools==75.8.0 wheel==0.45.1

echo "========== [3/9] Clone verl =========="
if [ ! -d "verl" ]; then
  git clone https://github.com/verl-project/verl.git
fi
cd verl

echo "========== [4/9] Install verl runtime =========="
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
"$VENV_PIP" install --no-deps -e .

echo "========== [5/9] Install minimal compatible deps =========="
# 关键：固定一组已验证组合，避免自动升级导致的依赖漂移
"$VENV_PIP" uninstall -y trl transformers huggingface_hub numpy tokenizers datasets pyarrow || true
"$VENV_PIP" install --no-cache-dir \
  "numpy==1.26.4" \
  "transformers==4.56.1" \
  "huggingface_hub==0.36.2" \
  "trl==0.11.4" \
  "datasets==2.21.0" \
  "pyarrow==17.0.0" \
  "tokenizers==0.22.0" \
  "matplotlib==3.9.2"

# 冲突策略：先忽略非关键冲突，只修复会影响训练的依赖
"$VENV_PIP" check || echo "[WARN] 存在依赖冲突，先继续；仅在训练受影响时做定向修复。"

# 关键训练依赖自检（只检查会直接影响训练链路的模块）
if ! "$VENV_PY" - <<'PY'
import importlib, sys
required = ["verl", "torch", "transformers", "trl", "datasets", "pyarrow", "ray", "vllm"]
bad = []
for m in required:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, str(e)))
if bad:
    print("[ERROR] 关键依赖异常：")
    for m, e in bad:
        print(f"  - {m}: {e}")
    sys.exit(2)
print("[OK] 关键训练依赖可导入")
PY
then
  echo "[ERROR] 仅修复关键依赖后重试，不做全量重装。"
  exit 3
fi

echo "========== [6/9] Online search datasets by AGENT_TYPE =========="
"$VENV_PY" - <<'PY'
import json, os
from huggingface_hub import HfApi
agent_type = os.environ.get("AGENT_TYPE", "")
api = HfApi()
items = []
for ds in api.list_datasets(search=agent_type, limit=20, full=True):
    items.append({
        "id": ds.id,
        "downloads": int(getattr(ds, "downloads", 0) or 0),
        "likes": int(getattr(ds, "likes", 0) or 0),
    })
items = sorted(items, key=lambda x: (x["downloads"], x["likes"]), reverse=True)[:10]
print("候选数据集 Top10:")
for i, d in enumerate(items, 1):
    print(f"{i:>2}. {d['id']} (downloads={d['downloads']}, likes={d['likes']})")
with open("dataset_candidates.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)
print("已写入 dataset_candidates.json")
if not items:
    raise RuntimeError("未搜索到候选数据集，请设置更具体的 AGENT_TYPE")
with open("dataset_selected.txt", "w", encoding="utf-8") as f:
    f.write(items[0]["id"])
print(f"默认选择 Top1 数据集: {items[0]['id']}")
PY

DATASET_ID="${DATASET_ID:-$(cat dataset_selected.txt)}"

echo "========== [7/9] Prepare dataset to parquet =========="
export DATASET_ID
export DATA_DIR="$WORKDIR/data/domain_data"
mkdir -p "$DATA_DIR"

"$VENV_PY" - <<'PY'
import json, os
from datasets import load_dataset

out_dir = os.environ["DATA_DIR"]
dataset_id = os.environ["DATASET_ID"]

def pick(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None

train = load_dataset(dataset_id, split="train")
try:
    valid = load_dataset(dataset_id, split="validation")
except Exception:
    valid = load_dataset(dataset_id, split="test")

prompt_col = pick(train.column_names, ["question", "prompt", "instruction", "input", "query", "problem"])
resp_col = pick(train.column_names, ["answer", "output", "response", "solution", "label"])
if prompt_col is None or resp_col is None:
    raise RuntimeError(f"无法自动识别字段: {train.column_names}")

def mapper(x):
    return {"prompt": str(x[prompt_col]), "response": str(x[resp_col])}

train = train.map(mapper, remove_columns=train.column_names)
valid = valid.map(mapper, remove_columns=valid.column_names)

if len(train) == 0 or len(valid) == 0:
    raise RuntimeError("训练/验证数据为空")

train.to_parquet(f"{out_dir}/train.parquet")
valid.to_parquet(f"{out_dir}/test.parquet")

with open(f"{out_dir}/data_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset_id": dataset_id,
        "prompt_col": prompt_col,
        "response_col": resp_col,
        "train_size": len(train),
        "val_size": len(valid)
    }, f, ensure_ascii=False, indent=2)

# 导出样例给用户检查（前3条）
sample = [{"prompt": train[i]["prompt"], "response": train[i]["response"]} for i in range(min(3, len(train)))]
with open(f"{out_dir}/data_samples.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print("[OK] data prepared")
PY

echo \"[CHECK] VERL 数据格式要求: train/test parquet 必须包含 prompt,response 两列\"
"$VENV_PY" - <<'PY'
import os
import pyarrow.parquet as pq
data_dir = os.environ["DATA_DIR"]
for fn in ["train.parquet", "test.parquet"]:
    path = os.path.join(data_dir, fn)
    tbl = pq.read_table(path)
    cols = set(tbl.column_names)
    if not {"prompt", "response"}.issubset(cols):
        raise RuntimeError(f"{fn} 不符合 VERL 格式，缺少 prompt/response 列，当前列={list(tbl.column_names)}")
print("[OK] parquet 列检查通过")
PY

echo \"[CHECK] 已生成样例文件: $DATA_DIR/data_samples.json\"
echo \"[CHECK] 请先人工确认样例内容再继续训练\"
cat \"$DATA_DIR/data_samples.json\"

echo "========== [8/9] Start PPO training =========="
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1

# 保存本次训练参数（可追溯）
cat > "$WORKDIR/run_params.txt" <<'PARAMS'
framework=verl
algo=ppo
model=Qwen/Qwen2.5-0.5B-Instruct
train_batch_size=256
max_prompt_length=512
max_response_length=256
actor_lr=1e-6
critic_lr=1e-5
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=2
rollout_gpu_memory_utilization=0.35
trainer_n_gpus_per_node=2
trainer_total_epochs=15
trainer_save_freq=10
trainer_test_freq=10
PARAMS
echo "[OK] 已保存训练参数: $WORKDIR/run_params.txt"

"$VENV_PY" -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.ppo_micro_batch_size_per_gpu=2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 2>&1 | tee "$WORKDIR/train.log"

echo "========== [9/9] Plot training curves =========="
"$VENV_PY" - <<'PY'
import os, re
import matplotlib.pyplot as plt

workdir = os.environ.get("WORKDIR", ".")
log_path = os.path.join(workdir, "train.log")
out_png = os.path.join(workdir, "training_curves.png")

patterns = {
    "loss": re.compile(r"loss[:=]\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)"),
    "reward": re.compile(r"reward[:=]\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)"),
    "kl": re.compile(r"\\bkl[:=]\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)"),
}
series = {k: [] for k in patterns}

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        for k, p in patterns.items():
            m = p.search(line.lower())
            if m:
                try:
                    series[k].append(float(m.group(1)))
                except ValueError:
                    pass

plt.figure(figsize=(10, 6))
drawn = 0
for k, vals in series.items():
    if vals:
        plt.plot(vals, label=k)
        drawn += 1

if drawn == 0:
    with open(os.path.join(workdir, "training_curves_note.txt"), "w", encoding="utf-8") as f:
        f.write("未在 train.log 中解析到 loss/reward/kl 数值，无法自动绘图。\\n")
else:
    plt.title("VERL PPO Training Curves")
    plt.xlabel("Logged Step Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 已生成图表: {out_png}")
PY

echo "========== [10/10] DONE =========="
```

---

## PPO 合规检查（每次运行后必做）

```bash
# 1) 检查是否真的调用了 PPO 入口
grep -n "verl.trainer.main_ppo" run_verl_skill.sh

# 2) 检查脚本中是否混入 SFT
if grep -Eqi "main_sft|\\bsft\\b|supervised fine-tuning" run_verl_skill.sh; then
  echo "[ERROR] 检测到 SFT 相关内容，违反本 Skill 约束"
  exit 3
fi

# 3) 检查训练日志关键字（确认是 PPO 训练链路）
grep -n "main_ppo\\|ppo" "$WORKDIR/train.log" | head
```

如果第 2 步触发，必须删除 SFT 相关配置后重跑。

```bash
# 4) 可选：执行前做脚本完整性确认（保存后立即记录）
sha256sum run_verl_skill.sh
```

---

## 即时报错修复清单（遇错就按此顺序）
1. `trl/value head` 报错：
   ```bash
   "$WORKDIR/.venv/bin/pip" install "trl==0.11.4"
   ```
2. 依赖冲突（先忽略非关键，只修复影响训练的项）：
   ```bash
   "$WORKDIR/.venv/bin/pip" check
   # 只修复训练关键依赖，不做全量升级/降级
   "$WORKDIR/.venv/bin/python" -c "import verl, torch, transformers, trl, datasets, pyarrow, ray, vllm; print('关键依赖OK')"
   ```
3. `flash-attn` 报错：先忽略，不阻断主流程。
4. OOM：依次降低
   - `data.max_response_length: 256 -> 128`
   - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 2 -> 1`
   - `critic.ppo_micro_batch_size_per_gpu: 2 -> 1`
   - `actor_rollout_ref.rollout.gpu_memory_utilization: 0.35 -> 0.30`

---

## 成功标志
- `train.log` 持续输出 step
- 周期性出现验证和 checkpoint 保存
- 产出 `run_params.txt`（训练参数快照）
- 产出 `training_curves.png`（若日志可解析）或 `training_curves_note.txt`
