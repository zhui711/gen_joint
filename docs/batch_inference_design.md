# CXR Multi-View Generation — 批量推理脚本设计文档

> **项目**：基于 OmniGen 的 CXR 多角度合成  
> **脚本名称**：`batch_infer.py`（位于 `gen_code/` 根目录）  
> **作者**：AI Code Engineer  
> **日期**：2025-02-27  

---

## 1. 整体架构与流程

### 1.1 端到端数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        Launcher (Shell)                         │
│  CUDA_VISIBLE_DEVICES=2,3  python3 batch_infer.py  --args ...   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  主进程 Main  │
                    │ 1. 解析参数    │
                    │ 2. 读取 JSONL  │
                    │ 3. 均分为 N 份  │
                    └──┬───────┬───┘
                       │       │
            ┌──────────▼─┐   ┌─▼──────────┐
            │ Worker 0   │   │ Worker 1   │
            │ cuda:0     │   │ cuda:1     │
            │ chunk[0]   │   │ chunk[1]   │
            │ Pipeline   │   │ Pipeline   │
            │ tqdm bar 0 │   │ tqdm bar 1 │
            └──────┬─────┘   └─────┬──────┘
                   │               │
                   ▼               ▼
            ┌──────────────────────────────┐
            │   输出目录 (共享文件系统)       │
            │   output_dir/                │
            │     LIDC-IDRI-XXXX/          │
            │       YYYY.png               │
            └──────────────────────────────┘
```

### 1.2 执行步骤摘要

| 步骤 | 描述 |
|------|------|
| **S1** | 主进程解析命令行参数，读取并解析完整的 JSONL 文件 |
| **S2** | 将 JSONL 数据按索引均匀切分为 `num_gpus` 个 chunk |
| **S3** | 使用 `torch.multiprocessing.spawn` 启动 `num_gpus` 个 worker 进程 |
| **S4** | 每个 worker 在指定 `cuda:{rank}` 上实例化独立的 `OmniGenPipeline`，加载 base model + LoRA |
| **S5** | 每个 worker 对分配到的 chunk 逐条推理，生成图像并保存至输出目录 |
| **S6** | 所有 worker 完成后，主进程打印汇总统计（成功/失败数）|

---

## 2. 多卡并行与数据分片方案详解

### 2.1 并行策略选择

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| `accelerate` 分布式 | 官方支持、灵活 | Pipeline 内部未适配 DDP，改造量大 | ✗ |
| `torch.multiprocessing.spawn` | 轻量、无需改造 Pipeline | 每个进程独立加载模型 | **✓** |
| 手动 `subprocess` 指定 rank | 最灵活 | 需要额外脚本传参 | ✗ |

**最终选择**：`torch.multiprocessing.spawn`。理由：
1. OmniGen 的 `OmniGenPipeline` 是单卡推理设计，不支持将单次推理分布到多卡上；最适合的并行方式是**数据并行**——每张卡加载完整模型，各自处理不同数据。
2. `spawn` 方式可以在同一脚本内完成进程管理，代码简洁。
3. H100 80GB 显存充足，两张卡各自加载一份模型不会 OOM。

### 2.2 数据分片逻辑

```python
# 主进程中
all_samples = [json.loads(line) for line in open(jsonl_path)]
chunks = []
chunk_size = math.ceil(len(all_samples) / num_gpus)
for i in range(num_gpus):
    chunks.append(all_samples[i * chunk_size : (i + 1) * chunk_size])
```

- **分片方式**：按 JSONL 行索引连续切分（非 round-robin），保证同一患者的视图尽量落在同一个 worker 中，有利于利用文件系统的页缓存（同一 condition image `0000.png` 被重复读取时命中缓存）。
- **尾部处理**：最后一个 chunk 可能少几条数据，无需额外 padding。

### 2.3 进程启动与 GPU 绑定

```python
import torch.multiprocessing as mp

def worker_fn(rank, num_gpus, chunks, args):
    device = torch.device(f"cuda:{rank}")
    # 该进程只看到 CUDA_VISIBLE_DEVICES 映射后的第 rank 张卡
    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    if args.lora_path:
        pipe.merge_lora(args.lora_path)
    pipe.to(device)
    
    my_chunk = chunks[rank]
    # ... 推理循环 ...

if __name__ == "__main__":
    mp.spawn(worker_fn, args=(num_gpus, chunks, args), nprocs=num_gpus, join=True)
```

> **重要**：启动脚本中必须设置 `CUDA_VISIBLE_DEVICES=2,3`，使得进程内部的 `cuda:0` 对应物理卡 2，`cuda:1` 对应物理卡 3。

---

## 3. 核心参数与 Pipeline 调用伪代码

### 3.1 推理参数一览

| 参数 | 值 | 说明 |
|------|----|------|
| `height` / `width` | 256 / 256 | CXR 图像分辨率，与训练一致 |
| `use_input_image_size_as_output` | `True` | 让输出分辨率自动跟随输入图像尺寸（256x256） |
| `guidance_scale` | 2.5 | CFG 引导强度（OmniGen Image Editing 推荐值） |
| `img_guidance_scale` | 2.0 | 图像引导强度 |
| `num_inference_steps` | 50 | 去噪步数 |
| `offload_model` | `False` | H100 80GB 显存充足，关闭 CPU offload |
| `use_kv_cache` | `True` | 启用 KV Cache 加速 |
| `offload_kv_cache` | `False` | 显存充足，KV Cache 留在 GPU |
| `separate_cfg_infer` | `False` | 关闭分离 CFG 推理（显存充足时速度更快） |
| `seed` | `42` | 可复现的固定种子 |
| `dtype` | `torch.bfloat16` | 半精度推理 |

### 3.2 单条推理伪代码

```python
for sample in tqdm(my_chunk, desc=f"[GPU {rank}]", position=rank):
    try:
        prompt = sample["instruction"]
        input_images = sample["input_images"]  # ["/abs/path/to/cond.png"]
        output_gt_path = sample["output_image"]  # 仅用于推导保存路径
        
        # --- 调用 Pipeline ---
        generated_images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=256,
            width=256,
            num_inference_steps=50,
            guidance_scale=2.5,
            img_guidance_scale=2.0,
            use_input_image_size_as_output=True,
            offload_model=False,
            use_kv_cache=True,
            offload_kv_cache=False,
            separate_cfg_infer=False,
            seed=42,
        )
        
        # --- 保存生成图像 ---
        save_path = derive_save_path(output_gt_path, output_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        generated_images[0].save(save_path)
        
    except Exception as e:
        logging.error(f"[GPU {rank}] Failed on {output_gt_path}: {e}")
        continue
```

### 3.3 LoRA 加载说明

训练产出的是 LoRA adapter（`adapter_config.json` + `adapter_model.safetensors`）。在推理时：

```python
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
pipe.merge_lora("./results/cxr_finetune_lora/checkpoints/0000500")
pipe.to(device)
```

`merge_lora()` 会将 LoRA 权重合并进主模型，后续推理无额外开销。

---

## 4. 目录结构与命名规则设计

### 4.1 核心设计原则

**生成图像的保存路径必须与 Ground Truth 的 `output_image` 字段产生可推导的一一映射**，以便评估脚本进行 pairwise 读取。

### 4.2 命名映射规则

JSONL 中 `output_image` 的绝对路径格式为：

```
/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256/{patient_id}/{view_id}.png
```

例如：
```
/raid/.../img_complex_fb_256/LIDC-IDRI-0251/0001.png
```

**保存规则**：提取 `output_image` 路径中最后两个层级（`{patient_id}/{view_id}.png`），拼接到用户指定的 `--output_dir` 下：

```
{output_dir}/{patient_id}/{view_id}.png
```

具体实现：

```python
def derive_save_path(gt_path: str, output_dir: str) -> str:
    """
    gt_path:   .../img_complex_fb_256/LIDC-IDRI-0251/0001.png
    output_dir: /raid/.../output/cxr_synth_results
    return:     /raid/.../output/cxr_synth_results/LIDC-IDRI-0251/0001.png
    """
    parts = gt_path.replace("\\", "/").split("/")
    relative = os.path.join(parts[-2], parts[-1])  # LIDC-IDRI-0251/0001.png
    return os.path.join(output_dir, relative)
```

### 4.3 输出目录结构示例

```
wt/output/cxr_synth_results/
├── LIDC-IDRI-0030/
│   ├── 0001.png
│   ├── 0002.png
│   ├── ...
│   └── 1500.png
├── LIDC-IDRI-0089/
│   ├── 0001.png
│   └── ...
├── LIDC-IDRI-0163/
│   └── ...
...（共 16 个 patient 目录）
```

### 4.4 后续评估脚本的读取方式（设计预留）

评估脚本只需要：
1. 逐行读取同一份 `cxr_synth_anno_test.jsonl`
2. 对于每一行：
   - **Ground Truth** = `sample["output_image"]`（原始绝对路径）
   - **Generated** = `derive_save_path(sample["output_image"], output_dir)`
3. 加载两张图像，计算 PSNR / SSIM / LPIPS 等指标

这样 **JSONL 文件既是推理的输入清单，也是评估的索引文件**，无需额外生成对应关系表。

---

## 5. 异常处理与进度追踪

### 5.1 异常处理策略

| 异常场景 | 处理方式 |
|----------|----------|
| 输入 condition 图像文件不存在 | `try/except` 捕获，打印 warning，跳过该条，计入 `fail_count` |
| Pipeline 推理过程中 CUDA OOM | 捕获 `RuntimeError`，记录日志，执行 `torch.cuda.empty_cache()`，跳过该条 |
| 输出目录创建失败 | `os.makedirs(..., exist_ok=True)`，若仍失败则 raise |
| JSONL 中某行 JSON 解析失败 | `try/except json.JSONDecodeError`，跳过 |

### 5.2 进度追踪

```python
from tqdm import tqdm

# 每个 worker 中
for sample in tqdm(
    my_chunk,
    desc=f"[GPU {rank}]",
    position=rank,      # 多进程不同行显示
    leave=True,
    dynamic_ncols=True,
):
    ...
```

- 使用 `position=rank` 让两个 worker 的进度条分别显示在终端的不同行。
- 每条推理完成后 tqdm 自动更新。

### 5.3 运行统计汇报

每个 worker 完成后，通过 `mp.Queue` 或简单的文件写入回传统计结果：

```python
# worker 结束时
stats = {"rank": rank, "total": len(my_chunk), "success": success_count, "fail": fail_count}
# 写入 {output_dir}/stats_gpu{rank}.json
```

主进程 `join` 后读取所有 stats 文件，汇总打印：

```
=== Inference Complete ===
GPU 0: 11992 / 11992 success (0 failed)
GPU 1: 11992 / 11992 success (0 failed)
Total: 23984 / 23984
```

---

## 6. 启动脚本设计

### 6.1 `batch_infer.py` 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | `str` | `Shitao/OmniGen-v1` | 基础模型路径 |
| `--lora_path` | `str` | `None` | LoRA checkpoint 路径（可选） |
| `--jsonl_path` | `str` | **必须** | 测试集 JSONL 文件路径 |
| `--output_dir` | `str` | **必须** | 生成图像保存目录 |
| `--num_gpus` | `int` | `2` | 并行 GPU 数量 |
| `--guidance_scale` | `float` | `2.5` | CFG 引导强度 |
| `--img_guidance_scale` | `float` | `2.0` | 图像引导强度 |
| `--num_inference_steps` | `int` | `50` | 去噪步数 |
| `--seed` | `int` | `42` | 随机种子 |

### 6.2 Shell 启动脚本 `lanuch/infer.sh`

```bash
#!/bin/bash
export HF_HOME="/raid/home/CAMCA/hj880/wt/ckpts/huggingface"
export CUDA_VISIBLE_DEVICES=2,3

python3 batch_infer.py \
    --model_path Shitao/OmniGen-v1 \
    --lora_path ./results/cxr_finetune_lora/checkpoints/0000500 \
    --jsonl_path /raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno/cxr_synth_anno_test.jsonl \
    --output_dir /raid/home/CAMCA/hj880/wt/output/cxr_synth_results \
    --num_gpus 2 \
    --guidance_scale 2.5 \
    --img_guidance_scale 2.0 \
    --num_inference_steps 50 \
    --seed 42
```

---

## 7. 性能估算

| 指标 | 估算值 |
|------|--------|
| 单条推理时间（256x256, 50 steps, H100） | ~2-4 秒 |
| 单卡吞吐量 | ~900-1800 条/小时 |
| 双卡总吞吐量 | ~1800-3600 条/小时 |
| 全部 23,984 条预计耗时 | **~7-14 小时** |
| 单卡显存占用（模型 + VAE + KV Cache） | ~20-30 GB |

> 注：以上为保守估计。256x256 分辨率下实际速度可能更快。

---

## 8. 文件清单

完成开发后，需要新增/修改的文件列表：

| 文件 | 操作 | 说明 |
|------|------|------|
| `gen_code/batch_infer.py` | **新增** | 批量推理主脚本 |
| `gen_code/lanuch/infer.sh` | **新增** | 启动 shell 脚本 |
| `gen_code/docs/batch_inference_design.md` | **新增** | 本设计文档 |

---

## 9. 完整 `batch_infer.py` 伪代码概览

```python
"""batch_infer.py — CXR Multi-View OmniGen Batch Inference (2-GPU Data Parallel)"""

import os, json, math, argparse, logging
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from OmniGen import OmniGenPipeline


def derive_save_path(gt_path: str, output_dir: str) -> str:
    parts = gt_path.replace("\\", "/").split("/")
    return os.path.join(output_dir, parts[-2], parts[-1])


def worker_fn(rank, num_gpus, chunks, args):
    device = torch.device(f"cuda:{rank}")
    
    # 1. 加载模型
    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    if args.lora_path:
        pipe.merge_lora(args.lora_path)
    pipe.to(device)
    
    my_chunk = chunks[rank]
    success, fail = 0, 0
    
    # 2. 推理循环
    for sample in tqdm(my_chunk, desc=f"[GPU {rank}]", position=rank, leave=True):
        try:
            images = pipe(
                prompt=sample["instruction"],
                input_images=sample["input_images"],
                height=256, width=256,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                img_guidance_scale=args.img_guidance_scale,
                use_input_image_size_as_output=True,
                offload_model=False,
                use_kv_cache=True,
                offload_kv_cache=False,
                separate_cfg_infer=False,
                seed=args.seed,
            )
            save_path = derive_save_path(sample["output_image"], args.output_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            images[0].save(save_path)
            success += 1
        except Exception as e:
            logging.error(f"[GPU {rank}] Error: {e} | sample: {sample.get('output_image','?')}")
            fail += 1
            torch.cuda.empty_cache()
    
    # 3. 保存统计
    stats = {"rank": rank, "total": len(my_chunk), "success": success, "fail": fail}
    stats_path = os.path.join(args.output_dir, f"stats_gpu{rank}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[GPU {rank}] Done: {success}/{len(my_chunk)} success, {fail} failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--img_guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取完整 JSONL
    with open(args.jsonl_path, "r") as f:
        all_samples = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(all_samples)} samples from {args.jsonl_path}")
    
    # 均匀分片
    chunk_size = math.ceil(len(all_samples) / args.num_gpus)
    chunks = [
        all_samples[i * chunk_size : (i + 1) * chunk_size]
        for i in range(args.num_gpus)
    ]
    for i, c in enumerate(chunks):
        print(f"  GPU {i}: {len(c)} samples")
    
    # 启动多进程
    mp.spawn(worker_fn, args=(args.num_gpus, chunks, args), nprocs=args.num_gpus, join=True)
    
    # 汇总统计
    print("\n=== Inference Complete ===")
    total_success, total_fail = 0, 0
    for i in range(args.num_gpus):
        stats_path = os.path.join(args.output_dir, f"stats_gpu{i}.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            print(f"  GPU {s['rank']}: {s['success']}/{s['total']} success, {s['fail']} failed")
            total_success += s["success"]
            total_fail += s["fail"]
    print(f"  Total: {total_success}/{total_success + total_fail} success, {total_fail} failed")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

---

*End of Design Document*
