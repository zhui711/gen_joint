# CXR Multi-View Generation: Comprehensive Inference & Evaluation Script Design

> **Script Name**: `test_omnigen_cxr.py`  
> **Location**: `gen_code/`  
> **Purpose**: High-throughput batch inference of OmniGen models on CXR data, followed by automatic evaluation of SSIM, PSNR, LPIPS, and FID.

---

## 1. Overall Architecture

The script operates in a **Map-Reduce** style architecture to maximize GPU utilization on a multi-card node (specifically 2x H100).

```mermaid
graph TD
    Main[Main Process] -->|1. Load & Split JSONL| SplitData
    SplitData -->|Chunk 0| Worker0[Worker 0 (cuda:0)]
    SplitData -->|Chunk 1| Worker1[Worker 1 (cuda:1)]
    
    subgraph Generation Phase
        Worker0 -->|Load Model + LoRA| W0_Init
        Worker1 -->|Load Model + LoRA| W1_Init
        
        W0_Init --> W0_BatchLoop[Batch Inference Loop]
        W1_Init --> W1_BatchLoop[Batch Inference Loop]
        
        W0_BatchLoop -->|Save Images| Disk[Output Directory]
        W1_BatchLoop -->|Save Images| Disk
    end
    
    Worker0 -->|Signal Done| Sync[Synchronization Barrier]
    Worker1 -->|Signal Done| Sync
    
    Sync -->|2. Trigger Eval| EvalProcess[Main Process / Eval Function]
    
    subgraph Evaluation Phase
        EvalProcess -->|Load Generated & GT| MerticCalc[Calculate Metrics]
        MerticCalc -->|SSIM, PSNR, LPIPS| Pairwise[Pairwise Metrics]
        MerticCalc -->|FID| Distrib[Distribution Metrics]
        
        Pairwise --> Report[Print & Save Report]
        Distrib --> Report
    end
```

### 1.1 Lifecycle
1.  **Preparation**: Parse arguments, load the test JSONL, and optionally slice it (Mini-Test).
2.  **Dispatch**: Split the (possibly sliced) dataset into `N` chunks for `N` GPUs.
3.  **Generation (Parallel)**: Launch worker processes. Each worker performs batch inference and saves images to a structured directory.
4.  **Synchronization**: The main process waits for all workers to join.
5.  **Evaluation (Serial)**: The main process scans the output directory and the original dataset to compute metrics.

---

## 2. Key Features Design

### 2.1 Mini-Test & Checkpoint Selection
-   **Checkpoint**: Via `--lora_path`.
    -   Usage: `pipe.merge_lora(lora_path)` inside the worker.
    -   Allows testing specific epochs (e.g., `checkpoints/0007500`).
-   **Sampling (`--max_samples`)**:
    -   Logic: Before splitting data for workers, the main process slices the list: `data = data[:args.max_samples]`.
    -   Benefit: Run a sanity check on 16 or 100 images in <1 minute before launching a full 7-hour run.

### 2.2 Secure Batch Inference (Anti-Shuffle)
To prevent data misalignment in batch processing (crucial for medical imaging where `view_001` must not be confused with `view_002`):

1.  **Input Assembly**: Construct strict parallel lists for `prompts` and `input_images`. Keep reference to the `metadata` (filenames) in a corresponding list.
2.  **Atomic Inference**: Call `pipe()` once per batch.
3.  **Strict Zipping**: Iterate over `zip(generated_images, batch_metadata)` immediately after inference to bind the image content to its intended filename.

```python
# Pseudo-code for safety
batch_prompts = [item['instruction'] for item in batch]
batch_inputs = [item['input_images'] for item in batch] # List[List[str]]
batch_paths = [item['output_image'] for item in batch]   # Ground Truth paths for reference

# Inference
images = pipe(prompt=batch_prompts, input_images=batch_inputs, ...)

# Strict binding
for img, gt_path in zip(images, batch_paths):
    save_path = derive_save_path(gt_path, log_dir)
    img.save(save_path)
```

### 2.3 Evaluation Integration
We will use `torchmetrics` for a standardized, robust implementation of all metrics.
-   **Pairwise (SSIM, PSNR, LPIPS)**:
    -   Iterate over the **generated** directory.
    -   Infer the **GT** path from the filename (dataset structure is known/fixed).
    -   Accumulate scores and average.
-   **Distribution (FID)**:
    -   Requires two distinct directories.
    -   Feature extraction using InceptionV3 (via `torchmetrics.image.fid.FrechetInceptionDistance`).
    -   **Optimization**: Since real images are scattered in patient folders (`.../img_complex_fb_256/{patient}/{view}.png`), we may need to either:
        a) Copy/Symlink GT images to a temp flat folder (slow).
        b) Use a custom dataset class that yields batches of (real, fake) to update the metric incrementally. **Option (b) is preferred for memory efficiency.**

---

## 3. Detailed Parameter Definition

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model_path` | `Shitao/OmniGen-v1` | Base model path. |
| `--lora_path` | `None` | **Required** for testing fine-tuned models. Path to adapter folder. |
| `--jsonl_path` | (Required) | Path to `cxr_synth_anno_test.jsonl`. |
| `--output_dir` | `./evaluation_results/` | Root dir for saving images. |
| `--batch_size` | `8` | Inference batch size per GPU. |
| `--num_gpus` | `2` | Number of GPUs to use. |
| `--max_samples` | `None` | If set (e.g., 100), only process first N samples. |
| `--skip_eval` | `False` | If True, only generate images, do not run metrics. |
| `--seed` | `42` | Random seed. |
| `--inference_steps` | `50` | Diffusion denoising steps. |
| `--guidance_scale` | `2.5` | CFG scale. |
| `--img_guidance_scale` | `2.0` | Image guidance scale. |

---

## 4. Implementation Logic (Pseudo-code)

### Part 1: Worker Function (Generation)

```python
def worker(rank, args, data_chunk):
    # 1. Setup Device & Model
    device = f"cuda:{rank}"
    pipe = OmniGenPipeline.from_pretrained(args.model_path)
    if args.lora_path:
        pipe.merge_lora(args.lora_path)
    pipe.to(device)
    
    # 2. Batch Loop
    dataloader = create_dataloader(data_chunk, batch_size=args.batch_size)
    
    for batch in tqdm(dataloader, position=rank, desc=f"GPU {rank}"):
        # batch is a list of dicts
        prompts = [item['instruction'] for item in batch]
        input_imgs = [item['input_images'] for item in batch]
        
        # Inference
        outputs = pipe(
            prompt=prompts,
            input_images=input_imgs,
            height=256, width=256,
            use_input_image_size_as_output=True,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            img_guidance_scale=args.img_guidance_scale,
            separate_cfg_infer=False, # Optimization for speed
            offload_model=False
        )
        
        # Save securely
        for img, item in zip(outputs, batch):
            # Parse path: .../patient_id/view_id.png
            relative_path = get_relative_path(item['output_image'])
            save_full_path = os.path.join(args.output_dir, relative_path)
            
            os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
            img.save(save_full_path)
```

### Part 2: Evaluation Function

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance as FID

def evaluate(args):
    print("Starting Evaluation...")
    device = "cuda:0" # Use one GPU for eval
    
    # Metrics Init
    calc_ssim = SSIM().to(device)
    calc_psnr = PSNR().to(device)
    calc_lpips = LPIPS(net_type='vgg').to(device)
    calc_fid = FID(feature=2048).to(device)
    
    # Iterate over generated files
    files = list_all_files(args.output_dir)
    
    for gen_path in tqdm(files, desc="Evaluating"):
        # 1. Load Gen and GT
        gt_path = resolve_gt_path(gen_path, args.jsonl_path_root)
        
        img_gen = load_tensor(gen_path).to(device)
        img_gt = load_tensor(gt_path).to(device)
        
        # 2. Update Pairwise
        calc_ssim.update(img_gen, img_gt)
        calc_psnr.update(img_gen, img_gt)
        calc_lpips.update(img_gen, img_gt)
        
        # 3. Update FID (needs uint8 [0, 255])
        calc_fid.update(keys(img_gt), real=True)
        calc_fid.update(keys(img_gen), real=False)
        
    # Compute
    results = {
        "SSIM": calc_ssim.compute().item(),
        "PSNR": calc_psnr.compute().item(),
        "LPIPS": calc_lpips.compute().item(),
        "FID": calc_fid.compute().item()
    }
    
    print(json.dumps(results, indent=2))
    save_results(results, args.output_dir)
```

### Part 3: Main Entry Point

```python
if __name__ == "__main__":
    args = parse_args()
    
    # 1. Load Data
    all_data = load_jsonl(args.jsonl_path)
    if args.max_samples:
        all_data = all_data[:args.max_samples]
        
    # 2. Split for Multi-GPU
    chunks = split_data(all_data, args.num_gpus)
    
    # 3. Run Generation
    mp.spawn(worker, args=(args, chunks), nprocs=args.num_gpus)
    
    # 4. Run Evaluation
    if not args.skip_eval:
        evaluate(args)
```
