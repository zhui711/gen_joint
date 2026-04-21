import os
import json
import torch
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm

try:
    import lpips
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    print("Please install lpips and torchmetrics: pip install lpips torchmetrics[image]")

def get_image_stats(img_np, name="Image"):
    """打印图像的像素级统计特征"""
    min_val = img_np.min()
    max_val = img_np.max()
    mean_val = img_np.mean()
    std_val = img_np.std()
    print(f"[{name}] Min:{min_val:3d} | Max:{max_val:3d} | Mean:{mean_val:.2f} | Std:{std_val:.2f}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 记载 VAE
    print("Loading SDXL-VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device).eval()
    
    # 2. 从 Jsonl 获取测试样本 (前5个)
    test_jsonl = "/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno/cxr_synth_anno_test.jsonl"
    gen_dir = "/raid/home/CAMCA/hj880/wt/code/cxr_synth/gen_code/outputs/cxr_finetune_lora_8000"
    
    samples = []
    with open(test_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5: break
            samples.append(json.loads(line))

    # OmniGen的数据预处理过程（严格对照代码）
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)), # 假设测评时是256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 评估指标初始化
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    gt_fid_buf, recon_fid_buf = [], []
    lpips_scores = []
    
    out_dir = "./vae_sanity_check_outputs"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            gt_path = sample['output_image']
            
            # --- 读取 GT ---
            gt_pil = Image.open(gt_path).convert("RGB")
            gt_pil = gt_pil.resize((256, 256), Image.BICUBIC)
            gt_np = np.array(gt_pil)
            get_image_stats(gt_np, "Ground Truth")
            
            # ---读取 Generated (OmniGen模型生成的预测图) ---
            rel_path = "/".join(gt_path.replace("\\", "/").split("/")[-2:])
            gen_path = os.path.join(gen_dir, rel_path)
            if os.path.exists(gen_path):
                gen_np = np.array(Image.open(gen_path).convert("RGB"))
                get_image_stats(gen_np, "OmniGen Gen ")
            else:
                print(f"[OmniGen Gen ] NOT FOUND for comparison: {gen_path}")

            # --- VAE Encode -> Decode 重构过程 ---
            input_tensor = preprocess(gt_pil).unsqueeze(0).to(device) # [1, 3, 256, 256], [-1, 1]
            
            # Scaling factor processing like OmniGen/pipeline.py
            latent = vae.encode(input_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
            
            # Decode
            recon_tensor = latent / vae.config.scaling_factor
            recon_tensor = vae.decode(recon_tensor).sample
            
            # 缩放回 [0, 1] 然后变 [0, 255] uint8
            recon_tensor = (recon_tensor * 0.5 + 0.5).clamp(0, 1)
            recon_samples = (recon_tensor * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            recon_np = recon_samples[0]
            
            get_image_stats(recon_np, "VAE Recon   ")
            
            # 保存对比图
            recon_pil = Image.fromarray(recon_np)
            recon_pil.save(os.path.join(out_dir, f"recon_{i}.png"))
            gt_pil.save(os.path.join(out_dir, f"gt_{i}.png"))

            # --- 累加计算 Metrics (参照 test_omnigen_cxr 的做法) ---
            # LPIPS: [-1, 1] RGB
            gt_lpips = torch.from_numpy(gt_np).permute(2,0,1).float().unsqueeze(0).to(device) / 127.5 - 1.0
            recon_lpips = torch.from_numpy(recon_np).permute(2,0,1).float().unsqueeze(0).to(device) / 127.5 - 1.0
            score = lpips_fn(recon_lpips, gt_lpips)
            lpips_scores.append(score.item())
            
            # FID: [0, 255] uint8 RGB
            gt_fid_buf.append(torch.from_numpy(gt_np).permute(2,0,1))
            recon_fid_buf.append(torch.from_numpy(recon_np).permute(2,0,1))
            
    # 计算均值 Metrics
    fid_metric.update(torch.stack(gt_fid_buf).to(device), real=True)
    fid_metric.update(torch.stack(recon_fid_buf).to(device), real=False)
    
    print("\n" + "="*50)
    print("      VAE SANITY CHECK RESULTS")
    print("="*50)
    print(f"LPIPS (VAE Upper Bound): {np.mean(lpips_scores):.4f}")
    print(f"FID   (VAE Upper Bound): {fid_metric.compute().item():.2f}")
    print("="*50)
    print("If metric results differ dramatically from SV-DRR's 0.15 FID, the divergence is fundamentally caused by evaluating native SDXL VAE against a completely different DRR pipeline.")

if __name__ == "__main__":
    main()