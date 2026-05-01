import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_10ch_mask(npz_path, output_path=None):
    # 1. 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"[Error] File not found: {npz_path}")
        return

    # 2. 读取 .npz 文件
    try:
        data = np.load(npz_path)
        if 'mask' not in data:
            print(f"[Error] 'mask' key not found in {npz_path}.")
            print(f"Available keys: {list(data.keys())}")
            return
        
        mask = data['mask']
        print(f"[Success] Loaded mask with shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
    except Exception as e:
        print(f"[Error] Failed to load {npz_path}: {e}")
        return

    # 3. 确保形状是 (10, H, W)
    if mask.shape[0] != 10:
        if mask.shape[-1] == 10:
            # 如果形状是 (H, W, 10)，转置为 (10, H, W)
            mask = np.transpose(mask, (2, 0, 1))
        else:
            print(f"[Warning] Expected 10 channels, but got shape {mask.shape}. Will visualize up to 10 channels.")

    # 4. 创建画布 (2行5列)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"10-Channel Anatomy Mask Breakdown\n{os.path.basename(npz_path)}", fontsize=16, fontweight='bold')

    axes = axes.flatten()
    
    # 5. 遍历每个通道并绘图
    for i in range(10):
        ax = axes[i]
        if i < mask.shape[0]:
            # 提取单通道
            channel_data = mask[i]
            
            # 使用 'magma' 或 'viridis' colormap 使得有解剖结构的地方变亮，背景变暗
            cax = ax.imshow(channel_data, cmap='magma', vmin=0, vmax=1)
            ax.set_title(f"Channel {i}", fontsize=14)
        else:
            # 如果通道数不足10，留白
            ax.set_title(f"Channel {i} (Empty)")
        
        ax.axis('off') # 隐藏坐标轴

    plt.tight_layout()

    # 6. 保存可视化结果
    if output_path is None:
        # 如果没有指定输出路径，默认保存在原文件同目录下，后缀为 _vis.png
        output_path = npz_path.replace('.npz', '_vis.png')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[Done] Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 10-channel .npz medical mask.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz mask file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output PNG image (optional)")
    args = parser.parse_args()

    visualize_10ch_mask(args.npz_path, args.output_path)