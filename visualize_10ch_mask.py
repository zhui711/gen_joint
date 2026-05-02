import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def visualize_10ch_mask(npz_path, output_path=None):
    if not os.path.exists(npz_path):
        print(f"[Error] File not found: {npz_path}")
        return

    try:
        data = np.load(npz_path)
        mask = data['mask']
        print(f"[Success] Loaded mask with shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"          Unique values in data: {np.unique(mask)}")
        # 统计每个像素平均有多少个通道是1
        overlap_mean = np.sum(mask, axis=0).mean()
        print(f"          Average overlapping channels per pixel: {overlap_mean:.2f}")
    except Exception as e:
        print(f"[Error] Failed to load {npz_path}: {e}")
        return

    if mask.ndim == 3 and mask.shape[0] != 10 and mask.shape[-1] == 10:
        mask = np.transpose(mask, (2, 0, 1))

    # 生成一个 10 种不同颜色的预设 (用 tab10 显色)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 画布：左边是一整张叠加彩图，右边是 2x5 的单通道拆解
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 8)
    
    # 融合图 (占用左边 2x3 的区域)
    ax_combined = fig.add_subplot(gs[:, 0:3])
    H, W = mask.shape[1:]
    combined_img = np.zeros((H, W, 3))
    
    for i in range(10):
        # 提取各个通道的 RGB，叠加到总图上
        for c in range(3):
            combined_img[:, :, c] += mask[i] * colors[i][c]
            
    # 因为存在重叠 (大部分区域加起来大于1)，我们将高于1的像素截断到 1.0 以免过曝溢出
    combined_img = np.clip(combined_img, 0, 1)
    
    ax_combined.imshow(combined_img)
    ax_combined.set_title(f"Combined Multi-class RGB View\n(Overlap Avg: {overlap_mean:.2f} ch/px)", fontsize=16)
    ax_combined.axis('off')

    # 添加自定义图例说明通道颜色
    patches = [plt.plot([],[], marker="s", ms=10, ls="", mec=None, color=colors[i][:3], 
            label=f"Channel {i}")[0] for i in range(10)]
    ax_combined.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # 单通道拆解图 (占用右边 2x5 区域)
    for i in range(10):
        row = i // 5
        col = (i % 5) + 3
        ax = fig.add_subplot(gs[row, col])
        if i < mask.shape[0]:
            # 为了更好的展示，这里的数值如果是只有 0和1，用普通的gray或者单一色图就可以。
            # magma因为背景是黑色，前景是黄色，对比比较强烈。由于数据只有 0和1，不会存在过渡色。
            ax.imshow(mask[i], cmap='magma', vmin=0, vmax=1)
            ax.set_title(f"CH {i} (Mean: {mask[i].mean():.2f})", fontsize=12)
        ax.axis('off')

    plt.tight_layout()

    if output_path is None:
        output_path = npz_path.replace('.npz', '_vis.png')
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[Done] Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 10-channel .npz medical mask.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz mask file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output PNG image")
    args = parser.parse_args()

    visualize_10ch_mask(args.npz_path, args.output_path)
