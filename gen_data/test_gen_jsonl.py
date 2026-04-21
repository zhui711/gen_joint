import os
import json
import math
import numpy as np

def check_double_filter_bug(
    data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
    CT_thickness="thin",  
    orientation="PA",
):
    print("="*50)
    print("🚀 开启代码沙盒推演（Experimental Check）")
    print(f"设定条件: orientation = '{orientation}'")
    print("="*50)
    
    # 1. 模拟读取病患列表（仅取前几个用于测试验证）
    if os.path.exists(os.path.join(data_root, "patients.json")):
        with open(os.path.join(data_root, "patients.json")) as f:
            patients_dict = json.load(f)
            patient_list = patients_dict.get(CT_thickness,[])
    else:
        patient_list = ["test_patient_001"] # Fallback

    patient_list = patient_list[:-16]
    if not patient_list:
        print("未找到病患数据，请检查 data_root 路径。")
        return

    # 2. 读取 camera_views.json 并进行争议代码段的测试
    camera_views_path = os.path.join(data_root, "camera_views.json")
    if not os.path.exists(camera_views_path):
        print(f"找不到 {camera_views_path}，无法进行测试。")
        return

    with open(camera_views_path) as f:
        camera_views = json.load(f)
        print(f"\n[探针 1] JSON 文件中总共有 {len(camera_views)} 个视角记录。")
        
        # 【您原始代码中的过滤逻辑】
        if orientation != "all":
            views = np.array([
                view["coordinate"]
                for view in camera_views
                if view["orientation"] == orientation
            ])
            print(f"[探针 2] 经过 `if view['orientation'] == '{orientation}'` 过滤后，")
            print(f"         views 数组的长度变为了: {len(views)} ！！！")
        else:
            views = np.array([view["coordinate"] for view in camera_views])
    
    # 【核心争议逻辑：变量赋值追踪】
    total_view = len(views)
    print(f"\n[探针 3] 执行 `total_view = len(views)` 后，")
    print(f"         当前 total_view 的值为: {total_view}")
    
    total_view = total_view if orientation == "all" else total_view // 2
    print(f"[探针 4] 执行 `total_view = total_view // 2` 后，")
    print(f"         最终 total_view 的值被截断为: {total_view} ！！！")

    # 3. 模拟进入循环，看看实际遍历了哪些图片
    print("\n" + "="*50)
    print(f"🔍 模拟处理第一个病患 ({patient_list[0]}) 的循环遍历")
    print("="*50)
    
    processed_targets =[]
    # 【您原始代码中的循环逻辑】
    for index_target in range(1, total_view):
        processed_targets.append(index_target)
        
    if processed_targets:
        print(f"实际参与训练的 target_image 索引范围是: {processed_targets[0]:04d}.png 到 {processed_targets[-1]:04d}.png")
        print(f"该病患总共生成了 {len(processed_targets)} 条 JSONL 数据。")
        
        # 计算理论上应该有多少条数据
        expected_max = len(views) - 1  # 减去 0000.png 作为 condition
        if processed_targets[-1] < expected_max:
            print(f"\n🚨 [结论：发现数据丢失]")
            print(f"预期应该处理到 {expected_max:04d}.png，但由于 // 2 逻辑，循环在 {processed_targets[-1]:04d}.png 就提前终止了！")
            print(f"丢失的图片范围是: {(processed_targets[-1] + 1):04d}.png 到 {expected_max:04d}.png")
    else:
        print("没有处理任何图片。")

    print("\n=== 推演结束，未保存任何文件 ===")


if __name__ == "__main__":
    check_double_filter_bug(
        data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
        CT_thickness="thin",
        orientation="PA",
    )