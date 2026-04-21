import os
import json
import math
import numpy as np

def generate_omnigen_jsonl(
    data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
    output_dir="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno",
    CT_thickness="thin",  # "all", "thin", or "thick"
    orientation="PA",
    split="train"         # "train" 或 "test"
):
    """
    为 OmniGen 生成训练集或测试集的 JSONL 文件。
    修复了原版中 // 2 导致的数据减半问题
    """
    assert split in["train", "test"], "split 必须是 'train' 或 'test'"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 安全读取并切分病患列表
    if os.path.exists(os.path.join(data_root, "patients.json")):
        with open(os.path.join(data_root, "patients.json")) as f:
            patients_dict = json.load(f)
            if CT_thickness == "thin":
                full_list = patients_dict["thin"]
            elif CT_thickness == "thick":
                full_list = patients_dict["thick"]
            else:  # all
                full_list = patients_dict["thin"] + patients_dict["thick"]
    else:
        # 修复：必须强制排序，否则 os.listdir 的随机性会导致 Train/Test 混合泄漏
        full_list =[
            folder for folder in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, folder))
        ]
        full_list.sort()
    
    # 根据 split 参数切分数据集
    if split == "train":
        patient_list = full_list[:-16]
        output_filename = "cxr_synth_anno_train.jsonl"
    else:
        patient_list = full_list[-16:]
        output_filename = "cxr_synth_anno_test.jsonl"
        
    print(f"=== Running in {split.upper()} mode ===")
    print(f"Number of patients in this split: {len(patient_list)}")
    
    # 2. 读取 camera_views.json 并建立 ID -> Coordinate 字典映射
    with open(os.path.join(data_root, "camera_views.json")) as f:
        camera_views = json.load(f)
        
    if orientation != "all":
        views_dict = {
            view["id"]: view["coordinate"]
            for view in camera_views
            if view["orientation"] == orientation
        }
    else:
        views_dict = {view["id"]: view["coordinate"] for view in camera_views}
        
    print(f"Number of valid views ({orientation}) per patient: {len(views_dict)}")
    
    output_json_path = os.path.join(output_dir, output_filename)
    image_data_list =[]
    missing_files_count = 0
    
    # 3. 遍历病患并生成指令
    for idx, pid in enumerate(patient_list):
        patient_path = os.path.join(data_root, pid)
        
        # 设定 condition
        cond_id = "0000"
        cond_img_path = os.path.join(patient_path, f"{cond_id}.png")
        
        # 如果 condition 图像不存在，或者 "0000" 不在当前筛选的 orientation 字典中，则跳过
        if not os.path.exists(cond_img_path) or cond_id not in views_dict:
            print(f"Warning: Missing condition image or invalid ID for patient {pid}, skipping entirely.")
            continue

        theta_cond, azimuth_cond = views_dict[cond_id][0], views_dict[cond_id][1]

        # 遍历字典中所有合法的 target 视图
        for target_id, coords in views_dict.items():
            if target_id == cond_id:
                continue
                
            target_img_path = os.path.join(patient_path, f"{target_id}.png")
            if not os.path.exists(target_img_path):
                missing_files_count += 1
                continue

            theta_target, azimuth_target = coords[0], coords[1]

            # 严格保持 Cond - Target 逻辑
            d_theta = theta_cond - theta_target
            d_azimuth = (azimuth_cond - azimuth_target) % (2 * math.pi)

            sin_d_azimuth = math.sin(d_azimuth)
            cos_d_azimuth = math.cos(d_azimuth)

            # 构造 OmniGen 指令
            instruction = (
                f"<img><|image_1|></img> Edit the view using delta pose: "
                f"d_theta={d_theta:.4f} rad, "
                f"sin(d_azimuth)={sin_d_azimuth:.4f}, "
                f"cos(d_azimuth)={cos_d_azimuth:.4f}."
            )

            jsonl_entry = {
                "task_type": "image_edit",
                "instruction": instruction,
                "input_images": [cond_img_path],
                "output_image": target_img_path,
            }

            image_data_list.append(jsonl_entry)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(patient_list):
            print(f"Processed {idx + 1}/{len(patient_list)} patients")
    
    print(f"Writing JSONL file to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        for entry in image_data_list:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully generated {split.upper()} JSONL file with {len(image_data_list)} entries.")
    if missing_files_count > 0:
        print(f"Note: {missing_files_count} target images were missing from disk and safely skipped.")
    print("-" * 50)


if __name__ == "__main__":
    generate_omnigen_jsonl(
        data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
        output_dir="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno",
        CT_thickness="thin",
        orientation="PA",
        split="train"
    )
    
    generate_omnigen_jsonl(
        data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
        output_dir="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno",
        CT_thickness="thin",
        orientation="PA",
        split="test"
    )