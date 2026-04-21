import os
import json
import math
import numpy as np


def generate_jsonl_from_dataset(
    data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
    output_dir="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno",
    CT_thickness="thin",  # "all", "thin", or "thick"
    orientation="PA",
):
    """
    Generate JSONL file from LidcidriHardFBData dataset.
    Only generates paths, does not load actual image data.
    
    Args:
        data_root: Path to the dataset root directory
        output_dir: Path to save the output JSONL file
        CT_thickness: "all", "thin", or "thick"
        orientation: "all", "PA", or "AP"
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read patient list
    if os.path.exists(os.path.join(data_root, "patients.json")):
        with open(os.path.join(data_root, "patients.json")) as f:
            patients_dict = json.load(f)
            if CT_thickness == "thin":
                patient_list = patients_dict["thin"]
            elif CT_thickness == "thick":
                patient_list = patients_dict["thick"]
            else:  # all
                patient_list = patients_dict["thin"] + patients_dict["thick"]
    else:
        # include all folders
        patient_list = []
        for folder in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, folder)):
                patient_list.append(folder)
    
    # Use all except last 16 patients as training set
    patient_list = patient_list[:-16]
    
    # Read camera views
    with open(os.path.join(data_root, "camera_views.json")) as f:
        camera_views = json.load(f)
        if orientation != "all":
            views = np.array([
                view["coordinate"]
                for view in camera_views
                if view["orientation"] == orientation
            ])
        else:
            views = np.array([view["coordinate"] for view in camera_views])
    
    total_view = len(views)
    total_view = total_view if orientation == "all" else total_view // 2
    
    print(f"Number of patients: {len(patient_list)}")
    print(f"Number of views per patient: {total_view}")
    print(f"Generating JSONL file...")
    
    # Output file path
    output_json_path = os.path.join(output_dir, "cxr_synth_anno.jsonl")
    
    # Store image information
    image_data_list = []
    
    # Process each patient
    entry_count = 0
    for idx, pid in enumerate(patient_list):
        patient_path = os.path.join(data_root, pid)
        
        # Use view 0 as condition
        index_cond = 0
        
        # Check if condition image exists
        cond_img_path = os.path.join(patient_path, f"{index_cond:04d}.png")
        if not os.path.exists(cond_img_path):
            print(f"Warning: Missing condition image for patient {pid}, skipping...")
            continue

        # Calculate delta pose (same as LidcidriHardFBData.get_T)
        theta_cond, azimuth_cond = views[index_cond][0], views[index_cond][1]

        for index_target in range(1, total_view):
            target_img_path = os.path.join(patient_path, f"{index_target:04d}.png")
            if not os.path.exists(target_img_path):
                print(
                    f"Warning: Missing target image for patient {pid}, view {index_target}, skipping..."
                )
                continue

            theta_target, azimuth_target = views[index_target][0], views[index_target][1]

            # Use cond - target (matching the dataset implementation)
            d_theta = theta_cond - theta_target
            d_azimuth = (azimuth_cond - azimuth_target) % (2 * math.pi)

            sin_d_azimuth = math.sin(d_azimuth)
            cos_d_azimuth = math.cos(d_azimuth)

            # Create instruction with angle information
            instruction = (
                f"<img><|image_1|></img> Edit the view using delta pose: "
                f"d_theta={d_theta:.4f} rad, "
                f"sin(d_azimuth)={sin_d_azimuth:.4f}, "
                f"cos(d_azimuth)={cos_d_azimuth:.4f}."
            )

            # Create JSONL entry with absolute paths
            jsonl_entry = {
                "task_type": "image_edit",
                "instruction": instruction,
                "input_images": [cond_img_path],
                "output_image": target_img_path,
            }

            image_data_list.append(jsonl_entry)
            entry_count += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(patient_list)} patients")
    
    # Write JSONL file
    print(f"Writing JSONL file to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        for entry in image_data_list:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully generated JSONL file with {len(image_data_list)} entries")
    print(f"Output file: {output_json_path}")


if __name__ == "__main__":
    generate_jsonl_from_dataset(
        data_root="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/img_complex_fb_256",
        output_dir="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno",
        CT_thickness="thin",
        orientation="PA",
    )
