"""
JSONL Generator with Anatomy Masks.

Duplicates gen_train_test_jsonl.py but appends "output_mask" pointing
to the corresponding .npz pseudo-mask file. Raises an error if the
mask file does not exist.

Usage:
    python gen_mask_jsonl.py \
        --data_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256 \
        --mask_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch \
        --output_dir /home/wenting/zr/wt_dataset/LIDC_IDRI/anno \
        --orientation PA --split train
"""

import os
import json
import math
import argparse


def image_path_to_mask_path(image_path, image_root, mask_root, suffix=".npz"):
    """Convert an image path to the corresponding mask path."""
    rel = os.path.relpath(image_path, image_root)
    base, _ = os.path.splitext(rel)
    return os.path.join(mask_root, base + suffix)


def generate_omnigen_jsonl_with_masks(
    data_root,
    mask_root,
    output_dir,
    CT_thickness="thin",
    orientation="PA",
    split="train",
):
    assert split in ["train", "test"], "split must be 'train' or 'test'"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read and split the patient list
    patients_json = os.path.join(data_root, "patients.json")
    if os.path.exists(patients_json):
        with open(patients_json) as f:
            patients_dict = json.load(f)
            if CT_thickness == "thin":
                full_list = patients_dict["thin"]
            elif CT_thickness == "thick":
                full_list = patients_dict["thick"]
            else:
                full_list = patients_dict["thin"] + patients_dict["thick"]
    else:
        full_list = [
            folder
            for folder in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, folder))
        ]
        full_list.sort()

    if split == "train":
        patient_list = full_list[:-16]
        output_filename = "cxr_synth_anno_mask_train.jsonl"
    else:
        patient_list = full_list[-16:]
        output_filename = "cxr_synth_anno_mask_test.jsonl"

    print(f"=== Running in {split.upper()} mode ===")
    print(f"Number of patients in this split: {len(patient_list)}")

    # 2. Read camera_views.json
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
    image_data_list = []
    missing_files_count = 0

    # 3. Iterate over patients
    for idx, pid in enumerate(patient_list):
        patient_path = os.path.join(data_root, pid)

        cond_id = "0000"
        cond_img_path = os.path.join(patient_path, f"{cond_id}.png")

        if not os.path.exists(cond_img_path) or cond_id not in views_dict:
            print(f"Warning: Missing condition image or invalid ID for patient {pid}, skipping.")
            continue

        theta_cond, azimuth_cond = views_dict[cond_id][0], views_dict[cond_id][1]

        for target_id, coords in views_dict.items():
            if target_id == cond_id:
                continue

            target_img_path = os.path.join(patient_path, f"{target_id}.png")
            if not os.path.exists(target_img_path):
                missing_files_count += 1
                continue

            # Hard check: mask must exist
            mask_path = image_path_to_mask_path(target_img_path, data_root, mask_root)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file not found for target image!\n"
                    f"  Image: {target_img_path}\n"
                    f"  Expected mask: {mask_path}\n"
                    f"Run generate_pseudo_masks.py first."
                )

            theta_target, azimuth_target = coords[0], coords[1]
            d_theta = theta_cond - theta_target
            d_azimuth = (azimuth_cond - azimuth_target) % (2 * math.pi)
            sin_d_azimuth = math.sin(d_azimuth)
            cos_d_azimuth = math.cos(d_azimuth)

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
                "output_mask": mask_path,
            }
            image_data_list.append(jsonl_entry)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(patient_list):
            print(f"Processed {idx + 1}/{len(patient_list)} patients")

    print(f"Writing JSONL file to {output_json_path}...")
    with open(output_json_path, "w") as f:
        for entry in image_data_list:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully generated {split.upper()} JSONL with {len(image_data_list)} entries.")
    if missing_files_count > 0:
        print(f"Note: {missing_files_count} target images were missing from disk and safely skipped.")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSONL with anatomy mask paths")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno",
    )
    parser.add_argument("--CT_thickness", type=str, default="thin")
    parser.add_argument("--orientation", type=str, default="PA")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    generate_omnigen_jsonl_with_masks(
        data_root=args.data_root,
        mask_root=args.mask_root,
        output_dir=args.output_dir,
        CT_thickness=args.CT_thickness,
        orientation=args.orientation,
        split=args.split,
    )
