import os
import glob
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


DATASET_DIR = "/data"
DEFAULT_SAVE_DIR = "./models/aug_preview"
DEFAULT_NUM_SAMPLES = 8


def build_transforms(target_width=1024, target_height=512):
    return A.Compose(
        [
            A.RandomCrop(width=target_width, height=target_height, p=1),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.3
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.15),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.15),
            # Color and contrast augmentations - enhanced for lighting variations
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.4),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            # Noise and blur augmentations
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.15),
            A.MotionBlur(blur_limit=4, p=0.15),
            # Weather and lighting effects
            A.RandomRain(
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.7,
                rain_type="drizzle",
                p=0.1,
            ),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.1),
            # Occlusion and masking
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.15,
            ),
        ]
    )


# Cityscapes trainIds palette (0..18); 255 = ignore
TRAINIDS_COLORS = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def map_labelids_to_trainids(mask_labelids):
    lid_to_tid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18,
    }
    out = np.full_like(mask_labelids, 255, dtype=np.uint8)
    for lid, tid in lid_to_tid.items():
        out[mask_labelids == lid] = tid
    return out


def colorize_trainids(mask_trainids):
    h, w = mask_trainids.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for tid in np.unique(mask_trainids):
        if tid == 255:
            out[mask_trainids == 255] = IGNORE_COLOR
        elif 0 <= tid < len(TRAINIDS_COLORS):
            out[mask_trainids == tid] = TRAINIDS_COLORS[tid]
    return out


def load_pair(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Try trainIds first, then labelIds
    p = Path(image_path)
    mask_name = p.name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    mask_path = Path(DATASET_DIR) / 'gtFine' / p.parent.parent.name / p.parent.name / mask_name
    loaded_labelids = False
    if not mask_path.exists():
        mask_name = p.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        mask_path = Path(DATASET_DIR) / 'gtFine' / p.parent.parent.name / p.parent.name / mask_name
        loaded_labelids = True

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load mask: {mask_path}")

    if loaded_labelids:
        mask = map_labelids_to_trainids(mask)

    return img, mask


def main():
    parser = argparse.ArgumentParser(description="Preview Cityscapes augmentations")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"]) 
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    train_glob = str(Path(DATASET_DIR) / 'leftImg8bit' / 'train' / '**' / '*_leftImg8bit.png')
    val_glob = str(Path(DATASET_DIR) / 'leftImg8bit' / 'val' / '**' / '*_leftImg8bit.png')
    image_paths = glob.glob(train_glob if args.split=="train" else val_glob, recursive=True)
    if not image_paths:
        print("No training images found.")
        return

    random.shuffle(image_paths)
    chosen = image_paths[:max(1, args.num_samples)]
    tfm = build_transforms()

    for idx, ip in enumerate(chosen, 1):
        try:
            img, mask = load_pair(ip)
            h, w = 512, 1024
            img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            aug = tfm(image=img_r, mask=mask_r)
            img_aug = aug['image']
            mask_aug = aug['mask']

            # Colorize
            col_orig = colorize_trainids(mask_r)
            col_aug = colorize_trainids(mask_aug)

            # Stack: [orig img | orig mask | aug img | aug mask]
            orig_img_u8 = img_r.astype(np.uint8)
            aug_img_u8 = img_aug.astype(np.uint8)
            panel = np.hstack([orig_img_u8, col_orig, aug_img_u8, col_aug])

            out_path = os.path.join(save_dir, f'preview_{idx:02d}.png')
            cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Skipping {ip}: {e}")


if __name__ == "__main__":
    main()
