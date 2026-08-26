import os
import numpy as np
import rasterio
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = r"D:\unet\38-Cloud-A-Cloud-Segmentation-Dataset-master\sample_tif"

VAL_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 4


# ============================================================
# 2. CUSTOM PYTORCH DATASET
# ============================================================

class CloudDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Read Red
        # ----------------------------------------------------

        with rasterio.open(sample["red"]) as src:
            red = src.read(1).astype(np.float32)

        # ----------------------------------------------------
        # Read Green
        # ----------------------------------------------------

        with rasterio.open(sample["green"]) as src:
            green = src.read(1).astype(np.float32)

        # ----------------------------------------------------
        # Read Blue
        # ----------------------------------------------------

        with rasterio.open(sample["blue"]) as src:
            blue = src.read(1).astype(np.float32)

        # ----------------------------------------------------
        # Read NIR
        # ----------------------------------------------------

        with rasterio.open(sample["nir"]) as src:
            nir = src.read(1).astype(np.float32)

        # ----------------------------------------------------
        # Normalize each band to 0-1
        # ----------------------------------------------------

        bands = [red, green, blue, nir]

        normalized_bands = []

        for band in bands:

            min_value = band.min()
            max_value = band.max()

            band = (
                band - min_value
            ) / (
                max_value - min_value + 1e-8
            )

            normalized_bands.append(band)

        # ----------------------------------------------------
        # Stack R, G, B, NIR
        # Shape = [4, H, W]
        # ----------------------------------------------------

        image = np.stack(
            normalized_bands,
            axis=0
        )

        # ----------------------------------------------------
        # Read Ground Truth Mask
        # ----------------------------------------------------

        with rasterio.open(sample["gt"]) as src:
            mask = src.read(1)

        # ----------------------------------------------------
        # Convert mask to binary
        # Cloud = 1
        # Non-cloud = 0
        # ----------------------------------------------------

        mask = (mask > 0).astype(np.float32)

        # Add channel dimension
        # [H, W] -> [1, H, W]

        mask = np.expand_dims(
            mask,
            axis=0
        )

        # ----------------------------------------------------
        # Convert NumPy -> PyTorch Tensor
        # ----------------------------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


# ============================================================
# 3. FIND TIF FILES
# ============================================================

red_files = []
green_files = []
blue_files = []
nir_files = []
gt_files = []


for filename in os.listdir(DATA_DIR):

    if not filename.lower().endswith(".tif"):
        continue

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    name = filename.lower()

    if name.startswith("red_patch"):
        red_files.append(filepath)

    elif name.startswith("green_patch"):
        green_files.append(filepath)

    elif name.startswith("blue_patch"):
        blue_files.append(filepath)

    elif name.startswith("nir_patch"):
        nir_files.append(filepath)

    elif name.startswith("gt_patch"):
        gt_files.append(filepath)


# Sort files
red_files.sort()
green_files.sort()
blue_files.sort()
nir_files.sort()
gt_files.sort()


# ============================================================
# 4. CHECK NUMBER OF FILES
# ============================================================

print("Red   :", len(red_files))
print("Green :", len(green_files))
print("Blue  :", len(blue_files))
print("NIR   :", len(nir_files))
print("GT    :", len(gt_files))


# ============================================================
# 5. CREATE SAMPLES
# ============================================================

number_of_samples = min(
    len(red_files),
    len(green_files),
    len(blue_files),
    len(nir_files),
    len(gt_files)
)


samples = []


for i in range(number_of_samples):

    samples.append({

        "red": red_files[i],

        "green": green_files[i],

        "blue": blue_files[i],

        "nir": nir_files[i],

        "gt": gt_files[i]
    })


print()
print("Total samples:", len(samples))


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

if len(samples) >= 2:

    train_samples, val_samples = train_test_split(

        samples,

        test_size=VAL_SIZE,

        random_state=RANDOM_STATE
    )

else:

    print()
    print("WARNING:")
    print("Only one sample is available.")
    print("80/20 train-validation split is skipped.")

    train_samples = samples

    val_samples = []


# ============================================================
# 7. CREATE DATASET
# ============================================================

train_dataset = CloudDataset(
    train_samples
)


if len(val_samples) > 0:

    val_dataset = CloudDataset(
        val_samples
    )

else:

    val_dataset = None


# ============================================================
# 8. CREATE DATALOADER
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)


if val_dataset is not None:

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )


# ============================================================
# 9. TEST DATASET
# ============================================================

image, mask = train_dataset[0]


print()
print("=" * 50)
print("DATASET TEST")
print("=" * 50)

print(
    "Image shape:",
    image.shape
)

print(
    "Mask shape :",
    mask.shape
)

print(
    "Image dtype:",
    image.dtype
)

print(
    "Mask dtype :",
    mask.dtype
)

print(
    "Image min  :",
    image.min().item()
)

print(
    "Image max  :",
    image.max().item()
)

print(
    "Mask values:",
    torch.unique(mask).tolist()
)


# ============================================================
# 10. TEST DATALOADER
# ============================================================

images, masks = next(
    iter(train_loader)
)


print()
print("=" * 50)
print("DATALOADER TEST")
print("=" * 50)

print(
    "Batch image shape:",
    images.shape
)

print(
    "Batch mask shape :",
    masks.shape
)

print("=" * 50)