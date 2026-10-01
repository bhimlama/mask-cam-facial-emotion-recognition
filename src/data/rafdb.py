"""
RAF-DB dataset loader.

RAF-DB (Real-world Affective Faces Database) contains ~30,000 facial images
from thousands of individuals. This repo uses only the basic expression
subset: 12,271 training images and 3,068 test images across 7 emotions.

The extracted zip has folders numbered 1-7, which need to be renamed to
match the FER2013 alphabetical order used throughout this repo:
    [angry, disgust, fear, happy, sad, surprise, neutral]

RAF-DB is used exclusively as a cross-dataset benchmark to test robustness
under domain shift.
"""

import os
import zipfile

from torchvision import datasets


# Mapping from RAF-DB numeric folder names to FER2013-compatible names.
# The numeric prefix ensures ImageFolder assigns the correct class index.
RAFDB_FOLDER_MAPPING = {
    '1': '5_surprise',
    '2': '2_fear',
    '3': '1_disgust',
    '4': '3_happy',
    '5': '4_sad',
    '6': '0_angry',
    '7': '6_neutral',
}


def prepare_rafdb(zip_path, extract_to):
    """Extract RAF-DB zip and rename test-set folders to FER2013 order.

    The rename step ensures `ImageFolder` produces class indices that match
    the model's output classes (FER2013 alphabetical order).

    Args:
        zip_path: path to the RAF-DB zip file
        extract_to: directory to extract into

    Returns:
        Path to the test-set folder (ready for ImageFolder).
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"RAF-DB already extracted at {extract_to}.")

    test_dir = os.path.join(extract_to, 'DATASET', 'test')

    # Rename numeric folders (only if they haven't been renamed already)
    if os.path.isdir(os.path.join(test_dir, '0_angry')):
        print("Folders already renamed. Skipping rename step.")
    else:
        print("Renaming numeric folders to emotion names...")
        for old_name, new_name in RAFDB_FOLDER_MAPPING.items():
            old_path = os.path.join(test_dir, old_name)
            new_path = os.path.join(test_dir, new_name)
            if os.path.isdir(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"  Renamed '{old_name}' -> '{new_name}'")

    return test_dir


def get_rafdb_dataset(zip_path, extract_to, transform):
    """Prepare RAF-DB and return an ImageFolder dataset.

    Args:
        zip_path: path to the RAF-DB zip file
        extract_to: directory to extract into
        transform: torchvision transform to apply to each image

    Returns:
        torchvision.datasets.ImageFolder for the RAF-DB test set.
    """
    test_dir = prepare_rafdb(zip_path, extract_to)
    return datasets.ImageFolder(root=test_dir, transform=transform)


def count_zip_files(zf, prefix, subfolder):
    """Count non-directory files under prefix/subfolder/ inside a zip.

    Helper used for dataset statistics (see notebooks/00_eda.ipynb).
    """
    folder = prefix + subfolder + '/'
    return sum(
        1 for name in zf.namelist()
        if name.startswith(folder) and not name.endswith('/')
    )
