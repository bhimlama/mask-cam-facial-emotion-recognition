"""
FERPlus dataset loader.

FERPlus is an extension of FER2013 where each image was re-labeled by 10
crowd taggers (Barsoum et al., 2016). It has 28,709 training images,
3,589 validation images, and 3,589 test images, all 48x48 grayscale.

The FERPlus label order is:
    [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]

The standard FER2013 label order (used throughout this repo) is:
    [angry, disgust, fear, happy, sad, surprise, neutral]

The `contempt` class is excluded to maintain consistency with the 7 basic
emotions.
"""

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Mapping from FERPlus class indices to FER2013 class indices (thesis Table 2)
# FERPlus: [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
# FER2013: [angry, disgust, fear, happy, sad, surprise, neutral]
FERPLUS_TO_FER2013 = [6, 3, 5, 4, 0, 1, 2]

# Class names in FER2013 alphabetical order
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class FERPlusDataset(Dataset):
    """FERPlus dataset loaded from the original CSVs.

    Args:
        fer_csv_path: path to fer2013.csv (contains pixel data and Usage split)
        ferplus_csv_path: path to fer2013new.csv (contains crowd-sourced votes)
        usage: one of 'Training', 'PublicTest', 'PrivateTest'
        transform: optional torchvision transform
    """

    def __init__(self, fer_csv_path, ferplus_csv_path, usage, transform=None):
        self.transform = transform

        df_fer = pd.read_csv(fer_csv_path)
        df_fer = df_fer[df_fer['Usage'] == usage].copy()

        df_ferplus = pd.read_csv(ferplus_csv_path)
        df_ferplus = df_ferplus[df_ferplus['Usage'] == usage].copy()

        # Decode pixel strings into 48x48 uint8 arrays
        self.images = [
            np.array(s.split(), dtype=np.uint8).reshape(48, 48)
            for s in tqdm(df_fer['pixels'], desc=f"Loading {usage} images")
        ]

        # Majority-vote labels from FERPlus annotations
        emotion_cols = [
            'neutral', 'happiness', 'surprise', 'sadness',
            'anger', 'disgust', 'fear', 'contempt',
        ]
        vote_counts = df_ferplus[emotion_cols].values
        vote_counts_7 = vote_counts[:, :7]  # drop contempt
        labels_raw = np.argmax(vote_counts_7, axis=1)

        # Remap to FER2013 order
        self.labels = np.array([FERPLUS_TO_FER2013[l] for l in labels_raw])

        self.class_names = CLASS_NAMES

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = self.images[idx]
        image = Image.fromarray(img_array)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
