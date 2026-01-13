# defectvad/data/vsia.py

import os
import pandas as pd

from .base_dataset import BaseDataset


class ViSADataset(BaseDataset):
    NAME = "visa"
    CATEGORIES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum'
    ]

    def __init__(self, root_dir, category, split, transform=None, mask_transform=None):
        csv_path = os.path.join(root_dir, "split_csv", "1cls.csv")
        df = pd.read_csv(csv_path)
        category = [category] if isinstance(category, str) else list(category)
        self.df = df[df["object"].isin(category)].reset_index(drop=True)
        self.root_dir = root_dir

        super().__init__(root_dir, category, split, transform, mask_transform)

    def _load_train_samples(self):
        df = self.df[self.df["split"] == "train"].reset_index(drop=True)
        self._load_samples_from_df(df)

    def _load_test_samples(self):
        df = self.df[self.df["split"] == "test"].reset_index(drop=True)
        self._load_samples_from_df(df)

    def _load_samples_from_df(self, df):
        self.samples = []
        for category in self.category:
            assert category in self.CATEGORIES

            df_category = df[df["object"] == category]
            image_paths = [os.path.join(self.root_dir, path) for path in df_category["image"]]
            mask_paths = [
                os.path.join(self.root_dir, path) if pd.notna(path) else None
                for path in df_category["mask"]
            ]
            labels = (df_category["label"] != "normal").astype(int).tolist()
            defect_types = df_category["label"].tolist()

            self.samples.extend([
                {
                    "category": category,
                    "image_path": image_path,
                    "label": label,
                    "defect_type": defect_type,
                    "mask_path": mask_path,
                }
                for image_path, label, defect_type, mask_path
                in zip(image_paths, labels, defect_types, mask_paths)
            ])
