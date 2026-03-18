# defectvad/common/base_trainer.py

import logging
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.name = model.__class__.__name__

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        if isinstance(data, torch.Tensor):
            return self.predict_images(data)
        if isinstance(data, dict):
            return self.predict_batch(data)
        if isinstance(data, DataLoader):
            return self.predict_dataloader(data)
        raise TypeError(f"Unsupported input type: {type(data)}")

    @torch.no_grad()
    def predict_images(self, images):
        self.model.eval()
        images = images.to(self.device)
        predictions = self.model(images)
        outputs = {"image": images, **predictions}
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}

    @torch.no_grad()
    def predict_batch(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        outputs = {**batch, **predictions}
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}

    @torch.no_grad()
    def predict_dataloader(self, dataloader):
        outputs = {}
        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(">> Predicting")

            for batch in progress_bar:
                batch_outputs = self.predict_batch(batch)

                for key, value in batch_outputs.items():
                    outputs.setdefault(key, []).append(value)

        for key, value in outputs.items():
            if torch.is_tensor(value[0]):
                outputs[key] = torch.cat(value, dim=0)
            else:
                outputs[key] = value

        return outputs
    
