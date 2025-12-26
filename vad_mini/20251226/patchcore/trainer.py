# src/vad_mini/models/patchcore/trainer.py

import torch
import torch.optim as optim

from vad_mini.models._components.base_trainer import BaseTrainer
from .torch_model import PatchcoreModel


class PatchcoreTrainer(BaseTrainer):
    def __init__(self, backbone="wide_resnet50_2", layers=["layer2", "layer3"],
        pre_trained=True, num_neighbors=9, coreset_sampling_ratio=0.1):

        model = PatchcoreModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        super().__init__(model, loss_fn=None)

        self.coreset_sampling_ratio = coreset_sampling_ratio

        # trainer_arguments
        self.max_epochs = 1
        self.gradient_clip_val = 0
        self.num_sanity_val_steps = 0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0, requires_grad=True, device=self.device)}

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)

        print(f"\n>> Applying core-set subsampling with ratio={self.coreset_sampling_ratio}")
        self.model.subsample_embedding(sampling_ratio=self.coreset_sampling_ratio)
        print(f">> Memory bank size: {self.model.memory_bank.size(0)}")
