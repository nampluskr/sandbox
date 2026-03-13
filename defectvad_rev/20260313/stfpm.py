import os
import sys

SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"

#####################################################################
# Model test
#####################################################################

import torch

from defectvad.models.stfpm.torch_model import STFPMModel
from defectvad.models.stfpm.loss import STFPMLoss
from trainer import BaseTrainer


class STFPMTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None):
        loss_fn = STFPMLoss()
        super().__init__(model, loss_fn, device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
        self.gradient_clip_val = 1.0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        teacher_features, student_features = self.model.forward(images)
        loss = self.loss_fn(teacher_features, student_features)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader

    DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
    CATEGORY = ["bottle"]
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True

    train_loader = get_dataloader(
        split="train", 
        data_dir=DATA_DIR,
        category=CATEGORY,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )

    model = STFPMModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )

    trainer = STFPMTrainer(model)
    trainer.fit(train_loader, max_epochs=5)

    del train_loader
