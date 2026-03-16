import os
import sys

SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"

#####################################################################
# Model test
#####################################################################
import logging
import torch

from defectvad.models.uninet.torch_model import UniNetModel
from defectvad.models.uninet.loss import UniNetLoss
from trainer import BaseTrainer


class UniNetTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.student.parameters()},
                {"params": self.model.bottleneck.parameters()},
                {"params": self.model.dfs.parameters()},
                {"params": self.model.teachers.target_teacher.parameters(), "lr": 1e-6},
            ],
            lr=1e-4,        # default value: 5e-3
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            eps=1e-10,
            amsgrad=True,
        )
        milestones = [int(self.max_steps * 0.8) if self.max_steps != -1 else (self.trainer.max_epochs * 0.8)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        masks = None
        # labels = None
        # masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)
        loss = self.model(images=images, masks=masks, labels=labels)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_18_uninet"
    LOG_FILE = f"{EXPERIMENT_NAME}.log"

    set_seed(42)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    set_logging(output_dir, LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {LOG_FILE}")

    train_loader = get_dataloader(
        split="train", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        # category=["tile"],
        batch_size=4,
        img_size=256,
        crop_size=None,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        # category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=True,
    )

    # wide_resnet50_2
    model = UniNetModel(
            student_backbone="wide_resnet50_2",
            teacher_backbone="wide_resnet50_2",
            loss=UniNetLoss(temperature=0.1)
        )
    trainer = UniNetTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
