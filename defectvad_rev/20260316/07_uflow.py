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

from defectvad.models.uflow.torch_model import UflowModel
from defectvad.models.uflow.loss import UFlowLoss
from trainer import BaseTrainer


class UflowTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = UFlowLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "initial_lr": 1e-3}],
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.4,
            total_iters=25000,
        )
        self.gradient_clip_val = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        z, ljd = self.model(images)
        loss = self.loss_fn(z, ljd)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_07_uflow"
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
        batch_size=16,
        img_size=256,
        crop_size=None,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=True,
    )

    # mcait: (448, 448)
    # resnet18 / wide_resnet50_2: (256, 256)
    model = UflowModel(
            input_size=(256, 256),
            backbone="wide_resnet50_2",
            flow_steps=4,
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False,
        )
    trainer = UflowTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
