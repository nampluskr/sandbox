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

from defectvad.models.reversedistill.torch_model import ReverseDistillationModel
from defectvad.models.reversedistill.loss import ReverseDistillationLoss
from trainer import BaseTrainer


class ReverseDistillTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = ReverseDistillationLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,                   # default: 0.005
            betas=(0.5, 0.99),
        )
        self.gradient_clip_val = 1.0    # default: None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.loss_fn(*self.model(images))
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_02_reversedistill"
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

    model = ReverseDistillationModel(
            # backbone="wide_resnet50_2",
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            input_size=(256, 256),
            anomaly_map_mode="add",
            pre_trained=True,
        )
    trainer = ReverseDistillTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
