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

from defectvad.models.fastflow.torch_model import FastflowModel
from defectvad.models.fastflow.loss import FastflowLoss
from trainer import BaseTrainer


class FastflowTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = FastflowLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        hidden_variables, jacobians = self.model(images)
        loss = self.loss_fn(hidden_variables, jacobians)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_05_fastflow"
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

    # resnet18 / wide_resnet50_2 / cait / deit
    model = FastflowModel(
            input_size=(256, 256),
            backbone="wide_resnet50_2",
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
        )
    trainer = FastflowTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
