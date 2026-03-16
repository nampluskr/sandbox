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

from defectvad.models.fre.torch_model import FREModel
from trainer import BaseTrainer


class FRETrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = torch.nn.MSELoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=self.model.fre_model.parameters(), 
            lr=1e-3
        )

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        features_in, features_out, _ = self.model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_14_fre"
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
        batch_size=16,
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

    # resnet18(16384), resnet50 (65536)
    model = FREModel(
        backbone="resnet18", 
        layer="layer3", 
        pooling_kernel_size=2,
        input_dim=16384,
        latent_dim=220,
    )
    trainer = FRETrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
