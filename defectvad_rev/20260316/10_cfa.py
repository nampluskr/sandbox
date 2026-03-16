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

from defectvad.models.cfa.torch_model import CfaModel
from defectvad.models.cfa.loss import CfaLoss
from trainer import BaseTrainer


class CfaTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, radius=0.5):
        loss_fn = CfaLoss(
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=radius
        )
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def on_train_start(self):
        super().on_train_start()
        self.model.initialize_centroid(data_loader=self.train_loader)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        distance = self.model(images)
        loss = self.loss_fn(distance)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_10_cfa"
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
        crop_size=224,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        # category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=224,
        normalize=True,
    )

    # vgg19_bn, resnet18, wide_resnet50_2, efficientnet_b5
    model = CfaModel(
            backbone="wide_resnet50_2",
            gamma_c=1,
            gamma_d=2,
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=0.5,     # 1e-5
        )
    trainer = CfaTrainer(model, evaluator=Evaluator, radius=0.5)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)

    del train_loader
    del test_loader
