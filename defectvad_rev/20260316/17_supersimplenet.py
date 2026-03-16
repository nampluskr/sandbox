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

from defectvad.models.supersimplenet.torch_model import SupersimplenetModel
from defectvad.models.supersimplenet.loss import SSNLoss
from trainer import BaseTrainer


class SupersimplenetTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = SSNLoss()
        super().__init__(model, loss_fn, device, evaluator)
        self.norm_clip_val = 1 if supervised else 0

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW([
                {"params": self.model.adaptor.parameters(), "lr": 0.0001},
                {"params": self.model.segdec.parameters(), "lr": 0.0002, "weight_decay": 0.00001},
        ])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.max_epochs * 0.8), int(self.max_epochs * 0.9)],
            gamma=0.4,
        )
        self.gradient_clip_val = self.norm_clip_val

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        # masks = batch["mask"].squeeze(1).to(self.device)
        masks = None
        labels = batch["label"].to(self.device)
        anomaly_map, anomaly_score, masks, labels = self.model(images, masks, labels)
        loss = self.loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        return {"loss": loss}


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_17_supersimplenet"
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

    # wide_resnet50_2
    supervised = False
    model = SupersimplenetModel(
            perlin_threshold=0.2,
            backbone="wide_resnet50_2.tv_in1k",
            layers=["layer2", "layer3"],
            # stop_grad=False if supervised else True,
            stop_grad=False,
            adapt_cls_features=False,
        )
    trainer = SupersimplenetTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
