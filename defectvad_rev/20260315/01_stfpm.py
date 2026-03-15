import os
import sys

# SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
SOURCE_DIR = "d:\\Non_Documents\\_github\\defectvad\\src"
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
os.environ["BACKBONE_DIR"] = "d:\\Non_Documents\\backbones"
os.environ["DATASET_DIR"] = "e:\\datasets"

#####################################################################
# Model test
#####################################################################
import logging
import torch

from defectvad.models.stfpm.torch_model import STFPMModel
from defectvad.models.stfpm.loss import STFPMLoss
from trainer import BaseTrainer


class STFPMTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        loss_fn = STFPMLoss()
        super().__init__(model, loss_fn, device, evaluator)

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
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_01_stfpm"
    LOG_FILE = f"{EXPERIMENT_NAME}.log"

    set_seed(42)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    set_logging(output_dir, LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {LOG_FILE}")

    train_loader = get_dataloader(
        split="train", 
        # data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        data_dir = "e:\\datasets\\mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        batch_size=16,
        img_size=256,
        crop_size=None,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        # data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        data_dir = "e:\\datasets\\mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=True,
    )

    model = STFPMModel(
        backbone="resnet50",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = STFPMTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
