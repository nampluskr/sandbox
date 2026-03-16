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

from defectvad.models.dfkde.torch_model import DfkdeModel
from trainer import BaseTrainer


class DfkdeTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        super().__init__(model, loss_fn, device, evaluator)

    def on_train_start(self):
        super().on_train_start()
        self.max_epochs = 1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0).float().to(self.device)}

    def on_train_end(self):
        super().on_train_end()
        self.model.fit()


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_12_dfkde"
    LOG_FILE = f"{EXPERIMENT_NAME}.log"

    set_seed(42)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    set_logging(output_dir, LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {LOG_FILE}")

    train_loader = get_dataloader(
        split="train", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        # category=["carpet", "grid", "leather", "tile", "wood"],
        category=["tile"],
        batch_size=16,
        img_size=256,
        crop_size=224,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        # category=["carpet", "grid", "leather", "tile", "wood"],
        category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=224,
        normalize=True,
    )

    # resnet18 / resnet50
    model = DfkdeModel(
            backbone="resnet50", 
            layers=["layer4"], 
            n_pca_components=16, 
            feature_scaling_method="scale", 
            max_training_points=40000,
        )
    trainer = DfkdeTrainer(model)
    trainer.fit(train_loader, max_epochs=1)

    evaluator = Evaluator(model)
    image_results = evaluator.evaluate_image_level(test_loader)
    logger.info("Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    del train_loader
    del test_loader
