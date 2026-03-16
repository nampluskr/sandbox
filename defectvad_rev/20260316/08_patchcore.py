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

from defectvad.models.patchcore.torch_model import PatchcoreModel
from trainer import BaseTrainer


class PatchcoreTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, coreset_sampling_ratio=0.1):
        super().__init__(model, loss_fn, device, evaluator)
        self.coreset_sampling_ratio = coreset_sampling_ratio

    def on_train_start(self):
        super().on_train_start()
        self.max_epochs = 1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0).float().to(self.device)}

    def on_train_end(self):
        super().on_train_end()
        self.model.subsample_embedding(sampling_ratio=self.coreset_sampling_ratio)


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_08_patchcore"
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
        crop_size=224,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        batch_size=1,
        img_size=256,
        crop_size=224,
        normalize=True,
    )

    # resnet18 / wide_resnet50_2
    model = PatchcoreModel(
            backbone="resnet18",
            pre_trained=True,
            layers=["layer2", "layer3"],
            num_neighbors=9,
    )
    trainer = PatchcoreTrainer(model)
    trainer.fit(train_loader, max_epochs=1)

    evaluator = Evaluator(model)
    image_results = evaluator.evaluate_image_level(test_loader)
    logger.info("Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))
    pixel_results = evaluator.evaluate_pixel_level(test_loader)
    logger.info("Pixel-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in pixel_results.items()]))

    del train_loader
    del test_loader
