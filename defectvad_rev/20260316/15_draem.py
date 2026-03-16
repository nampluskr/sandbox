import os
import sys

SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
os.environ["DATASET_DIR"] = "/home/namu/myspace/NAMU/datasets"

#####################################################################
# Model test
#####################################################################
import logging
import torch

from defectvad.models.draem.torch_model import DraemModel
from defectvad.models.draem.loss import DraemLoss
from defectvad.components.perlin import PerlinAnomalyGenerator
from trainer import BaseTrainer


class DraemTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, enable_sspcab=False):
        loss_fn = DraemLoss()
        super().__init__(model, loss_fn, device, evaluator)

        dtd_dir = os.path.join(os.environ["DATASET_DIR"], "dtd")
        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=dtd_dir, blend_factor=(0.1, 1.0))
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = 0.1

    def setup_sspcab(self) -> None:
        def get_activation(name: str) -> Callable:
            def hook(_, __, output: torch.Tensor) -> None:  # noqa: ANN001
                self.sspcab_activations[name] = output
            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def training_step(self, batch):
        input_image = batch["image"].to(self.device)
        augmented_image, anomaly_mask = self.augmenter(input_image)
        reconstruction, prediction = self.model(augmented_image)
        loss = self.loss_fn(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )
        return {"loss": loss}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[400, 600], gamma=0.1)


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_15_draem"
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
        normalize=False,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        category=["carpet", "grid", "leather", "tile", "wood"],
        # category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=False,
    )

    model = DraemModel(sspcab=False)
    trainer = DraemTrainer(model, evaluator=Evaluator, enable_sspcab=False)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
