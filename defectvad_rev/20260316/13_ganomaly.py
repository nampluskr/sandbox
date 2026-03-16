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

from defectvad.models.ganomaly.torch_model import GanomalyModel
from defectvad.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss
from trainer import BaseTrainer


class GanomalyTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        super().__init__(model, loss_fn, device, evaluator)

        self.n_features = 64
        self.latent_vec_size = 100
        self.extra_layers = 0
        self.add_final_conv_layer = True

        self.generator_loss = GeneratorLoss(wadv=1, wcon=50, wenc=1)
        self.discriminator_loss = DiscriminatorLoss()

        self.min_scores = torch.tensor(float("inf"), device=self.device)
        self.max_scores = torch.tensor(float("-inf"), device=self.device)

        self.lr = 1e-5    # default: 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

    def _reset_min_max(self):
        self.min_scores = torch.tensor(float("inf"), device=self.device)
        self.max_scores = torch.tensor(float("-inf"), device=self.device)

    def configure_optimizers(self):
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
 
    def configure_early_stoppers(self):
        pass
        # self.train_early_stopper = None
        # self.valid_early_stopper = EarlyStopper(patience=10, min_delta=1e-4, monitor="auroc")

    def training_step(self, batch) -> dict:
        images = batch["image"].to(self.device)
        d_opt, g_opt = self.optimizer_d, self.optimizer_g

        # forward pass
        padded, fake, latent_i, latent_o = self.model(images)
        pred_real, _ = self.model.discriminator(padded)

        # generator update
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        g_opt.zero_grad()
        g_loss.backward(retain_graph=True)
        g_opt.step()

        # discrimator update
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
        return {"g_loss": g_loss, "d_loss": d_loss}

    def on_validation_epoch_start(self):
        self._reset_min_max()
        super().on_validation_epoch_start()

    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            self.max_scores = torch.max(self.max_scores, torch.max(outputs["pred_score"]))
            self.min_scores = torch.min(self.min_scores, torch.min(outputs["pred_score"]))
        return {**batch, **outputs}

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_13_ganomaly"
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
        crop_size=None,
        normalize=True,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        # category=["carpet", "grid", "leather", "tile", "wood"],
        category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=True,
    )

    model = GanomalyModel(
            input_size=(256, 256), 
            num_input_channels=3, 
            n_features=64, 
            latent_vec_size=100,
            extra_layers=0, 
            add_final_conv_layer=True,
        )
    trainer = GanomalyTrainer(model, evaluator=Evaluator)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
