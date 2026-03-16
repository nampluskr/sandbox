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
from pathlib import Path
import torch

from defectvad.models.dsr.torch_model import DsrModel
from defectvad.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss
from defectvad.models.dsr.anomaly_generator import DsrAnomalyGenerator
from defectvad.components.perlin import PerlinAnomalyGenerator
from trainer import BaseTrainer


# Download the pretrained models
# Download link: https://drive.google.com/file/d/15plhikrUjYCcx23JVxxBKb-HBwKAb8UK/view?usp=sharing


class DsrTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, upsampling_train_ratio=0.7):
        super().__init__(model, loss_fn, device, evaluator)

        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = PerlinAnomalyGenerator()
        self.second_stage_loss = DsrSecondStageLoss()
        self.third_stage_loss = DsrThirdStageLoss()

        self.automatic_optimization = False
        self.upsampling_train_ratio = upsampling_train_ratio
        self.second_phase_epoch: int
        self.phase_switched = False

    @staticmethod
    def prepare_pretrained_model():
        pretrained_models_dir = Path(os.environ["BACKBONE_DIR"])
        return pretrained_models_dir / "vq_model_pretrained_128_4096.pckl"

    def configure_optimizers(self):
        num_steps = self.max_epochs
        self.second_phase_epoch = int(num_steps * self.upsampling_train_ratio)
        anneal_epoch = int(0.8 * self.second_phase_epoch)

        logging.info(f" > Phase 1 will run for {self.second_phase_epoch} epochs")
        logging.info(f" > Learning rate will anneal at epoch {anneal_epoch}")
        logging.info(f" > Phase 2 will start at epoch {self.second_phase_epoch + 1}")

        # Phase 1 optimizer (reconstruction + anomaly detection)
        self.optimizer_d = torch.optim.Adam(
            params=list(self.model.image_reconstruction_network.parameters())
            + list(self.model.subspace_restriction_module_hi.parameters())
            + list(self.model.subspace_restriction_module_lo.parameters())
            + list(self.model.anomaly_detection_module.parameters()),
            lr=0.0002,
        )
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=anneal_epoch,
            gamma=0.1
        )
        # Phase 2 optimizer (upsampling only)
        self.optimizer_u = torch.optim.Adam(
            params=self.model.upsampling_module.parameters(),
            lr=0.0002
        )
        self.gradient_clip_val = 1.0

    def on_train_start(self) -> None:
        super().on_train_start()
        ckpt: Path = self.prepare_pretrained_model()
        self.model.load_pretrained_discrete_model_weights(ckpt, self.device)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # Phase 2 시작 시 Phase 1 모듈들을 고정 (한 번만 실행)
        if self.current_epoch == self.second_phase_epoch + 1 and not self.phase_switched:
            logging.info(" > Now training upsampling module (Phase 2)")
            logging.info(" > Freezing Phase 1 modules...")

            for param in self.model.image_reconstruction_network.parameters():
                param.requires_grad = False
            for param in self.model.subspace_restriction_module_hi.parameters():
                param.requires_grad = False
            for param in self.model.subspace_restriction_module_lo.parameters():
                param.requires_grad = False
            for param in self.model.anomaly_detection_module.parameters():
                param.requires_grad = False

            for param in self.model.upsampling_module.parameters():
                param.requires_grad = True

            self.phase_switched = True

    def training_step(self, batch):
        ph1_opt = self.optimizer_d
        ph2_opt = self.optimizer_u

        if self.current_epoch <= self.second_phase_epoch:
            # Phase 1: Subspace restriction + Anomaly detection 학습
            input_image = batch["image"].to(self.device)
            anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
            model_outputs = self.model(input_image, anomaly_mask)
            loss = self.second_stage_loss(
                model_outputs["recon_feat_hi"],
                model_outputs["recon_feat_lo"],
                model_outputs["embedding_bot"],
                model_outputs["embedding_top"],
                input_image,
                model_outputs["obj_spec_image"],
                model_outputs["anomaly_map"],
                model_outputs["true_anomaly_map"],
            )
            ph1_opt.zero_grad()
            loss.backward()

            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.image_reconstruction_network.parameters())
                    + list(self.model.subspace_restriction_module_hi.parameters())
                    + list(self.model.subspace_restriction_module_lo.parameters())
                    + list(self.model.anomaly_detection_module.parameters()),
                    max_norm=self.gradient_clip_val
                )
            ph1_opt.step()
        else:
            # Phase 2: Upsampling module 학습
            input_image = batch["image"].to(self.device)
            input_image, anomaly_maps = self.perlin_generator(input_image)
            model_outputs = self.model(input_image)
            loss = self.third_stage_loss(
                model_outputs["anomaly_map"],
                anomaly_maps
            )
            ph2_opt.zero_grad()
            loss.backward()

            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.upsampling_module.parameters(),
                    max_norm=self.gradient_clip_val
                )
            ph2_opt.step()
        return {"loss": loss}

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)

        if self.current_epoch <= self.second_phase_epoch:
            self.scheduler_d.step()


if __name__ == "__main__":

    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging

    EXPERIMENT_NAME = "mvtec_16_dsr"
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
        normalize=False,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir="/home/namu/myspace/NAMU/datasets/mvtec",
        # category=["carpet", "grid", "leather", "tile", "wood"],
        category=["tile"],
        batch_size=1,
        img_size=256,
        crop_size=None,
        normalize=False,
    )

    model = DsrModel(latent_anomaly_strength=0.2)
    trainer = DsrTrainer(model, evaluator=Evaluator, upsampling_train_ratio=0.7)
    trainer.fit(train_loader, max_epochs=10, valid_loader=test_loader)


    del train_loader
    del test_loader
