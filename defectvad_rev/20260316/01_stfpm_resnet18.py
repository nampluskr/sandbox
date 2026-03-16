import os
import sys

# os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
# os.environ["DATASET_DIR"] = "/home/namu/myspace/NAMU/datasets"
# SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"

os.environ["BACKBONE_DIR"] = "d:\\Non_Documents\\backbones"
os.environ["DATASET_DIR"] = "e:\\datasets"
SOURCE_DIR = "d:\\Non_Documents\\_github\\defectvad\\src"

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


#####################################################################
# Model Trainer
#####################################################################
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
    import logging
    from mvtec import get_dataloader
    from evaluator import Evaluator
    from utils import set_seed, set_logging, save_weights, load_weights

    #################################################################
    # Config
    #################################################################
    EXPERIMENT_NAME = "mvtec_01_stfpm_resnet18"
    CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    # resnet18 / resnet50
    model = STFPMModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = STFPMTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)

    #################################################################
    # Dataloaders / Training / Evaluation
    #################################################################
    DATA_DIR = os.path.join(os.environ["DATASET_DIR"], "mvtec")
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    LOG_FILE = f"{EXPERIMENT_NAME}.log"
    WEIGHTS_PATH = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}.pth") 

    set_seed(42)
    set_logging(OUTPUT_DIR, LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {LOG_FILE}")

    train_loader = get_dataloader(
        split="train", 
        data_dir=DATA_DIR,
        category=CATEGORIES,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir=DATA_DIR,
        category=CATEGORIES,
        batch_size=1,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )

    ## Training
    trainer.fit(train_loader, max_epochs=MAX_EPOCHS, valid_loader=test_loader if VALIDATE else None)
    save_weights(model, weights_path=WEIGHTS_PATH)

    ## Evaluation
    load_weights(model, weights_path=WEIGHTS_PATH)
    evaluator = Evaluator(model)
    image_results = evaluator.evaluate_image_level(test_loader)
    logger.info("Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    if len(CATEGORIES) > 1:
        for category in CATEGORIES:
            test_loader = get_dataloader(
                split="test", 
                data_dir=DATA_DIR,
                category=category,
                batch_size=1,
                img_size=IMG_SIZE,
                crop_size=CROP_SIZE,
                normalize=NORMALIZE,
            )
            logger.info(f" > {category}:")
            image_results = evaluator.evaluate_image_level(test_loader)
            logger.info("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    del train_loader
    del test_loader
