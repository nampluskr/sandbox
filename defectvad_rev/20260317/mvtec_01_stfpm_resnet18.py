import os
import sys
import platform

if platform.system() == "Linux":
    os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
    os.environ["DATASET_DIR"] = "/home/namu/myspace/NAMU/datasets"
    SOURCE_DIR = "/home/namu/myspace/NAMU/defectvad_rev/src"
else:
    os.environ["BACKBONE_DIR"] = "d:\\Non_Documents\\backbones"
    os.environ["DATASET_DIR"] = "e:\\datasets"
    SOURCE_DIR = "d:\\Non_Documents\\_github\\defectvad\\src"

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


if __name__ == "__main__":
    import logging
    from defectvad.common.mvtec import get_dataloader
    from defectvad.common.evaluator import Evaluator
    from defectvad.common.utils import set_seed, set_logging, save_weights, load_weights

    #################################################################
    # Configurations
    #################################################################
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True
    MAX_EPOCHS = 10
    VALIDATE = True

    from defectvad.models.stfpm.torch_model import STFPMModel
    from defectvad.models.stfpm.torch_trainer import STFPMTrainer

    # resnet18 / resnet50
    model = STFPMModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = STFPMTrainer(model, evaluator=Evaluator(model) if VALIDATE else None)

    #################################################################
    # Test Codes
    #################################################################
    EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
    SEED = 42

    DATA_DIR = os.path.join(os.environ["DATASET_DIR"], "mvtec")
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", EXPERIMENT_NAME))
    LOG_FILE = f"{EXPERIMENT_NAME}.log"
    WEIGHTS_PATH = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}.pth")

    LOADER_KWARGS = dict(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )

    set_seed(SEED)
    set_logging(OUTPUT_DIR, LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info(f" > Logging initialized: {LOG_FILE}")

    ## Data loading
    train_loader = get_dataloader(split="train", category=CATEGORIES, batch_size=BATCH_SIZE, **LOADER_KWARGS)
    test_loader = get_dataloader(split="test", category=CATEGORIES, batch_size=1, **LOADER_KWARGS)

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
            test_loader = get_dataloader(split="test", category=category, batch_size=1, **LOADER_KWARGS)
            logger.info(f" > {category}:")
            image_results = evaluator.evaluate_image_level(test_loader)
            logger.info("   Image-level: " + ", ".join([f"{k}:{v:.3f}" for k, v in image_results.items()]))

    del evaluator, trainer, model
    del train_loader, test_loader

