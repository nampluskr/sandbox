# defectvad/common/utils.py

import logging
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch


logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f" > Random seed set to {seed}")


def set_logging(log_dir=None, log_file="train.log", level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler (always)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy third-party libs
    logging.getLogger("numexpr").setLevel(logging.WARNING)

