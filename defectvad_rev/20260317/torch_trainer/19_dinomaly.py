from typing import Any
import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.dinomaly.torch_model import DinomalyModel
from defectvad.models.dinomaly.optimizer import StableAdamW, WarmCosineScheduler


# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}

class DinomalyTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, DinomalyModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
        super().__init__(model, loss_fn, device, evaluator)

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze bottleneck and decoder
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.trainable_modules = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])
        self._initialize_trainable_modules(self.trainable_modules)

    def configure_optimizers(self):
        optimizer_config = TRAINING_CONFIG["optimizer"]
        assert isinstance(optimizer_config, dict)
        self.optimizer = StableAdamW([{"params": self.trainable_modules.parameters()}], **optimizer_config)

        # Create a scheduler config with dynamically determined total steps
        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        assert isinstance(scheduler_config, dict)
        scheduler_config["total_iters"] = self.max_steps
        self.scheduler = WarmCosineScheduler(self.optimizer, **scheduler_config)

        self.gradient_clip_val = 0.1

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.model(images, global_step=self.current_step)
        return {"loss": loss}

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
