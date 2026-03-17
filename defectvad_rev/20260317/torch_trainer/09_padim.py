import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.padim.torch_model import PadimModel


class PadimTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, coreset_sampling_ratio=0.1):
        if not isinstance(model, PadimModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

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
        self.model.fit()
