import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.anomalydino.torch_model import AnomalyDINOModel


class AnomalyDINOTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, AnomalyDINOModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
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
