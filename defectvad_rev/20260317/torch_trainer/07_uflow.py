import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.uflow.torch_model import UflowModel
from defectvad.models.uflow.loss import UFlowLoss


class UflowTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, UflowModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = UFlowLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "initial_lr": 1e-3}],
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.4,
            total_iters=25000,
        )
        self.gradient_clip_val = None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        z, ljd = self.model(images)
        loss = self.loss_fn(z, ljd)
        return {"loss": loss}
