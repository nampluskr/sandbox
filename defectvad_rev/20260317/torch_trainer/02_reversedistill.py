import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.reversedistill.torch_model import ReverseDistillationModel
from defectvad.models.reversedistill.loss import ReverseDistillationLoss


class ReverseDistillTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, ReverseDistillationModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = ReverseDistillationLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,                   # default: 0.005
            betas=(0.5, 0.99),
        )
        self.gradient_clip_val = 1.0    # default: None

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        loss = self.loss_fn(*self.model(images))
        return {"loss": loss}
