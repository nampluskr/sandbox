import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.csflow.torch_model import CsFlowModel
from defectvad.models.csflow.loss import CsFlowLoss


class CsflowTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, CsFlowModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = CsFlowLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=2e-4,
            eps=1e-04,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )
        self.gradient_clip_val = 1.0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        z_dist, jacobians = self.model(images)
        loss = self.loss_fn(z_dist, jacobians)
        return {"loss": loss}
