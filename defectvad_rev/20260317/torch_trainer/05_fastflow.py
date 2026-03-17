import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.fastflow.torch_model import FastflowModel
from defectvad.models.fastflow.loss import FastflowLoss


class FastflowTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, FastflowModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = FastflowLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        hidden_variables, jacobians = self.model(images)
        loss = self.loss_fn(hidden_variables, jacobians)
        return {"loss": loss}
