import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.cfa.torch_model import CfaModel
from defectvad.models.cfa.loss import CfaLoss


class CfaTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, radius=0.5):
        if not isinstance(model, CfaModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
        loss_fn = CfaLoss(
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=radius
        )
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def on_train_start(self):
        super().on_train_start()
        self.model.initialize_centroid(data_loader=self.train_loader)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        distance = self.model(images)
        loss = self.loss_fn(distance)
        return {"loss": loss}
