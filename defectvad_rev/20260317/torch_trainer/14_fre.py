import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.fre.torch_model import FREModel


class FRETrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, FREModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
        loss_fn = torch.nn.MSELoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            params=self.model.fre_model.parameters(), 
            lr=1e-3
        )

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        features_in, features_out, _ = self.model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        return {"loss": loss}
