# src/defectvad/models/stfpm/torch_trainer.py

import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.stfpm.torch_model import STFPMModel
from defectvad.models.stfpm.loss import STFPMLoss


class STFPMTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, STFPMModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = STFPMLoss()
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
        self.gradient_clip_val = 1.0

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        teacher_features, student_features = self.model.forward(images)
        loss = self.loss_fn(teacher_features, student_features)
        return {"loss": loss}
