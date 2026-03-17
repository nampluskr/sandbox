import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.uninet.torch_model import UniNetModel
from defectvad.models.uninet.loss import UniNetLoss


class UniNetTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None):
        if not isinstance(model, UniNetModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")
        super().__init__(model, loss_fn, device, evaluator)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.student.parameters()},
                {"params": self.model.bottleneck.parameters()},
                {"params": self.model.dfs.parameters()},
                {"params": self.model.teachers.target_teacher.parameters(), "lr": 1e-6},
            ],
            lr=1e-4,        # default value: 5e-3
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            eps=1e-10,
            amsgrad=True,
        )
        milestones = [int(self.max_steps * 0.8) if self.max_steps != -1 else (self.trainer.max_epochs * 0.8)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        masks = None
        # labels = None
        # masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)
        loss = self.model(images=images, masks=masks, labels=labels)
        return {"loss": loss}
