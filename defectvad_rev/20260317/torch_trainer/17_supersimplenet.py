import torch

from defectvad.common.trainer import BaseTrainer
from defectvad.models.supersimplenet.torch_model import SupersimplenetModel
from defectvad.models.supersimplenet.loss import SSNLoss


class SupersimplenetTrainer(BaseTrainer):
    def __init__(self, model, loss_fn=None, device=None, evaluator=None, supervised=False):
        if not isinstance(model, SupersimplenetModel):
            raise TypeError(f"Unexpected  model: {type(model).__name__}")

        loss_fn = SSNLoss()
        super().__init__(model, loss_fn, device, evaluator)
        self.norm_clip_val = 1 if supervised else 0

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW([
                {"params": self.model.adaptor.parameters(), "lr": 0.0001},
                {"params": self.model.segdec.parameters(), "lr": 0.0002, "weight_decay": 0.00001},
        ])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.max_epochs * 0.8), int(self.max_epochs * 0.9)],
            gamma=0.4,
        )
        self.gradient_clip_val = self.norm_clip_val

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        # masks = batch["mask"].squeeze(1).to(self.device)
        masks = None
        labels = batch["label"].to(self.device)
        anomaly_map, anomaly_score, masks, labels = self.model(images, masks, labels)
        loss = self.loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        return {"loss": loss}
