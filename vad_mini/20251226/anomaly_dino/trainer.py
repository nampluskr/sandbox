# src/vad_mini/models/anomaly_dino/trainer.py

import torch
import torch.optim as optim

from vad_mini.models._components.base_trainer import BaseTrainer
from .torch_model import AnomalyDINOModel


class AnomalyDINOTrainer(BaseTrainer):
    def __init__(self, encoder_name="dinov2_vit_small_14", num_neighbours=1,
        masking=False, coreset_subsampling=False, sampling_ratio=0.1):

        model = AnomalyDINOModel(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )
        super().__init__(model, loss_fn=None)

        # trainer_arguments
        self.max_epochs = 1
        self.gradient_clip_val = 0
        self.num_sanity_val_steps = 0

        self._memory_bank_built = False

    def training_step(self, batch):
        images = batch["image"].to(self.device)
        _ = self.model(images)
        return {"loss": torch.tensor(0.0, requires_grad=True, device=self.device)}

    def on_train_end(self):
        """훈련 종료 후, 메모리 뱅크 생성"""
        if not self._memory_bank_built:
            print("\n>> Building memory bank with collected embeddings...")
            self.model.fit()  # embedding_store → memory_bank + 코어셋 샘플링
            print(f">> Memory bank size: {self.model.memory_bank.size(0)}")
            self._memory_bank_built = True
        super().on_train_end()

    def on_validation_epoch_start(self):
        """검증 시작 전, 메모리 뱅크 존재 확인"""
        if self.model.memory_bank.numel() == 0:
            raise RuntimeError(
                "Memory bank is empty. Ensure `model.fit()` is called after training."
            )
        super().on_validation_epoch_start()
