### Test 항목
"""
test_dataset_exists:            dataset 이 존재하는지
test_dataset_created:           dataset 이 생성되는지 (not None, len > 0, normal/anomaly 개수)
test_invalid_root_dir_error:    잘못된 경로
test_invalid_category_error:    잘못된 카테고리
test_invalid_split_error:       잘못된 스플릿
test_getitem_valid_train_data:  올바른 train data
test_getitem_valid_test_data:   올바른 test data
"""

import os
import pytest
import torch
import torchvision.transforms as T


class BaseTestDataset:
    DatasetClass = None
    root_key = None
    category = None
    test_label_set = {0, 1}
    transform = None
    img_size = None

    def _create_dataset(self, config, split):
        return self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split=split,
            transform=self.transform,
            mask_transform=self.transform,
        )

    def test_dataset_class_exists(self):
        assert self.DatasetClass is not None

    # def test_invalid_category_raises_error(self, config):
    #     with pytest.raises(FileNotFoundError):
    #         self.DatasetClass(
    #             root_dir=config[self.root_key],
    #             category="__invalid_category__",
    #             split="train",
    #             transform=self.transform,
    #             mask_transform=self.transform,
    #         )

    def test_invalid_split_raises_error(self, config):
        with pytest.raises(ValueError):
            self.DatasetClass(
                root_dir=config[self.root_key],
                category=self.category,
                split="__invalid_split__",
                transform=self.transform,
                mask_transform=self.transform,
            )

    def test_train_dataset_initialization(self, config):
        dataset = self._create_dataset(config, "train")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

    def test_test_dataset_initialization(self, config):
        dataset = self._create_dataset(config, "test")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

    # def test_train_labels(self, config):
    #     dataset = self._create_dataset(config, "train")

    #     if len(dataset) == 0:
    #         pytest.skip("Empty train dataset")

    #     labels = {int(dataset[i]["label"]) for i in range(min(10, len(dataset)))}
    #     assert labels == {0}

    # def test_test_labels(self, config):
    #     dataset = self._create_dataset(config, "test")

    #     if len(dataset) == 0:
    #         pytest.skip("Empty test dataset")

    #     labels = {int(dataset[i]["label"]) for i in range(min(20, len(dataset)))}
    #     assert labels.issubset(self.test_label_set)

    def test_getitem_output_format(self, config):
        dataset = self._create_dataset(config, "train")

        if len(dataset) == 0:
            pytest.skip("Empty dataset")

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "label", "defect_type", "mask"}

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].ndim == 3
        assert sample["image"].shape[1:] == (self.img_size, self.img_size)

        assert sample["label"] == 0
        assert sample["defect_type"] == "normal"
        assert sample["mask"] is None


class TestMVTecDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import MVTecDataset
        cls.DatasetClass = MVTecDataset
        cls.root_key = "MVTec_DIR"
        cls.category = "bottle"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256
        cls.test_label_set = {0, 1}


class TestViSADataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import ViSADataset
        cls.DatasetClass = ViSADataset
        cls.root_key = "VISA_DIR"
        cls.category = "candle"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256
        cls.test_label_set = {1}


class TestBTADDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import BTADDataset
        cls.DatasetClass = BTADDataset
        cls.root_key = "BTAD_DIR"
        cls.category = "01"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256
        cls.test_label_set = {0, 1}
