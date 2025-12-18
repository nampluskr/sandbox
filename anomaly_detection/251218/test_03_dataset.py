import os
import pytest
import torch
import torchvision.transforms as T
from abc import abstractmethod


class BaseTestDataset:
    DatasetClass = None
    root_key = None
    category = None
    transform = None
    img_size = None

    @abstractmethod
    def get_expected_label_set(self, split):
        """split에 따라 기대되는 label 집합 반환"""
        pass

    def _create_dataset(self, config, split):
        """데이터셋 생성 보조 함수"""
        return self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split=split,
            transform=self.transform,
            mask_transform=self.transform,
        )

    # ==================== 테스트 케이스 ====================

    def test_dataset_exists(self, config):
        """데이터셋 클래스가 존재하는지 확인"""
        assert self.DatasetClass is not None, f"{self.__class__.__name__}: DatasetClass not set"

        root_dir = config.get(self.root_key)
        assert root_dir is not None, f"Config missing '{self.root_key}'"
        assert os.path.exists(root_dir), f"Root directory does not exist: {root_dir}"

    def test_dataset_created(self, config):
        """train/test 데이터셋이 성공적으로 생성되는지 확인 (not None, len > 0)"""
        for split in ["train", "test"]:
            dataset = self._create_dataset(config, split)

            assert dataset is not None, f"{split} dataset is None"
            assert len(dataset) > 0, f"{split} dataset is empty"

            # normal/anomaly 수 체크 (옵션: 로깅)
            normal_count = sum(s["label"] == 0 for s in dataset.samples)
            anomaly_count = sum(s["label"] == 1 for s in dataset.samples)
            print(f"{split}: normal={normal_count}, anomaly={anomaly_count}")

    def test_invalid_root_dir_error(self, config):
        """잘못된 root_dir일 때 오류 발생 확인"""
        with pytest.raises(FileNotFoundError):
            self.DatasetClass(
                root_dir="/path/to/nowhere",
                category=self.category,
                split="train",
                transform=self.transform,
                mask_transform=self.transform,
            )

    def test_invalid_category_error(self, config):
        """잘못된 카테고리일 때 오류 발생 확인"""
        with pytest.raises(Exception):  # FileNotFoundError, ValueError 등
            self.DatasetClass(
                root_dir=config[self.root_key],
                category="__invalid_category__",
                split="train",
                transform=self.transform,
                mask_transform=self.transform,
            )

    def test_invalid_split_error(self, config):
        """잘못된 split일 때 ValueError 발생 확인"""
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            self.DatasetClass(
                root_dir=config[self.root_key],
                category=self.category,
                split="__invalid_split__",
                transform=self.transform,
                mask_transform=self.transform,
            )

    def test_getitem_valid_train_data(self, config):
        """train 데이터의 __getitem__ 출력 형식 및 내용 검증"""
        dataset = self._create_dataset(config, "train")
        self._validate_sample(dataset[0], expected_defect_type="normal", split="train")

    def test_getitem_valid_test_data(self, config):
        """test 데이터의 __getitem__ 출력 형식 및 내용 검증"""
        dataset = self._create_dataset(config, "test")
        self._validate_sample(dataset[0], expected_defect_type=None, split="test")

    def _validate_sample(self, sample, expected_defect_type, split):
        """샘플 형식 및 내용 검증 보조 함수"""
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "label", "defect_type", "mask"}

        # 이미지 텐서 검증
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].ndim == 3
        assert sample["image"].shape[1:] == (self.img_size, self.img_size)

        # 라벨 검증
        label = int(sample["label"])
        expected_labels = self.get_expected_label_set(split)
        assert label in expected_labels, f"Label {label} not in {expected_labels}"

        # defect_type 검증
        if expected_defect_type:
            assert sample["defect_type"] == expected_defect_type

        # mask 타입 검증
        if label == 0:
            assert sample["mask"] is None
        else:
            assert sample["mask"] is None or isinstance(sample["mask"], torch.Tensor)


# ==================== 각 데이터셋 테스트 설정 ====================

class TestMVTecDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import MVTecDataset
        cls.DatasetClass = MVTecDataset
        cls.root_key = "MVTec_DIR"
        cls.category = "bottle"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256

    def get_expected_label_set(self, split):
        return {0} if split == "train" else {0, 1}


class TestViSADataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import ViSADataset
        cls.DatasetClass = ViSADataset
        cls.root_key = "VISA_DIR"
        cls.category = "candle"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256

    def get_expected_label_set(self, split):
        return {0} if split == "train" else {1}  # ViSA test는 anomaly만 label=1


class TestBTADDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import BTADDataset
        cls.DatasetClass = BTADDataset
        cls.root_key = "BTAD_DIR"
        cls.category = "01"
        cls.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        cls.img_size = 256

    def get_expected_label_set(self, split):
        return {0} if split == "train" else {0, 1}
