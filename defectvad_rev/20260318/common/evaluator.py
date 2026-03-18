import logging
import torch
from tqdm import tqdm

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryF1Score, BinaryPrecisionRecallCurve
from torchmetrics import MetricCollection


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "aupr": BinaryAveragePrecision(),
        }).to(self.device)

    @torch.no_grad()
    def evaluate_image_level(self, dataloader, threshold=None):
        self.model.eval()
        self.metrics.reset()
        all_scores, all_labels = [], []

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(" > Image-level Evaluation")
            for batch in progress_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).long()
                preds = self.model(images)  # pred_score, anomaly_map

                scores = preds["pred_score"].flatten()
                labels = labels.flatten()
                self.metrics.update(scores, labels)

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        f1_score, f1_threshold = self.resolve_f1_threshold(all_scores, all_labels, threshold)

        results = self.metrics.compute()
        return {
            # **{k: v.item() for k, v in results.items()},
            "auroc": results["auroc"].item(),
            "aupr": results["aupr"].item(),
            "f1": f1_score.item(),
            "th": f1_threshold.item(),
        }
    
    @torch.no_grad()
    def evaluate_pixel_level(self, dataloader, threshold=None):
        self.model.eval()
        self.metrics.reset()
        all_scores, all_labels = [], []

        with tqdm(dataloader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(" > Pixel-level Evaluation")

            for batch in progress_bar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device).long()

                preds = self.model(images)
                anomaly_map = preds["anomaly_map"]

                scores = anomaly_map.flatten()
                labels = masks.flatten()
                self.metrics.update(scores, labels)

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        f1_score, f1_threshold = self.resolve_f1_threshold_fast(all_scores, all_labels, threshold)

        results = self.metrics.compute()
        return {
            # **{k: v.item() for k, v in results.items()},
            "auroc": results["auroc"].item(),
            "aupr": results["aupr"].item(),
            "f1": f1_score.item(),
            "th": f1_threshold.item(),
        }

    def resolve_f1_threshold(self, scores, labels, threshold=None):
        if threshold is not None:
            f1_metric = BinaryF1Score(threshold=threshold).to(self.device)
            return f1_metric(scores, labels), torch.tensor(threshold).float()
        else:
            fi_curve = BinaryPrecisionRecallCurve().to(self.device)
            precisions, recalls, thresholds = fi_curve(scores, labels)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            max_idx = torch.argmax(f1_scores)
            return f1_scores[max_idx], thresholds[max_idx]

    def resolve_f1_threshold_fast(self, scores, labels, threshold=None, num_thresholds=1000):
        if threshold is not None:
            f1_metric = BinaryF1Score(threshold=threshold).to(self.device)
            return f1_metric(scores, labels), torch.tensor(threshold).float()

        sorted_scores, indices = torch.sort(scores)
        sorted_labels = labels[indices]

        tp = torch.cumsum(sorted_labels, dim=0)
        fp = torch.cumsum(1 - sorted_labels, dim=0)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + (tp[-1] - tp) + 1e-8)  # TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        max_idx = torch.argmax(f1)
        return f1[max_idx], sorted_scores[max_idx]

    @torch.no_grad()
    def calibrate_image_thresholds(self, dataloader):
        self.model.eval()

        image_scores = []
        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            images = images[normal_mask]
            preds = self.model(images)
            batch_scores = preds["pred_score"].flatten()
            image_scores.append(batch_scores.cpu())

        if len(image_scores) == 0:
            raise ValueError(" > No normal samples found!")

        image_scores = torch.cat(image_scores)
        image_mean = image_scores.mean().item()
        image_std = image_scores.std().item()

        return {
            "99%": torch.quantile(image_scores, 0.99).item(),
            "97%": torch.quantile(image_scores, 0.97).item(),
            "95%": torch.quantile(image_scores, 0.95).item(),
            "3sig": image_mean + 3 * image_std,
            "2sig": image_mean + 2 * image_std,
            "1sig": image_mean + 1 * image_std,
        }

    @torch.no_grad()
    def calibrate_pixel_thresholds(self, dataloader, num_bins=10000):
        """
        TODO:
        suspicious_ratio = (anomaly_map > pixel_threshold).float().mean()
        if suspicious_ratio > 0.01:
            # "anomaly"

        # 이미지 판정
        image_thresholds = calibrate_image_thresholds(train_loader)
        pixel_thresholds = calibrate_pixel_thresholds(train_loader)

        # 추론시
        image_score = anomaly_map.max()
        if image_score > image_thresholds["95%"]:
            # 의심 이미지
            defect_mask = anomaly_map > pixel_thresholds["99%"]
            if defect_mask.sum() > min_pixels:
                # 최종 불량 판정
        """

        self.model.eval()

        #############################################################
        # Pass 1: 기초 통계 계산
        #############################################################

        pixel_count = 0
        pixel_mean = 0.0
        pixel_m2 = 0.0
        pixel_min = float('inf')
        pixel_max = float('-inf')

        logger.info(" > Pass 1: Computing pixel statistics...")
        for batch in tqdm(dataloader, desc="Pass 1", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            images = images[normal_mask]
            preds = self.model(images)
            anomaly_maps = preds["anomaly_map"]
            pixels = anomaly_maps.flatten().cpu()

            # Welford's online algorithm
            for pixel in pixels:
                pixel_count += 1
                delta = pixel - pixel_mean
                pixel_mean += delta / pixel_count
                delta2 = pixel - pixel_mean
                pixel_m2 += delta * delta2

            # Min, Max 추적
            batch_min = pixels.min().item()
            batch_max = pixels.max().item()
            pixel_min = min(pixel_min, batch_min)
            pixel_max = max(pixel_max, batch_max)

        if pixel_count == 0:
            raise ValueError(" > No normal samples found!")

        pixel_std = (pixel_m2 / pixel_count) ** 0.5

        # Histogram 설정
        bin_min = pixel_min
        bin_max = pixel_max
        bin_width = (bin_max - bin_min) / num_bins
        histogram = torch.zeros(num_bins)

        #############################################################
        # Pass 2: Histogram 구축
        #############################################################

        logging.info(" > Pass 2: Building histogram...")
        for batch in tqdm(dataloader, desc="Pass 2", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"]

            normal_mask = (labels == 0)
            if not normal_mask.any():
                continue

            images = images[normal_mask]
            preds = self.model(images)
            anomaly_maps = preds["anomaly_map"]
            pixels = anomaly_maps.flatten().cpu()

            # Bin index 계산
            bin_indices = ((pixels - bin_min) / bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

            # Histogram 업데이트
            for idx in bin_indices:
                histogram[idx] += 1

        # Cumulative histogram
        logging.info(" > Computing quantiles...")
        cumsum = histogram.cumsum(dim=0)

        # Quantile 계산
        def get_quantile(q):
            target_count = pixel_count * q
            bin_idx = torch.searchsorted(cumsum, target_count)
            bin_idx = min(bin_idx, num_bins - 1)
            return bin_min + (bin_idx + 0.5) * bin_width

        return {
            "99%": get_quantile(0.99).item(),
            "97%": get_quantile(0.97).item(),
            "95%": get_quantile(0.95).item(),
            "3sig": (pixel_mean + 3 * pixel_std).item(),
            "2sig": (pixel_mean + 2 * pixel_std).item(),
            "1sig": (pixel_mean + 1 * pixel_std).item(),
        }
