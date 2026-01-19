import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters
from PIL import Image


logger = logging.getLogger(__name__)


class BaseData:
    def __init__(self, file_path, primaries=None, rotation=0):
        self.file_path = file_path
        self.rotation = rotation

        # Default: D65 standard illuminant
        self.primaries = primaries or {
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
            "W": (0.3127, 0.3290),
        }
        self._xyz = None
        self._rgb = None
        self._gray = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self.xyz[..., 1]
            self._gray = np.clip(self._gray / self._gray.max(), 0, 1)
        return self._gray

    def save_xyz(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, data=self.xyz)

    def save_rgb(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        rgb_uint8 = (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8, mode="RGB").save(save_path)

    def save_gray(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gray_uint8 = (self.gray * 255).astype(np.uint8)
        Image.fromarray(gray_uint8, mode="L").save(save_path)


class MeasuredData(BaseData):
    def __init__(self, file_path, primaries=None, rotation=0, unit=2.5):
        if not file_path.endswith(".npz"):
            raise ValueError(f"Data must be .npz file: {os.path.splitext(file_path)[-1]}")
        super().__init__(file_path, primaries, rotation)
        self.pattern = parse_filename(file_path)["pattern"]
        self._threshold = None
        self._mean = None
        self._std = None
        self.unit = unit

    @property
    def xyz(self):
        if self._xyz is None:
            self._xyz = np.load(self.file_path)["data"]
            self._xyz = rotate(self._xyz, self.rotation)
        return self._xyz

    @property
    def rgb(self):
        if self._rgb is None:
            y_max = self.xyz[..., 1].max()
            self._rgb = xyz_to_rgb(self.xyz, self.primaries, Y_white=y_max)
            self._rgb = np.clip(self._rgb, 0, 1)
        return self._rgb

    @property
    def threshold(self):
        if self._threshold is None:
            self._threshold = filters.threshold_otsu(self.gray)
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = float(value)
        self._mean = None
        self._std = None

    def _compute_stats(self, p1=1, p2=99, update_state=True):
        mask = self.gray > self.threshold
        values = self.gray[mask]
        low, high = np.percentile(values, [p1, p2])
        filtered = values[(values >= low) & (values <= high)]
        mean, std = stats.norm.fit(filtered)
        if update_state:
            self._mean, self._std = mean, std
        return mean, std

    @property
    def mean(self):
        if self._mean is None:
            self._compute_stats(update_state=True)
        return self._mean

    @property
    def std(self):
        if self._std is None:
            self._compute_stats(update_state=True)
        return self._std

    def calibrate(self, mu=0.5, sigma=0.1, p1=1, p2=99, threshold=None, shift=False):
        threshold = threshold or self.threshold
        mask = self.gray > threshold
        mean, std = self._compute_stats(p1, p2, update_state=False)

        if shift:
            result = self.gray - mean + mu
        else:
            z = (self.gray - mean) / (std + 1e-8)
            result = sigma * z + mu

        result[~mask] = 0.0
        return result

    def info(self):
        print(f"> {self.pattern}: mean={self.mean:.3f}, std={self.std:.3f}, threshold={self.threshold:.3f}")

    def show(self, std=[], unit=None):
        unit = unit or self.unit
        num_images = 5 + len(std)
        fig, axes = plt.subplots(ncols=num_images, figsize=(unit * num_images, unit * 2), facecolor="lightgray")

        axes[0].imshow(self.rgb, vmin=0.0, vmax=1.0)
        axes[1].imshow(self.gray, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].imshow(self.calibrate(mu=0.25, shift=True), cmap="gray", vmin=0.0, vmax=1.0)
        axes[3].imshow(self.calibrate(mu=0.50, shift=True), cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].imshow(self.calibrate(mu=0.75, shift=True), cmap="gray", vmin=0.0, vmax=1.0)

        axes[0].set_title(f"{self.pattern} (RGB)")
        axes[1].set_title("Y Normalized")
        axes[2].set_title(f"Y ($\\mu=${0.25}, $\\sigma$={self.std:.3f})")
        axes[3].set_title(f"Y ($\\mu=${0.50}, $\\sigma$={self.std:.3f})")
        axes[4].set_title(f"Y ($\\mu=${0.75}, $\\sigma$={self.std:.3f})")

        for i in range(len(std)):
            gray = self.calibrate(sigma=std[i], threshold=self.threshold)
            axes[4 + i].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
            axes[4 + i].set_title(f"Y ($\\sigma$={std[i]:.2f})")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        plt.show()

    # def show(self, std=[], unit=None):
    #     unit = unit or self.unit
    #     num_images = 3 + len(std)
    #     fig, axes = plt.subplots(ncols=num_images, figsize=(unit * num_images, unit * 2), facecolor="lightgray")

    #     axes[0].imshow(self.rgb, vmin=0.0, vmax=1.0)
    #     axes[1].imshow(self.gray, cmap="gray", vmin=0.0, vmax=1.0)
    #     axes[2].imshow(self.calibrate(mu=0.5, shift=True), cmap="gray", vmin=0.0, vmax=1.0)

    #     axes[0].set_title(f"{self.pattern} (RGB)")
    #     axes[1].set_title("Y Normalized")
    #     axes[2].set_title(f"Y ($\\sigma$={self.std:.3f})")

    #     for i in range(len(std)):
    #         gray = self.calibrate(sigma=std[i], threshold=self.threshold)
    #         axes[3 + i].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    #         axes[3 + i].set_title(f"Y ($\\sigma$={std[i]:.2f})")

    #     for ax in axes:
    #         ax.axis("off")

    #     fig.tight_layout()
    #     plt.show()

    def split(self, mu=0.5, std=[], unit=None):
        unit = unit or self.unit
        num_images = len(std)
        fig, axes = plt.subplots(ncols=num_images, figsize=(unit * num_images, unit * 2), facecolor="lightgray")

        for i in range(len(std)):
            gray = self.calibrate(mu=mu, sigma=std[i], threshold=self.threshold)
            axes[i].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
            axes[i].set_title(f"Y ($\\mu$={mu}, $\\sigma$={std[i]:.2f})")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        plt.show()
        
    def hist(self, std=[], unit=None):
        unit = unit or self.unit
        num_images = 5 + len(std)
        fig, axes = plt.subplots(ncols=num_images, figsize=(unit * num_images, unit * 1.05), facecolor="lightgray")

        axes[0].hist(self.rgb.ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
        axes[1].hist(self.gray.ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
        axes[1].axvline(self.threshold, color='red', linestyle='--', linewidth=1)
        axes[2].hist(self.calibrate(mu=0.25, shift=True).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
        axes[3].hist(self.calibrate(mu=0.50, shift=True).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
        axes[4].hist(self.calibrate(mu=0.75, shift=True).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)

        axes[0].set_title(f"{self.pattern} (RGB)")
        axes[1].set_title("Y Normalized")
        axes[2].set_title(f"Y ($\\mu=${0.25}, $\\sigma$={self.std:.3f})")
        axes[3].set_title(f"Y ($\\mu=${0.50}, $\\sigma$={self.std:.3f})")
        axes[4].set_title(f"Y ($\\mu=${0.75}, $\\sigma$={self.std:.3f})")

        for i in range(len(std)):
            gray = self.calibrate(mu=0.5, sigma=std[i], threshold=self.threshold)
            axes[4 + i].hist(gray.ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
            axes[4 + i].set_title(f"Y ($\\sigma$={std[i]:.2f})")

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_facecolor('lightgray')

        fig.tight_layout()
        plt.show()


class ImageData(BaseData):
    def __init__(self, file_path, primaries=None, rotation=0, unit=2.5):
        if not file_path.endswith(".png"):
            raise ValueError(f"Image must be .png file: {os.path.splitext(file_path)[-1]}")
        super().__init__(file_path, primaries, rotation)
        self.pattern = os.path.splitext(os.path.basename(file_path))[0]
        self.unit = unit

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = Image.open(self.file_path).convert("RGB")
            self._rgb = np.array(self._rgb).astype(np.float32) / 255.0
            self._rgb = rotate(self._rgb, self.rotation)
        return self._rgb

    @property
    def xyz(self):
        if self._xyz is None:
            self._xyz = rgb_to_xyz(self._rgb, self.primaries, Y_white=1.0)
        return self._xyz

    def show(self, unit=None):
        unit = unit or self.unit
        num_images = 3
        fig, axes = plt.subplots(ncols=num_images, figsize=(unit * num_images, unit * 2), facecolor="lightgray")

        axes[0].imshow(self.rgb, vmin=0.0, vmax=1.0)
        axes[1].imshow(self.gray, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].imshow(self.gray - 0.5, cmap="gray", vmin=0.0, vmax=1.0)

        axes[0].set_title(f"{self.pattern} (RGB)")
        axes[1].set_title("Y Normalized")
        axes[2].set_title("Y ($\\mu$=0.5)")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        plt.show()


#####################################################################
# Helper functions
#####################################################################

def _linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear <= 0.0031308
    srgb = np.empty_like(linear, dtype=np.float32)
    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)


def _srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    mask = srgb <= 0.04045
    linear = np.empty_like(srgb)
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)
    return np.clip(linear, 0.0, 1.0)


def _get_rgb_to_xyz_matrix(primaries, Y_white):
    def to_xyz(x, y, Y=1.0):
        y = max(y, 1e-8)
        X = x * (Y / y)
        Z = (1.0 - x - y) * (Y / y)
        return np.array([X, Y, Z], dtype=np.float32)
    xyz_r = to_xyz(*primaries["R"], Y=1.0)
    xyz_g = to_xyz(*primaries["G"], Y=1.0)
    xyz_b = to_xyz(*primaries["B"], Y=1.0)
    xyz_w = to_xyz(*primaries["W"], Y=Y_white)
    matrix = np.stack([xyz_r, xyz_g, xyz_b], axis=-1).astype(np.float32)
    scale = np.linalg.solve(matrix, xyz_w).astype(np.float32)
    return matrix * scale[np.newaxis, :]


def xyz_to_rgb(xyz, primaries, Y_white=1.0):
    matrix = _get_rgb_to_xyz_matrix(primaries, Y_white)
    linear = np.linalg.inv(matrix) @ xyz.reshape(-1, 3).T
    linear = linear.T.reshape(xyz.shape)
    linear = np.clip(linear, 0, 1).astype(np.float32)
    return _linear_to_srgb(linear)


def rgb_to_xyz(srgb, primaries, Y_white=1.0):
    matrix = _get_rgb_to_xyz_matrix(primaries, Y_white)
    linear = _srgb_to_linear(srgb)
    return np.dot(linear, matrix.T)


def parse_filename(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"(.+?) ([\d.]+) ([\d.]+)", filename)
    if match:
        return {
            "pattern": match.group(1),
            "freq": float(match.group(2)),
            "dimming": float(match.group(3)),
        }
    return {"pattern": None, "freq": None, "dimming": None}


def rotate(data, rotation=0):
    if rotation == 180:
        return np.flip(data, axis=(0, 1))
    elif rotation == 90:
        return np.rot90(data, k=1, axes=(0, 1))
    elif rotation == 270:
        return np.rot90(data, k=3, axes=(0, 1))
    elif rotation == 0:
        return data
    else:
        raise ValueError("Rotation must be one of: 0, 90, 180, 270.")


def normalize(data, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(data)
    data = (data - vmin) / (vmax - vmin + 1e-8)
    return np.clip(data, 0.0, 1.0)


def calibrate(data, mu=0.5, sigma=0.1, p1=1, p2=99, threshold=None, shift=False):
    threshold = threshold or filters.threshold_otsu(data)
    mask = data > threshold
    values = data[mask]
    low, high = np.percentile(values, [p1, p2])
    filtered = values[(values >= low) & (values <= high)]
    mean, std = stats.norm.fit(filtered)

    if shift:
        result = data - mean + mu
    else:
        z = (data - mean) / (std + 1e-8)
        result = sigma * z + mu

    result[~mask] = 0.0
    return result


def info(data, name=""):
    print(f"> {'' if name == '' else name + ':'} min {data.min():.2f}, max {data.max():.2f}")
    

if __name__ == "__main__":
    if 1:
        DATA_DIR = r"D:\Non_Documents\2025\_data_2025\AMB689LT01_MX-Miracle3_DVR_Normal"
        file_path = os.path.join(DATA_DIR, "data_npz_anomaly", "t2_10_c_ 120 20_f16.npz")

        data = MeasuredData(file_path, rotation=0)
        data.info()
        data.show()
        data.hist()
        
        data.split(std=np.linspace(0.01, 0.05, 5))
        data.split(std=np.linspace(0.06, 0.10, 5))
        data.split(std=np.linspace(0.11, 0.15, 5))
        data.split(std=np.linspace(0.16, 0.20, 5))
        
    if 1:
        DATA_DIR = r"D:\Non_Documents\2025\_data_2025\AMB678FQ02_D2I-IC_ER1"
        file_path = os.path.join(DATA_DIR, "images", "t2_10_f_.png")

        image = ImageData(file_path)
        image.show()
