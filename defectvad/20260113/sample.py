import numpy as np
from typing import Optional
from PIL import Image


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ color space to RGB using standard D65 illuminant matrix.
    Output is clipped to [0, 1] range.
    """
    M = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])
    rgb = np.dot(xyz, M.T)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color space to XYZ using standard D65 inverse matrix.
    Input assumed to be in [0, 1] range.
    """
    M_inv = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    xyz = np.dot(rgb, M_inv.T)
    return xyz


class Sample:
    """
    A wrapper class for handling color measurement data in XYZ or RGB format.
    Supports lazy conversion between color spaces and saving results to files.
    Can be initialized from .npz (XYZ) or image files (RGB).
    """

    def __init__(self, file_path: str):
        """
        Initialize Data object from a file.
        :param file_path: Path to .npz (XYZ) or .png/.jpg (RGB) file.
        """
        self.file_path = file_path
        self._xyz: Optional[np.ndarray] = None
        self._rgb: Optional[np.ndarray] = None
        self._gray: Optional[np.ndarray] = None

        if file_path.endswith(".npz"):
            data = np.load(file_path)
            if 'data' not in data:
                raise KeyError("NPZ file must contain 'data' key.")
            self._xyz = data['data'].astype(np.float32)  # (H, W, 3)
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file_path).convert("RGB")
            self._rgb = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        else:
            raise ValueError("Supported formats: .npz, .png, .jpg, .jpeg")

    @property
    def xyz(self) -> np.ndarray:
        """
        Get XYZ data. Computed from RGB if not loaded directly.
        """
        if self._xyz is None:
            if self._rgb is not None:
                self._xyz = rgb_to_xyz(self._rgb)
            else:
                raise RuntimeError("Cannot compute XYZ: missing base data.")
        return self._xyz

    @property
    def rgb(self) -> np.ndarray:
        """
        Get RGB data. Computed from XYZ if not loaded directly.
        """
        if self._rgb is None:
            if self._xyz is not None:
                self._rgb = xyz_to_rgb(self._xyz)
            else:
                raise RuntimeError("Cannot compute RGB: missing base data.")
        return self._rgb

    @property
    def gray(self) -> np.ndarray:
        """
        Get grayscale image using Y channel (luminance) from XYZ.
        """
        if self._gray is None:
            self._gray = self.xyz[..., 1]  # Y channel as grayscale
        return self._gray

    def save_xyz(self, save_path: str):
        """
        Save XYZ data to .npz file.
        :param save_path: Output file path (e.g., 'output.npz').
        """
        np.savez_compressed(save_path, data=self.xyz)
        print(f"Saved XYZ to {save_path}")

    def save_rgb(self, save_path: str):
        """
        Save RGB image to file (.png or .jpg).
        :param save_path: Output image path.
        """
        rgb_uint8 = (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8, mode="RGB")
        img.save(save_path)
        print(f"Saved RGB image to {save_path}")

    def save_gray(self, save_path: str):
        """
        Save grayscale image (Y channel) as normalized .png.
        :param save_path: Output image path.
        """
        gray = self.gray
        gray_norm = gray - gray.min()
        if gray_norm.max() > 0:
            gray_norm = gray_norm / gray_norm.max()
        gray_uint8 = (gray_norm * 255).astype(np.uint8)
        img = Image.fromarray(gray_uint8, mode="L")
        img.save(save_path)
        print(f"Saved Grayscale image to {save_path}")

    def __repr__(self) -> str:
        shape = self.rgb.shape if self._rgb is not None else self.xyz.shape
        return f"Data(shape={shape}, src={self.file_path})"
