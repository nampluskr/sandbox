# custom/data

import os
import re
import numpy as np
from PIL import Image

# TODO: normalize
# TODO: rotate

class Sample:
    def __init__(self, file_path, primaries=None, rotation=0):
        self.file_path = file_path
        self.rotation = rotation
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
    def xyz(self):
        if self._xyz is None:
            if self.file_path.endswith(".npz"):
                data = np.load(self.file_path)["data"]
                y_channel = data[..., 1]
                data = normalize(data, vmin=y_channel.min(), vmax=y_channel.max())
                self._xyz = rotate(data, self.rotation)
            else:
                self._xyz = rgb_to_xyz(self.rgb, self.primaries)
        return self._xyz

    @property
    def rgb(self):
        if self._rgb is None:
            if self.file_path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image = Image.open(self.file_path).convert("image")
                image = np.array(image).astype(np.float32) / 255.0
                self._rgb = rotate(image, self.rotation)
            else:
                self._rgb = xyz_to_rgb(self.xyz, self.primaries)
        return self._rgb

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self.xyz[..., 1]
        return self._gray

    def save_xyz(self, save_path):
        np.savez_compressed(save_path, data=self.xyz)

    def save_rgb(self, save_path):
        rgb_uint8 = (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8, mode="RGB")
        img.save(save_path)

    def save_gray(self, save_path):
        gray_norm = self.gray
        gray_norm = gray_norm - gray_norm.min()
        if gray_norm.max() > 0:
            gray_norm = gray_norm / gray_norm.max()
        gray_uint8 = (gray_norm * 255).astype(np.uint8)
        img = Image.fromarray(gray_uint8, mode="L")
        img.save(save_path)


# Helper 함수들은 동일하게 유지
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
        return np.clip(data, 0, None)
    data = (data - vmin) / (vmax - vmin + 1e-8)
    return np.clip(data, 0.0, None)
