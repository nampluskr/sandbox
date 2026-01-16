import os
import re
import numpy as np
from scipy import stats


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

from skimage.filters import threshold_otsu

def equalize(data, p1=5, p2=95, sigma=0.1, crop_ratio=None):
    if crop_ratio is not None:
        h, w = data.shape[:2]
        ch = int(h * crop_ratio) // 2
        cw = int(w * crop_ratio) // 2
        cropped = data[ch:h-ch, cw:w-cw]
        low, high = np.percentile(cropped, [p1, p2])
    else:
        low, high = np.percentile(data, [p1, p2])

    threshold = threshold_otsu(data)
    print(f">> Treshold(Otsu): {threshold:.3f}")
    values = data[data > threshold]
    low, high = np.percentile(values, [p1, p2])
    filtered = values[(values >= low) & (values <= high)]
    mean, std = stats.norm.fit(filtered)
    
    data[data < threshold] = 0.0
    z = (data - mean) / std
    z_hat = sigma * z + 0.5
    return z_hat
