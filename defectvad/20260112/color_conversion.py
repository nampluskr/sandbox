import os
import re
import numpy as np
from PIL import Image


class DataConverter:
    def __init__(self, data_path, primaries=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path not found: {data_path}")
        if os.path.isdir(data_path):
            raise IsADirectoryError(f"Path is a directory, not a file: {data_path}")

        self.data = np.load(data_path)
        if self.data.ndim != 3:
            raise ValueError("Data must be 3-dimensional.")

        info = parse_filename(data_path)
        self.pattern = info["pattern"]
        self.freq = info["freq"]
        self.dimming = info["dimming"]
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }

    def to_rgb(self):
        return XYZ_to_RGB(self.data, self.primaries, Y_white=self.dimming)


class ImageConverter:
    def __init__(self, image_path, primaries=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Path not found: {image_path}")
        if os.path.isdir(image_path):
            raise IsADirectoryError(f"Path is a directory, not a file: {image_path}")

        self.image = np.load(image_path)
        if self.image.ndim != 3:
            raise ValueError("Image must be 3-dimensional.")

        info = parse_filename(image_path)
        self.pattern = info["pattern"]
        self.freq = info["freq"]
        self.dimming = info["dimming"]
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }

    def to_xyz(self):
        return RGB_to_XYZ(self.image, self.primaries, Y_white=1.0)


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
    return (data - vmin) / (vmax - vmin + 1e-8)


def xy_to_XYZ(x, y, Y=1.0):
    if y == 0:
        y += 1e-8
    X = x * (Y / y)
    Z = (1.0 - x - y) * (Y / y)
    return np.array([X, Y, Z], dtype=np.float32)


def get_RGB2XYZ_matrix(primaries, Y_white):
    XYZ_r = xy_to_XYZ(*primaries["R"], Y=1.0)
    XYZ_g = xy_to_XYZ(*primaries["G"], Y=1.0)
    XYZ_b = xy_to_XYZ(*primaries["B"], Y=1.0)
    XYZ_w = xy_to_XYZ(*primaries["W"], Y=Y_white)
    matrix = np.stack([XYZ_r, XYZ_g, XYZ_b], axis=-1).astype(np.float32)
    scale = np.linalg.solve(matrix, XYZ_w).astype(np.float32)
    return matrix * scale[np.newaxis, :]


def linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear <= 0.0031308
    srgb = np.empty_like(linear, dtype=np.float32)
    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)


def XYZ_to_RGB(XYZ, primaries, Y_white=1.0):
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_white)
    M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
    shape = XYZ.shape
    XYZ_flat = XYZ.reshape(-1, 3).T
    linear = M_XYZ2RGB @ XYZ_flat
    linear = linear.T.reshape(shape)
    linear = np.clip(linear, 0, 1).astype(np.float32)
    return linear_to_srgb(linear)


def srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    mask = srgb <= 0.04045
    linear = np.empty_like(srgb)
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)
    return np.clip(linear, 0.0, 1.0)


def RGB_to_XYZ(srgb, primaries, Y_white=1.0):
    linear = srgb_to_linear(srgb)
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_white)
    return np.dot(linear, M_RGB2XYZ.T)


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


class XYZtoRGBConverter:
    def __init__(self, src_dir, target_dir, primaries=None, normalize=False):
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }
        self.normalize = normalize
        os.makedirs(target_dir, exist_ok=True)

    def convert_and_save(self):
        if not os.path.exists(self.src_dir):
            raise FileNotFoundError(f"Source directory does not exist: {self.src_dir}")

        file_found = False
        for file_name in os.listdir(self.src_dir):
            if not file_name.endswith(".npz"):
                continue
            file_found = True
            file_path = os.path.join(self.src_dir, file_name)
            info = parse_filename(file_name)
            dimming = info["dimming"]

            try:
                data = np.load(file_path)
                if 'data' not in data:
                    raise KeyError(f"No 'data' key in .npz file: {file_name}")
                xyz_data = data['data']
            except Exception as e:
                raise RuntimeError(f"Failed to load .npz file {file_name}: {e}")

            if xyz_data.ndim != 3 or xyz_data.shape[-1] != 3:
                raise ValueError(f"Invalid shape for XYZ data in {file_name}: {xyz_data.shape}")

            if self.normalize:
                y_data = xyz_data[..., 1]
                xyz_data = normalize(xyz_data, vmin=y_data.min(), vmax=y_data.max())
                rgb_data = XYZ_to_RGB(xyz_data, self.primaries, Y_white=1.0)
            else:
                rgb_data = XYZ_to_RGB(xyz_data, self.primaries, Y_white=dimming)
            
            rgb_image = (rgb_data * 255).astype(np.uint8)
            save_name = os.path.splitext(file_name)[0] + ".png"
            save_path = os.path.join(self.target_dir, save_name)

            Image.fromarray(rgb_image).save(save_path)

        if not file_found:
            print("Warning: No .npz files found in the source directory.")

        print(f"All files have been converted and saved to {self.target_dir}.")
        

class RGBtoXYZConverter:
    def __init__(self, src_dir, target_dir, primaries=None, output_format="npz"):
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }
        self.output_format = output_format.lower()
        if self.output_format not in ["npz", "npy"]:
            raise ValueError("output_format must be 'npz' or 'npy'.")
        os.makedirs(target_dir, exist_ok=True)

    def convert_and_save(self):
        if not os.path.exists(self.src_dir):
            raise FileNotFoundError(f"Source directory does not exist: {self.src_dir}")

        file_found = False
        for file_name in os.listdir(self.src_dir):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue
            file_found = True
            file_path = os.path.join(self.src_dir, file_name)

            try:
                image = Image.open(file_path).convert("RGB")
                rgb_data = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            except Exception as e:
                raise RuntimeError(f"Failed to load image {file_name}: {e}")

            # Convert sRGB to XYZ
            xyz_data = RGB_to_XYZ(rgb_data, self.primaries, Y_white=1.0)

            # Save
            save_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(self.target_dir, save_name)

            if self.output_format == "npz":
                np.savez(save_path + ".npz", data=xyz_data)
            elif self.output_format == "npy":
                np.save(save_path + ".npy", xyz_data)

        if not file_found:
            print("Warning: No image files found in the source directory.")

        print(f"All RGB images have been converted and saved to {self.target_dir} in {self.output_format.upper()} format.")

class DataConverter:
    def __init__(self, src_dir, target_dir, primaries=None, normalize=False, rotation=0):
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.primaries = primaries or {
            "W": (0.3127, 0.3290),
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
        }
        self.normalize = normalize
        os.makedirs(target_dir, exist_ok=True)

    def convert_to_rgb_and_save(self):
        """Convert XYZ data to RGB images and save as PNG."""
        self._convert_and_save(mode="rgb")

    def convert_to_luminance_and_save(self):
        """Extract Y (luminance) channel and save as grayscale PNG."""
        self._convert_and_save(mode="luminance")

    def _convert_and_save(self, mode):
        if not os.path.exists(self.src_dir):
            raise FileNotFoundError(f"Source directory does not exist: {self.src_dir}")

        file_found = False
        for file_name in os.listdir(self.src_dir):
            if not file_name.endswith(".npz"):
                continue
            file_found = True
            file_path = os.path.join(self.src_dir, file_name)
            info = parse_filename(file_name)
            dimming = info["dimming"]

            try:
                data = np.load(file_path)
                if 'data' not in data:
                    raise KeyError(f"No 'data' key in .npz file: {file_name}")
                xyz_data = data['data']
            except Exception as e:
                raise RuntimeError(f"Failed to load .npz file {file_name}: {e}")

            if xyz_data.ndim != 3 or xyz_data.shape[-1] != 3:
                raise ValueError(f"Invalid shape for XYZ data in {file_name}: {xyz_data.shape}")

            if self.normalize:
                y_min, y_max = xyz_data[..., 1].min(), xyz_data[..., 1].max()
                xyz_data = normalize(xyz_data, vmin=y_min, vmax=y_max)

            if mode == "rgb":
                rgb_data = XYZ_to_RGB(xyz_data, self.primaries, Y_white=dimming)
                img_array = (rgb_data * 255).astype(np.uint8)
                save_name = os.path.splitext(file_name)[0] + ".png"
                Image.fromarray(img_array).save(os.path.join(self.target_dir, save_name))

            elif mode == "luminance":
                y_channel = xyz_data[..., 1]  # Y 채널 (휘도)
                y_normalized = (y_channel / y_channel.max() * 255).astype(np.uint8) if y_channel.max() > 0 else y_channel.astype(np.uint8)
                save_name = os.path.splitext(file_name)[0] + "_Y.png"
                Image.fromarray(y_normalized, mode='L').save(os.path.join(self.target_dir, save_name))

        if not file_found:
            print("Warning: No .npz files found in the source directory.")

        print(f"All files have been saved to {self.target_dir} (mode: {mode}).")
