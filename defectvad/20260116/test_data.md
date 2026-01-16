### Data 클래스 정의

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import xyz_to_rgb, rgb_to_xyz, rotate, normalize, equalize

class BaseData:
    def __init__(self, file_path, primaries=None, rotation=0, normalize=False):
        self.file_path = file_path
        self.rotation = rotation

        # Default: D65
        self.primaries = primaries or {
            "R": (0.640, 0.330),
            "G": (0.300, 0.600),
            "B": (0.150, 0.060),
            "W": (0.3127, 0.3290),
        }
        self.normalize = normalize
        self._xyz = None
        self._rgb = None
        self._gray = None
        self._y_min = None
        self._y_max = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self.xyz[..., 1]
        return self._gray

    @property
    def y_min(self):
        if self._y_min is None:
            self._y_min = self.gray.min()
        return self._y_min

    @property
    def y_max(self):
        if self._y_max is None:
            self._y_max = self.gray.max()
        return self._y_max

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
    def __init__(self, file_path, primaries=None, rotation=0, normalize=False):
        if not file_path.endswith(".npz"):
            raise ValueError
        super().__init__(file_path, primaries, rotation, normalize)

    @property
    def xyz(self):
        if self._xyz is None:
            data = np.load(self.file_path)["data"]
            self._xyz = rotate(data, self.rotation)
            self._gray = self._xyz[..., 1]
            self._y_min = self._gray.min()
            self._y_max = self._gray.max()

            if self.normalize:
                self._xyz = normalize(self._xyz, self._y_min, self._y_max)
                self._gray = self._xyz[..., 1]
        return self._xyz

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = xyz_to_rgb(self.xyz, self.primaries, Y_white=1.0)
            self._rgb = np.clip(self._rgb, 0.0, 1.0)
        return self._rgb


# class ImageData(BaseData):
#     def __init__(self, file_path, primaries=None, rotation=0, normalize=False):
#         if not file_path.endswith(".png"):
#             raise ValueError
#         super().__init__(file_path, primaries, rotation, normalize)

#     @property
#     def xyz(self):
#         if self._xyz is None:
#             self._xyz = rgb_to_xyz(self.rgb, self.primaries, Y_white=1.0)
#         return self._xyz

#     @property
#     def rgb(self):
#         if self._rgb is None:
#             image = Image.open(self.file_path).convert("image")
#             image = np.array(image).astype(np.float32) / 255.0
#             self._rgb = rotate(image, self.rotation)
#         return self._rgb
```

### 데이터 분포 변환

```python
sig1, sig2, sig3 = 0.01, 0.02, 0.03

def show_images_histograms(rgb, gray):
    fig, axes = plt.subplots(1, 5, figsize=(15, 6))
    axes[0].imshow(rgb, vmin=0.0, vmax=1.0)
    axes[1].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(equalize(gray, sigma=sig1), cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].imshow(equalize(gray, sigma=sig2), cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].imshow(equalize(gray, sigma=sig3), cmap="gray", vmin=0.0, vmax=1.0)

    axes[0].set_title("RGB Normalized")
    axes[1].set_title("Y Normalized")
    axes[2].set_title(f"Y Normalized (sig={sig1})")
    axes[3].set_title(f"Y Normalized (sig={sig2})")
    axes[4].set_title(f"Y Normalized (sig={sig3})")

    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].hist(rgb.ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
    axes[1].hist(gray.ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
    axes[2].hist(equalize(gray, sigma=sig1).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
    axes[3].hist(equalize(gray, sigma=sig2).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)
    axes[4].hist(equalize(gray, sigma=sig3).ravel(), bins=256, range=(0, 1), alpha=0.6, density=True)

    axes[0].set_title("RGB Normalized")
    axes[1].set_title("Y Normalized")
    axes[2].set_title(f"Y Normalized (sig={sig1})")
    axes[3].set_title(f"Y Normalized (sig={sig2})")
    axes[4].set_title(f"Y Normalized (sig={sig3})")

    for ax in axes:
        ax.set_xlim(-0.01, 1.01)

    fig.tight_layout()
    plt.show()


DATA_DIR = r"D:\Non_Documents\2025\_data_2025\AMB678FQ02_D2I-IC_ER1"
file_path = os.path.join(DATA_DIR, "data_npz", "W255 120 183_f16.npz")
print(f"{os.path.splitext(os.path.basename(file_path))[0][:-4]}")

XYZ = np.load(file_path)["data"]
XYZ = rotate(XYZ, rotation=180)
X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
Y_min, Y_max = Y.min(), Y.max()
print()
print(f"X: min {X.min():.2f}, max {X.max():.2f}")
print(f"Y: min {Y.min():.2f}, max {Y.max():.2f}")
print(f"Z: min {Z.min():.2f}, max {Z.max():.2f}")

primaries = {"W": (0.303, 0.314), "R": (0.680, 0.319), "G": (0.248, 0.708), "B": (0.144, 0.048)}
RGB = xyz_to_rgb(XYZ, primaries=primaries, Y_white=Y_max)
R, G, B = RGB[..., 0], RGB[..., 1], RGB[..., 2]
print()
print(f"R: min {R.min():.2f}, max {R.max():.2f}")
print(f"G: min {G.min():.2f}, max {G.max():.2f}")
print(f"B: min {B.min():.2f}, max {B.max():.2f}")

show_images_histograms(RGB, Y/Y_max)
```
