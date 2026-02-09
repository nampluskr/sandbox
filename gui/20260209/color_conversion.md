### Color Conversion

```python
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import xyz2rgb, xyz2lab, xyz2luv
from skimage.color import rgb2xyz, rgb2lab, rgb2luv

from colors import XYZ_to_RGB, XYZ_to_Lab, XYZ_to_Luv, XYZ_to_Yxy


def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    return rmse
```

```python
data_dir = r"E:\_data_archive\20260116_A27_2D\measured"
Y_white_max = 800
print(f"Y_white_max: {Y_white_max:.2f}")

filename = "W255 120_HS 183_f16.npz"
data = np.load(os.path.join(data_dir, filename))["data"]

XYZ = data.astype(np.float32) / Y_white_max
h, w = data.shape[:2]

RGB = XYZ_to_RGB(XYZ)
Lab = XYZ_to_Lab(XYZ)
Luv = XYZ_to_Luv(XYZ)
Yxy = XYZ_to_Yxy(XYZ)

rgb = xyz2rgb(XYZ)
lab = xyz2lab(XYZ)
luv = xyz2luv(XYZ)
xyz = rgb2xyz(rgb)
```

```python
X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
print(f"X: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
print(f"Y: min={Y.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")
print(f"Z: min={Z.min():.2f}, max={Z.max():.2f}, mean={Z.mean():.2f}")

R, G, B = RGB[..., 0], RGB[..., 1], RGB[..., 2]
print()
print(f"R: min={R.min():.2f}, max={R.max():.2f}, mean={R.mean():.2f}")
print(f"G: min={G.min():.2f}, max={G.max():.2f}, mean={G.mean():.2f}")
print(f"B: min={B.min():.2f}, max={B.max():.2f}, mean={B.mean():.2f}")

Y, x, y = Yxy[..., 0], Yxy[..., 1], Yxy[..., 2]
print()
print(f"Y: min={R.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")
print(f"x: min={G.min():.2f}, max={x.max():.2f}, mean={x.mean():.2f}")
print(f"y: min={B.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

L, a, b = Lab[..., 0], Lab[..., 1], Lab[..., 2]
print()
print(f"L: min={L.min():.2f}, max={L.max():.2f}, mean={L.mean():.2f}")
print(f"a: min={a.min():.2f}, max={a.max():.2f}, mean={a.mean():.2f}")
print(f"b: min={b.min():.2f}, max={b.max():.2f}, mean={b.mean():.2f}")

L, u, v = Luv[..., 0], Luv[..., 1], Luv[..., 2]
print()
print(f"L: min={L.min():.2f}, max={L.max():.2f}, mean={L.mean():.2f}")
print(f"u: min={u.min():.2f}, max={u.max():.2f}, mean={u.mean():.2f}")
print(f"v: min={v.min():.2f}, max={v.max():.2f}, mean={v.mean():.2f}")
```

### RMSE

```python
print(f"> RMSE X: {rmse(XYZ[..., 0], xyz[..., 0]):.6f}")
print(f"> RMSE Y: {rmse(XYZ[..., 1], xyz[..., 1]):.6f}")
print(f"> RMSE Z: {rmse(XYZ[..., 2], xyz[..., 2]):.6f}")

print()
print(f"> RMSE R: {rmse(RGB[..., 0], rgb[..., 0]):.6f}")
print(f"> RMSE G: {rmse(RGB[..., 1], rgb[..., 1]):.6f}")
print(f"> RMSE B: {rmse(RGB[..., 2], rgb[..., 2]):.6f}")

print()
print(f"> RMSE L: {rmse(Lab[..., 0], lab[..., 0]):.6f}")
print(f"> RMSE a: {rmse(Lab[..., 1], lab[..., 1]):.6f}")
print(f"> RMSE b: {rmse(Lab[..., 2], lab[..., 2]):.6f}")

print()
print(f"> RMSE L: {rmse(Luv[..., 0], luv[..., 0]):.6f}")
print(f"> RMSE u: {rmse(Luv[..., 1], luv[..., 1]):.6f}")
print(f"> RMSE v: {rmse(Luv[..., 2], luv[..., 2]):.6f}")
```

### Show Data

```python
XYZ_vis = np.clip(XYZ, 0, 1.1) / 1.1
xyz_vis = np.clip(xyz, 0, 1.1) / 1.1

fig, axes = plt.subplots(2, 4, figsize=(6, 6))
axes[0][0].imshow(XYZ_vis, vmin=0, vmax=1.0)
axes[0][1].imshow(XYZ_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[0][2].imshow(XYZ_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[0][3].imshow(XYZ_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
axes[1][0].imshow(xyz_vis, vmin=0, vmax=1.0)
axes[1][1].imshow(xyz_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[1][2].imshow(xyz_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[1][3].imshow(xyz_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
for ax in list(axes[0].flatten()) + list(axes[1].flatten()):
    ax.axis("off")
fig.tight_layout()
plt.show()
```

```python
fig, axes = plt.subplots(2, 4, figsize=(6, 6))
axes[0][0].imshow(RGB, vmin=0, vmax=1.0)
axes[0][1].imshow(RGB[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[0][2].imshow(RGB[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[0][3].imshow(RGB[..., 2], cmap="gray", vmin=0, vmax=1.0)
axes[1][0].imshow(rgb, vmin=0, vmax=1.0)
axes[1][1].imshow(rgb[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[1][2].imshow(rgb[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[1][3].imshow(rgb[..., 2], cmap="gray", vmin=0, vmax=1.0)
for ax in list(axes[0].flatten()) + list(axes[1].flatten()):
    ax.axis("off")
fig.tight_layout()
plt.show()
```

```python
fig, axes = plt.subplots(2, 4, figsize=(6, 6))
axes[0][0].imshow(Yxy, vmin=0, vmax=1.0)
axes[0][1].imshow(Yxy[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[0][2].imshow(Yxy[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[0][3].imshow(Yxy[..., 2], cmap="gray", vmin=0, vmax=1.0)
# axes[1][0].imshow(Yxy, vmin=0, vmax=1.0)
# axes[1][1].imshow(Yxy[..., 0], cmap="gray", vmin=0, vmax=1.0)
# axes[1][2].imshow(Yxy[..., 1], cmap="gray", vmin=0, vmax=1.0)
# axes[1][3].imshow(Yxy[..., 2], cmap="gray", vmin=0, vmax=1.0)
for ax in list(axes[0].flatten()) + list(axes[1].flatten()):
    ax.axis("off")
fig.tight_layout()
plt.show()
```

```python
L_lab = np.clip(Lab[..., 0] / 100, 0, 1)
a_lab = np.clip((Lab[..., 1] + 50) / 100, 0, 1)
b_lab = np.clip((Lab[..., 2] + 50) / 100, 0, 1)
Lab_vis = np.stack([L_lab, a_lab, b_lab], axis=-1)

L_lab = np.clip(lab[..., 0] / 100, 0, 1)
a_lab = np.clip((lab[..., 1] + 50) / 100, 0, 1)
b_lab = np.clip((lab[..., 2] + 50) / 100, 0, 1)
lab_vis = np.stack([L_lab, a_lab, b_lab], axis=-1)

fig, axes = plt.subplots(2, 4, figsize=(6, 6))
axes[0][0].imshow(Lab_vis, vmin=0, vmax=1.0)
axes[0][1].imshow(Lab_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[0][2].imshow(Lab_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[0][3].imshow(Lab_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
axes[1][0].imshow(lab_vis, vmin=0, vmax=1.0)
axes[1][1].imshow(lab_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[1][2].imshow(lab_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[1][3].imshow(lab_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
for ax in list(axes[0].flatten()) + list(axes[1].flatten()):
    ax.axis("off")
fig.tight_layout()
plt.show()
```

```python
L_luv = np.clip(Luv[..., 0] / 100, 0, 1)
u_luv = np.clip((Luv[..., 1] + 50) / 100, 0, 1)
v_luv = np.clip((Luv[..., 2] + 50) / 100, 0, 1)
Luv_vis = np.stack([L_luv, u_luv, v_luv], axis=-1)

L_luv = np.clip(luv[..., 0] / 100, 0, 1)
u_luv = np.clip((luv[..., 1] + 50) / 100, 0, 1)
v_luv = np.clip((luv[..., 2] + 50) / 100, 0, 1)
luv_vis = np.stack([L_luv, u_luv, v_luv], axis=-1)

fig, axes = plt.subplots(2, 4, figsize=(6, 6))
axes[0][0].imshow(Luv_vis, vmin=0, vmax=1.0)
axes[0][1].imshow(Luv_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[0][2].imshow(Luv_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[0][3].imshow(Luv_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
axes[1][0].imshow(luv_vis, vmin=0, vmax=1.0)
axes[1][1].imshow(luv_vis[..., 0], cmap="gray", vmin=0, vmax=1.0)
axes[1][2].imshow(luv_vis[..., 1], cmap="gray", vmin=0, vmax=1.0)
axes[1][3].imshow(luv_vis[..., 2], cmap="gray", vmin=0, vmax=1.0)
for ax in list(axes[0].flatten()) + list(axes[1].flatten()):
    ax.axis("off")
fig.tight_layout()
plt.show()
```
