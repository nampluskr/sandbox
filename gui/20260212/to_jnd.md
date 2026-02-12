### JND

```python
import os

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from colors import XYZ_to_RGB, XYZ_to_Lab, XYZ_to_Luv, XYZ_to_Yxy
```

```python
def get_uniformity(data):
    # return data.min() / data.max()
    return 1- (data.max() - data.min()) / (data.max() + data.min())

def get_luminance(path):
    data = np.load(path)["data"]
    luminance = data[..., 1].astype(np.float32)
    threshold = threshold_otsu(luminance)
    mask = luminance >= threshold
    luminance[~mask] = 0.0
    return luminance, mask

# def get_luminance(path):
#     data = np.load(path)["data"]
#     luminance = data[..., 1].astype(np.float32)

#     threshold = threshold_otsu(luminance)
#     mask_otsu = luminance >= threshold

#     low = np.percentile(luminance, 1)
#     high = np.percentile(luminance, 99)
#     mask_percentile = (luminance >= low) & (luminance <= high)

#     mask = mask_otsu | mask_percentile
#     luminance[~mask] = 0.0
#     return luminance, mask
```

```python
def to_jnd(luminance):
    A = 71.498068
    B = 94.593053
    C = 41.912053
    D = 9.8247004
    E = 0.28175407
    F = -1.1878455
    G = -0.18014349
    H = 0.14710899
    I = - 0.017046845
    mask = luminance > 0
    log = np.log10(luminance[mask])
    jnd = np.zeros_like(luminance)
    jnd[mask] = A + B*log + C*log**2 + D*log**3 + E*log**4 + \
        F*log**5 + G*log**6 + H*log**7 + I*log**8
    jnd[~mask] = 0.0
    return jnd

def to_lightness(luminance, max_luminance=800):
    mask = luminance > 0
    lightness = np.zeros_like(luminance)
    roi = luminance[mask] / max_luminance
    lightness[mask] = np.where(roi > 0.008856,
                  116 * np.power(roi, 1/3) - 16,
                  903.3 * roi)
    lightness[~mask] = 0.0
    return lightness

def to_luminance(jnd):
    a = -1.3011877
    b = -2.5840191E-2
    c = 8.0242636E-2
    d = -1.0320229E-1
    e = 1.3646699E-1
    f = 2.8745620E-2
    g = -2.5468404E-2
    h = -3.1978977E-3
    k = 1.2992634E-4
    m = 1.3635334E-3
    mask = jnd > 0
    log = np.log(jnd[mask])
    luminance = np.zeros_like(jnd)
    luminance[mask] = 10**((a + c*log + e*log**2 + g*log**3 + m*log**4) / \
           (1 + b*log + d*log**2 + f*log**3 + h*log**4 + k*log**5))
    luminance[~mask] = 0.0
    return luminance
```

```python
fig, axes = plt.subplots(3, 1, figsize=(6, 9))
luminance = np.linspace(0.01, 800, 1023)
lightness = to_lightness(luminance)
jnd = np.linspace(1, 800, 1023)

axes[0].plot(luminance, to_jnd(luminance), "k", lw=1.5)
axes[0].set_xlabel("Luminance")
axes[0].set_ylabel("JND")

axes[1].plot(luminance, to_jnd(luminance) / 800 * 100, "k", lw=1.5, label="Luminance")
axes[1].plot(luminance, to_lightness(luminance, max_luminance=800), "r", lw=1.5, label="Lightness")
axes[1].set_xlabel("Luminance")
axes[1].set_ylabel("Lightness")

axes[2].semilogy(jnd, to_luminance(jnd), "k", lw=1.5)
axes[2].set_xlabel("JND")
axes[2].set_ylabel("Luminance")

for ax in axes.flatten():
    ax.set_xlim(0, 800)
    ax.grid(ls=":", lw=1, c="k")
fig.tight_layout()
plt.show()
```

```python
# data_dir = r"E:\_data_archive\20260116_A27_2D\measured"
# filename = "t2_10_c 120_HS 20_f16.npz"
# filename = "W127 120_HS 20_f16.npz"

data_dir = r"E:\_data_archive\20260114_Q8_Type3_TMD3\measured"
filename = "G255 10_HS 500_f16.npz"
# filename = "t2_10_h 10_HS 1_f16.npz"

luminance, mask = get_luminance(os.path.join(data_dir, filename))
jnd = to_jnd(luminance)
lightness = to_lightness(luminance, max_luminance=800)

roi = luminance[mask] / luminance[mask].max()
print("\n*** Luminane:")
print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
      f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")

roi = jnd[mask]
print("\n*** JND:")
print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
      f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")

roi = lightness[mask]
print("\n*** Lightness:")
print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
      f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")
```

```python
# data_dir = r"D:\Non_Documents\2025\_data_2024\AMB6749C01_WU-BRS-Demux_ER2\data_npz"
# # data_dir = r"D:\Non_Documents\2025\_data_2024\AMB679FN01_MX-Eureka3_PRA\data_npz"
# filename = "t2_10_b_ 120 183_f16.npz"
# # filename = "W255 120 183_f16.npz"

data_dir = r"E:\_data_archive\20250927_AMB689LT01_MX-Miracle3_DVR_Normal\measured"
# filename = "t2_18_b 120 1500_f16.npz"
filename = "W255 120 500_f16.npz"

# data_dir = r"E:\_data_archive\20251030_AMB678FQ02_D2I-IC_ER1\measured"
# filename = "W255 120 80_f16.npz"
# filename = "t2_10_b 120 800_f16.npz"

print(f">> {filename}")
luminance, mask = get_luminance(os.path.join(data_dir, filename))
jnd = to_jnd(luminance)
lightness = to_lightness(luminance, max_luminance=1500)

if 1:
    roi = luminance[mask]
    print("*** Luminane:")
    print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
        f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")

    roi = jnd[mask]
    print("*** JND:")
    print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
        f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")

    roi = lightness[mask]
    print("*** Lightness:")
    print(f">> mean={roi.mean():.2f}, std={roi.std():.2f} "
        f"(min={roi.min():.2f}, max={roi.max():.2f}, unif={get_uniformity(roi):.2f})")

if 1:
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    
    # Luminance
    luminance_mean = luminance[mask].mean()
    luminance_std = luminance[mask].std()
    # luminance = np.zeros_like(luminance)
    luminance[mask] = luminance[mask] - luminance[mask].mean()
    luminance[luminance < -luminance[mask].std() * 3] = 0.0

    im0 = axes[0].imshow(luminance, cmap="Grays_r", vmin=-luminance_std * 2, vmax=luminance_std * 2)
    axes[0].set_title(f"Luminance\n($\\mu$={luminance_mean:.2f}, $\\sigma={luminance_std:.2f}$)")
    cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.7, ticks=[-luminance_std * 2, 0, luminance_std * 2])

    im1 = axes[1].imshow(luminance, cmap="coolwarm", vmin=-luminance_std * 2, vmax=luminance_std * 2)
    axes[1].set_title(f"Luminance\n($\\mu$={luminance_mean:.2f}, $\\sigma={luminance_std:.2f}$)")
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.7, ticks=[-luminance_std * 2, 0, luminance_std * 2])

    # JND
    jnd_mean = jnd[mask].mean()
    jnd_std = jnd[mask].std()
    jnd[mask] = jnd[mask] - jnd_mean
    jnd[luminance < -luminance[mask].std() * 3] = 0.0
    im2 = axes[2].imshow(jnd, cmap="coolwarm", vmin=-jnd_std * 2, vmax=jnd_std * 2)
    axes[2].set_title(f"JND\n($\\mu$={jnd_mean:.2f}, $\\sigma={jnd_std:.2f}$)")
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.7, ticks=[-jnd_std * 2, 0, jnd_std * 2])

    # Lightness
    lightness_mean = lightness[mask].mean()
    lightness_std = lightness[mask].std()
    lightness[mask] = lightness[mask] - lightness_mean
    lightness[luminance < -luminance[mask].std() * 3] = 0.0
    im3 = axes[3].imshow(lightness, cmap="coolwarm", vmin=-lightness_std * 2, vmax=lightness_std * 2)
    axes[3].set_title(f"Lightness\n($\\mu$={lightness_mean:.2f}, $\\sigma$={lightness_std:.2f})")
    cbar3 = plt.colorbar(im3, ax=axes[3], shrink=0.7, ticks=[-lightness_std * 2, 0, lightness_std * 2])

    for ax in axes.flatten():
        ax.axis("off")

    fig.tight_layout()
    plt.show()
```

```python
# def get_mask_boundary(mask):
#     dilated = ndimage.binary_dilation(mask)
#     boundary = dilated & ~mask
#     return boundary

# boundary = get_mask_boundary(mask)

# fig, axes = plt.subplots(1, 3, figsize=(9, 6))

# # Luminance
# luminance_norm = luminance / luminance[mask].max()
# luminance_norm[mask] = luminance_norm[mask] - luminance_norm[mask].mean()
# im0 = axes[0].imshow(luminance_norm, cmap="coolwarm", vmin=-0.3, vmax=0.3)
# axes[0].set_title("Luminance (Norm.)")
# luminance_masked = np.ma.masked_where(mask, luminance_norm)
# axes[0].imshow(luminance_masked, cmap="gray", vmin=0, vmax=1)
# axes[0].contour(boundary, levels=[0.1], colors='black', linewidths=2)
# cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.7, ticks=[-0.3, 0, 0.3])

# # JND
# jnd[mask] = jnd[mask] - jnd[mask].mean()
# im1 = axes[1].imshow(jnd, cmap="coolwarm", vmin=-10, vmax=10)
# axes[1].set_title("JND")
# jnd_masked = np.ma.masked_where(mask, jnd)
# axes[1].imshow(jnd_masked, cmap="gray", vmin=0, vmax=1)
# axes[1].contour(boundary, levels=[0.1], colors='black', linewidths=2)
# cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.7, ticks=[-10, 0, 10])

# # Lightness
# lightness[mask] = lightness[mask] - lightness[mask].mean()
# im2 = axes[2].imshow(lightness, cmap="coolwarm", vmin=-1, vmax=1)
# axes[2].set_title("Lightness")
# lightness_masked = np.ma.masked_where(mask, lightness)
# axes[2].imshow(lightness_masked, cmap="gray", vmin=0, vmax=1)
# axes[2].contour(boundary, levels=[0.1], colors='black', linewidths=2)
# cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.7, ticks=[-1, 0, 1])

# for ax in axes.flatten():
#     ax.axis("off")

# fig.tight_layout()
# plt.show()
```

```python
# def jnd_from_luminance(L):
#     a = 71.498068
#     b = 94.180286
#     c = 41.484032
#     d = 0.028465
#     return a + b * np.exp(-d * L) + c * np.exp(d * L)

# H, W = 256, 256
# y, x = np.ogrid[:H, :W]
# luminance = 200 + 20 * np.sin(0.1 * x) * np.sin(0.05 * y)  # 저주파 무라 패턴
# luminance += 3 * np.random.normal(0, 1, (H, W))  # 잔여 잡음
# luminance = np.clip(luminance, 1, 1000)  # 물리적 범위 제한

# 1. 휘도를 JND 수로 변환
# J = jnd_from_luminance(luminance)
J = to_jnd(luminance)

# 2. 지역 평균 JND 계산 (저주파 배경 추정)
J_smooth = ndimage.gaussian_filter(J, sigma=10)  # 주변 평균 (무라 제거된 백그라운드)

# 3. ΔJND 맵 계산 (지역 차이)
delta_J = np.abs(J - J_smooth)

# 4. 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 휘도
im0 = axes[0].imshow(luminance, cmap='Grays_r', vmin=luminance.min(), vmax=luminance.max())
axes[0].set_title("Luminance (cd/m²)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# JND 맵
im1 = axes[1].imshow(J, cmap='coolwarm', vmin=np.percentile(J, 10), vmax=np.percentile(J, 90))
axes[1].set_title("JND Map")
plt.colorbar(im1, ax=axes[1], shrink=0.8, label='JND Index')

# ΔJND 맵 (무라 강도)
im2 = axes[2].imshow(delta_J, cmap='coolwarm', vmin=0, vmax=50)
axes[2].set_title(r"$\Delta$JND Map (Mura Visibility)")
cbar = plt.colorbar(im2, ax=axes[2], shrink=0.8, label='$\\Delta$JND')
cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
cbar.ax.axhline(1.0, color='w', linestyle='--', linewidth=1)  # 1 JND 기준선

for ax in axes:
    ax.axis("off")

fig.tight_layout()
plt.show()

# 결과 해석
print(f"ΔJND 통계: 평균 = {delta_J.mean():.2f}, 최대 = {delta_J.max():.2f}")
print("ΔJND ≥ 1.0: 인간이 감지 가능한 무라 영역")
```
