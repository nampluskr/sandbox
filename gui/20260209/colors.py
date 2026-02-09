import numpy as np
import skimage
import matplotlib.pyplot as plt


PRIMARIES_D65 = {
    "W": (0.312713, 0.329013),
    "R": (0.640, 0.330),
    "G": (0.300, 0.600),
    "B": (0.150, 0.060)
}

def srgb_to_linear(srgb):
    """ sRGB[0, 1] to linear RGB[0, 1] """
    srgb = np.clip(srgb, 0.0, 1.0)
    mask = srgb <= 0.04045
    linear = np.empty_like(srgb, dtype=np.float32)
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = np.power((srgb[~mask] + 0.055)/1.055, 2.4)
    return linear


def linear_to_srgb(linear):
    """ linear RGB[0, 1] to sRGB[0, 1] """
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear <= 0.0031308
    srgb = np.empty_like(linear, dtype=np.float32)
    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0/2.4) - 0.055
    return srgb


def xy_to_XYZ(x, y, Y=1.0):
    X = x * (Y / y)
    Z = (1 - x - y) * (Y / y)
    return np.array([X, Y, Z], dtype=np.float32)


def XYZ_to_Yxy(XYZ):
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    denom = X + Y + Z

    x = np.zeros_like(X, dtype=np.float32)
    y = np.zeros_like(Y, dtype=np.float32)

    mask_X = X > 1e-6
    mask_Y = Y > 1e-6
    x[mask_X] = X[mask_X] / denom[mask_X]
    y[mask_Y] = Y[mask_Y] / denom[mask_Y]
    return np.stack([x, y, Y], axis=-1)


def get_RGB2XYZ_matrix(primaries=PRIMARIES_D65, Y_white=1.0):
    XYZ_r = xy_to_XYZ(*primaries["R"], Y=1.0)
    XYZ_g = xy_to_XYZ(*primaries["G"], Y=1.0)
    XYZ_b = xy_to_XYZ(*primaries["B"], Y=1.0)
    XYZ_w = xy_to_XYZ(*primaries["W"], Y=Y_white)

    matrix = np.stack([XYZ_r, XYZ_g, XYZ_b], axis=-1).astype(np.float32)
    scale  = np.linalg.solve(matrix, XYZ_w).astype(np.float32)
    return matrix * scale[np.newaxis, :]


def RGB_to_XYZ(sRGB, primaries=PRIMARIES_D65, Y_white=1.0):
    linear = srgb_to_linear(sRGB)
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_white)

    XYZ = M_RGB2XYZ @ linear.reshape(-1, 3).T
    return XYZ.T.reshape(sRGB.shape)


def XYZ_to_RGB(XYZ, primaries=PRIMARIES_D65, Y_white=1.0):
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_white)
    M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

    linear = M_XYZ2RGB @ XYZ.reshape(-1, 3).T
    linear = linear.T.reshape(XYZ.shape)
    linear = np.clip(linear, 0, 1).astype(np.float32)
    return linear_to_srgb(linear)


def XYZ_to_Lab(XYZ, primaries=PRIMARIES_D65, Y_white=1.0):
    if primaries == PRIMARIES_D65:
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    else:
        Xn, Yn, Zn = xy_to_XYZ(*primaries["W"], Y=Y_white)

    XYZ = np.clip(XYZ, 1e-10, None)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    xr, yr, zr = X / Xn, Y / Yn, Z / Zn

    epsilon = 216 / 24389   # ≈ 0.008856
    kappa = 24389 / 27      # ≈ 903.296
    fx = np.where(xr > epsilon, np.cbrt(xr), (xr*kappa + 16)/116)
    fy = np.where(yr > epsilon, np.cbrt(yr), (yr*kappa + 16)/116)
    fz = np.where(zr > epsilon, np.cbrt(zr), (zr*kappa + 16)/116)

    L = (116*fy - 16).astype(np.float32)
    a = (500*(fx - fy)).astype(np.float32)
    b = (200*(fy - fz)).astype(np.float32)
    return np.stack([L, a, b], axis=-1)


def XYZ_to_Luv(XYZ, primaries=PRIMARIES_D65, Y_white=1.0):
    if primaries == PRIMARIES_D65:
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    else:
        Xn, Yn, Zn = xy_to_XYZ(*primaries["W"], Y=Y_white)
    XYZ = np.clip(XYZ, 1e-10, None)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]

    denom = X + 15*Y + 3*Z
    denom = np.where(denom == 0, 1e-10, denom)
    u_prime = (4 * X) / denom
    v_prime = (9 * Y) / denom

    denom_n = Xn + 15*Yn + 3*Zn
    un_prime = (4 * Xn) / denom_n
    vn_prime = (9 * Yn) / denom_n

    yr = Y / Yn
    epsilon = 216 / 24389
    kappa = 24389 / 27
    L = np.where(yr > epsilon, 116*np.cbrt(yr) - 16, yr*kappa)
    L = np.maximum(L, 0)
    u = (13 * L * (u_prime - un_prime)).astype(np.float32)
    v = (13 * L * (v_prime - vn_prime)).astype(np.float32)
    return np.stack([L, u, v], axis=-1)


if __name__ == "__main__":

    pass
