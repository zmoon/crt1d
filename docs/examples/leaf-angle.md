---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Leaf angles

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import crt1d as crt
```

## Leaf inclination angle distributions $g(\theta_l)$

```{code-cell} ipython3
named = {
    "spherical": crt.leaf_angle.g_spherical,
    "uniform": crt.leaf_angle.g_uniform,
    "planophile": crt.leaf_angle.g_planophile,
    "erectrophile": crt.leaf_angle.g_erectophile,
    "plagiophile": crt.leaf_angle.g_plagiophile,
}

theta_l = np.linspace(0, np.pi/2, 200)

fig, ax = plt.subplots()

for name, g_fn in named.items():
    ax.plot(np.rad2deg(theta_l), g_fn(theta_l), label=name)

ax.autoscale(True, "x", tight=True)
ax.set(
    xlabel=r"Leaf inclination angle $\theta_l$ [deg.]",
    ylabel="Probability density",
    title=r"$g(\theta_l)$ for various distributions",
)
ax.legend()
fig.tight_layout();
```

👆 Compare to Bonan (2019) Fig. 2.6.

### Ellipsoidal

```{code-cell} ipython3
fig, ax = plt.subplots()

ellipsoidal_xs = [0.5, 1, 2, 4]

for x in ellipsoidal_xs:
    ax.plot(np.rad2deg(theta_l), crt.leaf_angle.g_ellipsoidal(theta_l, x), label=f"$x = {x}$")

ax.autoscale(True, "x", tight=True)
ax.set(
    xlabel=r"Leaf inclination angle $\theta_l$ [deg.]",
    ylabel="Probability density",
    title=r"$g(\theta_l)$ for different ellipsoidal distributions",
)
ax.legend()
fig.tight_layout();
```

👆 Compare to Bonan (2019) Fig. 2.7. We can see that $x = 1$ is the same as spherical in the previous figure. This makes sense since $x$ is the ratio of the ellipsoid axes, $b/a$. As $x > 1$, the preference is for leaves that are only slightly inclined above the horizontal (oblate [spheroid](https://en.wikipedia.org/wiki/Spheroid); left below). For $x < 1$, we have a prolate spheroid and a preference for greater inclination angles.

![Wikipedia spheroids](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Spheroids.svg/360px-Spheroids.svg.png)

+++

### Mean leaf angle from $g(\theta_l)$

```{code-cell} ipython3
df = pd.DataFrame(index=named.keys(), data={
    "mla_deg": [crt.leaf_angle.mla_from_g(g_fn) for g_fn in named.values()]
})

for x in ellipsoidal_xs:
    df.loc[f"ellipsoidal_x={x}", "mla_deg"] = crt.leaf_angle.mla_from_g(
        lambda theta_l: crt.leaf_angle.g_ellipsoidal(theta_l, x)
    )

df.round(3)
```

👆 Note that the spherical leaf angle distribution's mean leaf angle of $\approx 57.3^\circ$ is properly recovered when $x=1$ is used with the ellipsoidal $g$.

There is a formula (Campbell 1990, eq. 16) that can be used to approximate the mean leaf angle for the ellipsoidal distribution without needing to numerically integrate $g(\theta_l)$. Below we test it out.

```{code-cell} ipython3
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

x = np.linspace(0.01, 10, 200)

yhat = crt.leaf_angle.x_to_mla_approx(x)
y = np.array([crt.leaf_angle.x_to_mla_integ(xi) for xi in x])

ax1.plot(x, y, label="numerical integration of exact PDF")
ax1.plot(x, yhat, label="approximation formula")

ax2.plot(x, yhat - y, c="0.2", label="approx. minus integ.")
ax2.axhline(0, c="0.7", lw=1)

ax1.set(ylabel="Mean leaf angle [deg.]", title="Ellipsoidal mean leaf angle from $x$")
ax2.set(xlabel="$x$ (spheroid axis ratio)", ylabel="Mean leaf angle [deg.]")
ax1.legend()
ax2.legend()
fig.tight_layout();
```

The nice thing about the approximate formula is that we can invert it and obtain $x$ from mean leaf angle. Of course, we can also do this numerically.

```{code-cell} ipython3
mlas = [10, 20, 50, 57.296, 60, 80]

df = pd.DataFrame({
    "mla_deg": mlas,
    "x_approx": [crt.leaf_angle.mla_to_x_approx(mla) for mla in mlas],
    "x_integ": [crt.leaf_angle.mla_to_x_integ(mla) for mla in mlas],
})

df.round(3)
```

### $\chi_l$

This index characterizes the departure of the leaf angle distribution from spherical. Vertical leaves have $\chi_l = -1$ and horizontal leaves $\chi_l = +1$.

```{code-cell} ipython3
df = pd.DataFrame(index=named.keys(), data={
    "xl": [crt.leaf_angle.xl_from_g(g_fn) for g_fn in named.values()]
})

for x in ellipsoidal_xs:
    df.loc[f"ellipsoidal_x={x}", "xl"] = crt.leaf_angle.xl_from_g(
        lambda theta_l: crt.leaf_angle.g_ellipsoidal(theta_l, x)
    )

df.round(3)
```

## $G(\psi)$

Most of the time we use the ellipsoidal $G(\psi)$. In the canopy RT schemes, this is used for constructing $K_b(\psi)$ and also used directly by some schemes.

$$
K_b(\psi) = G(\psi) / \cos(\psi)
$$

```{code-cell} ipython3
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 7))

ax1.sharey(ax2)

sza = np.linspace(0, 89, 200)
psi = np.deg2rad(sza)
mu = np.cos(psi)

for x in ellipsoidal_xs:
    approx = crt.leaf_angle.G_ellipsoidal_approx(psi, x)
    exact = crt.leaf_angle.G_ellipsoidal(psi, x)

    l, = ax1.plot(sza, approx, ":", lw=3, label=f"approx., $x={x}$")
    c = c=l.get_color()
    ax1.plot(sza, exact, label=f"analytical, $x={x}$", c=c)
    ax2.plot(mu, exact, c=c)
    ax3.plot(sza, approx - exact, c=c)
    ax4.plot(mu, exact/mu, c=c)

# Horizontal/vertical limits for reference
for name, G_fn in {"horizontal": crt.leaf_angle.G_horizontal, "vertical": crt.leaf_angle.G_vertical}.items():
    ax1.plot(sza, G_fn(psi), c="0.7", lw=1, zorder=0)
    ax2.plot(mu, G_fn(psi), c="0.7", lw=1, zorder=0)

ax1.set(xlabel="Solar zenith angle [deg.]", ylabel="$G$", title="Ellipsoidal $G$ vs SZA")

ax2.set(xlabel="$\mu = \cos \psi$", ylabel="$G$", title="Ellipsoidal $G$ vs $\mu$")

ax3.set(xlabel="Solar zenith angle [deg.]", ylabel=r"$\delta G$", title="Approx. minus analytical")
ax3.axhline(0, c="0.7", lw=1)

ax4.set(xlabel="$\mu = \cos \psi$", ylabel="$K_b$", title="$K_b = G/\mu$ vs $\mu$", ylim=(None, 5))

for ax in [ax1, ax2, ax4]:
    ax.set_ylim(ymin=0)

for ax in fig.get_axes():
    ax.autoscale(True, "x", tight=True)

fig.legend(bbox_to_anchor=(0.98, 0.5), loc="center left")
fig.tight_layout();
```
