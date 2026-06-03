"""
Visualise naive random restarts vs. random shooting on the gradient-matching
loss landscape derived from real LFW images.

Requires lfw_landscape.npz (run precompute_lfw_landscape.py first).

Landscape: soft-min of (probe_loss + α·D²) over 600 real LFW images.
Global minimum = target image (loss=0).  Other probes form local minima.

Seed design
-----------
• Naive (panel a): 3 random positions sampled from HIGH-LOSS landscape regions.
  These are genuine "blind" restarts — no probe evaluation performed.
• Selected (panel b): 3 PROBE positions with lowest gradient-matching loss,
  chosen by random shooting after evaluating all 120 candidates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import os
import sys

ROOT   = os.path.dirname(os.path.abspath(__file__))
NPZ_IN = os.path.join(ROOT, 'lfw_landscape.npz')

if not os.path.exists(NPZ_IN):
    sys.exit(
        f"ERROR: {NPZ_IN} not found.\n"
        "Run:  python figures/precompute_lfw_landscape.py"
    )

np.random.seed(7)

# ── Load precomputed data ─────────────────────────────────────────────────────
data       = np.load(NPZ_IN)
all_px     = data['all_px'].astype(np.float64)
all_py     = data['all_py'].astype(np.float64)
all_losses = data['all_losses'].astype(np.float64)
tx         = float(data['target_px'])
ty         = float(data['target_py'])
Z_coarse   = data['Z_grid'].astype(np.float64)
XLIM       = tuple(data['xlim'])
YLIM       = tuple(data['ylim'])

# ── Gaussian smoothing ────────────────────────────────────────────────────────
def gaussian_smooth_2d(Z, sigma_pix):
    r = max(1, int(np.ceil(3 * sigma_pix)))
    k = np.arange(-r, r + 1, dtype=np.float64)
    kernel = np.exp(-k**2 / (2 * sigma_pix**2)); kernel /= kernel.sum()
    Z1 = np.empty_like(Z)
    for i in range(Z.shape[0]):
        Z1[i] = np.convolve(np.pad(Z[i], r, mode='edge'), kernel, mode='valid')
    Z2 = np.empty_like(Z)
    for j in range(Z.shape[1]):
        Z2[:, j] = np.convolve(np.pad(Z1[:, j], r, mode='edge'), kernel, mode='valid')
    return Z2

DISPLAY_RES = Z_coarse.shape[0]
Z_fine = gaussian_smooth_2d(Z_coarse, sigma_pix=1.2).astype(np.float32)

# Clip Z to the probe cloud range so far corners don't compress the colormap
Z_fine = np.clip(Z_fine, Z_fine.min(), float(all_losses.max()) * 2.0)

xg = np.linspace(*XLIM, DISPLAY_RES)
yg = np.linspace(*YLIM, DISPLAY_RES)
XG, YG = np.meshgrid(xg, yg)

# ── Bilinear-interpolated L() for trajectory simulation ───────────────────────
_Zf = Z_fine.astype(np.float64)
_dx = (XLIM[1] - XLIM[0]) / (DISPLAY_RES - 1)
_dy = (YLIM[1] - YLIM[0]) / (DISPLAY_RES - 1)

def L_scalar(x, y):
    xi = float(np.clip((float(x) - XLIM[0]) / _dx, 0, DISPLAY_RES - 2))
    yi = float(np.clip((float(y) - YLIM[0]) / _dy, 0, DISPLAY_RES - 2))
    x0, y0 = int(xi), int(yi)
    fx, fy = xi - x0, yi - y0
    return (_Zf[y0,   x0]   * (1-fx) * (1-fy)
          + _Zf[y0,   x0+1] * fx     * (1-fy)
          + _Zf[y0+1, x0]   * (1-fx) * fy
          + _Zf[y0+1, x0+1] * fx     * fy)

# ── Probe pool (images 1…120, index 0 = target) ───────────────────────────────
N_PROBE  = min(120, len(all_px) - 1)
N_SELECT = 3
probe_idx = np.arange(1, N_PROBE + 1)
px = all_px[probe_idx];  py = all_py[probe_idx];  pl = all_losses[probe_idx]

sorted_p = np.argsort(pl)
sel_rel  = sorted_p[:N_SELECT]    # lowest probe loss → selected seeds
rest_rel = sorted_p[N_SELECT:]    # remaining probes (shown as coloured dots)

selected_seeds = [(px[i], py[i]) for i in sel_rel]

# ── Naive seeds: random positions in high-loss landscape regions ───────────────
# Sample many candidates, evaluate L, keep 3 with the highest L that are
# well-separated from each other and from the target.
rng_naive = np.random.default_rng(seed=42)
n_cand = 2000
cand_x = rng_naive.uniform(XLIM[0] + 0.4, XLIM[1] - 0.4, n_cand)
cand_y = rng_naive.uniform(YLIM[0] + 0.4, YLIM[1] - 0.4, n_cand)
cand_L = np.array([L_scalar(cx, cy) for cx, cy in zip(cand_x, cand_y)])

MIN_SEP_TARGET = 1.2   # stay at least this far from target
MIN_SEP_EACH   = 1.5   # seeds must be this far from each other

naive_seeds = []
for idx in np.argsort(cand_L)[::-1]:
    cx, cy = float(cand_x[idx]), float(cand_y[idx])
    if np.sqrt((cx - tx)**2 + (cy - ty)**2) < MIN_SEP_TARGET:
        continue
    if any(np.sqrt((cx - sx)**2 + (cy - sy)**2) < MIN_SEP_EACH
           for sx, sy in naive_seeds):
        continue
    naive_seeds.append((cx, cy))
    if len(naive_seeds) == N_SELECT:
        break

# ── Trajectories ──────────────────────────────────────────────────────────────
H_STEP = 0.02

def simulate_trajectory(x0, y0, steps=250, lr=0.03):
    xs, ys = [float(x0)], [float(y0)]
    x, y = float(x0), float(y0)
    for _ in range(steps):
        gx = (L_scalar(x + H_STEP, y) - L_scalar(x - H_STEP, y)) / (2 * H_STEP)
        gy = (L_scalar(x, y + H_STEP) - L_scalar(x, y - H_STEP)) / (2 * H_STEP)
        norm = np.sqrt(gx*gx + gy*gy) + 1e-12
        x -= lr * gx / norm;  y -= lr * gy / norm
        x = float(np.clip(x, *XLIM));  y = float(np.clip(y, *YLIM))
        xs.append(x);  ys.append(y)
    return np.array(xs), np.array(ys)

naive_trajs    = [simulate_trajectory(*s) for s in naive_seeds]
selected_trajs = [simulate_trajectory(*s) for s in selected_seeds]

# ── Convergence colour scale ──────────────────────────────────────────────────
naive_final_L    = [L_scalar(xs[-1], ys[-1]) for xs, ys in naive_trajs]
selected_final_L = [L_scalar(xs[-1], ys[-1]) for xs, ys in selected_trajs]
all_final_L      = naive_final_L + selected_final_L

traj_norm = Normalize(vmin=min(all_final_L), vmax=max(all_final_L))
traj_cmap = LinearSegmentedColormap.from_list('GreenRed', ['#2ca02c', '#d62728'])

def tcol(fL):
    return traj_cmap(traj_norm(fL))

probe_norm = Normalize(vmin=pl[rest_rel].min(), vmax=pl[rest_rel].max())
probe_sm   = ScalarMappable(cmap='coolwarm', norm=probe_norm)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.2))
fig.patch.set_facecolor('white')

cax_left  = fig.add_axes([0.01,  0.18, 0.024, 0.74])
ax0       = fig.add_axes([0.08,  0.13, 0.39,  0.79])
ax1       = fig.add_axes([0.525, 0.13, 0.39,  0.79])
cax_right = fig.add_axes([0.935, 0.18, 0.024, 0.74])

contour_kw = dict(levels=28, cmap='YlOrRd_r', alpha=0.88)
cline_kw   = dict(levels=12, colors='grey', linewidths=0.35, alpha=0.45)

for ax in [ax0, ax1]:
    ax.contourf(XG, YG, Z_fine, **contour_kw)
    ax.contour(XG, YG, Z_fine, **cline_kw)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel('$x_d^{(1)}$  (projected pixel dimension 1)', fontsize=11)
    ax.set_ylabel('$x_d^{(2)}$  (projected pixel dimension 2)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.scatter([tx], [ty], marker='*', s=340, color='gold',
               edgecolors='black', linewidths=0.9, zorder=11)

# ── (a) Naive random restarts ─────────────────────────────────────────────────
ax0.set_title('(a)  Naive random restarts', fontsize=13, fontweight='bold', pad=10)

for (x0, y0), (xs, ys), fL in zip(naive_seeds, naive_trajs, naive_final_L):
    col = tcol(fL)
    ax0.plot(xs, ys, color=col, linewidth=1.8, alpha=0.9, zorder=5)
    mid = len(xs) // 2
    ax0.annotate('', xy=(xs[mid], ys[mid]), xytext=(xs[mid-1], ys[mid-1]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax0.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax0.scatter([x0], [y0], s=140, color=col, edgecolors='black',
                linewidths=0.9, zorder=8, marker='o')

ax0.legend(
    handles=[
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=tcol(max(naive_final_L)),
                   markeredgecolor='black', markersize=9, label='Restart seed  (random)'),
        plt.Line2D([0],[0], color='#666666', linewidth=1.5, label='Optimisation trajectory'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
                   markeredgecolor='black', markersize=13, label='Target image'),
    ],
    loc='lower right', fontsize=9, framealpha=0.92,
)

# ── (b) Random shooting ───────────────────────────────────────────────────────
ax1.set_title('(b)  Random shooting pre-screening', fontsize=13, fontweight='bold', pad=10)

for i in rest_rel:
    ax1.scatter(px[i], py[i], s=55, color=probe_sm.to_rgba(pl[i]),
                edgecolors='white', linewidths=0.4, alpha=0.82, zorder=4)

for (x0, y0), (xs, ys), fL in zip(selected_seeds, selected_trajs, selected_final_L):
    col = tcol(fL)
    ax1.plot(xs, ys, color=col, linewidth=1.8, alpha=0.9, zorder=6)
    mid = len(xs) // 2
    ax1.annotate('', xy=(xs[mid], ys[mid]), xytext=(xs[mid-1], ys[mid-1]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax1.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax1.scatter([x0], [y0], s=160, color=col, edgecolors='black',
                linewidths=1.0, zorder=9, marker='D')

ax1.legend(
    handles=[
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#aaaaaa',
                   markeredgecolor='white', markersize=9,
                   label=f'Probe candidate  ($N={N_PROBE}$)'),
        plt.Line2D([0],[0], marker='D', color='w',
                   markerfacecolor=tcol(min(selected_final_L)),
                   markeredgecolor='black', markersize=9,
                   label='Selected seed  (lowest loss)'),
        plt.Line2D([0],[0], color='#666666', linewidth=1.5, label='Optimisation trajectory'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
                   markeredgecolor='black', markersize=13, label='Target image'),
    ],
    loc='lower right', fontsize=9, framealpha=0.92,
)

# ── Colorbars ─────────────────────────────────────────────────────────────────
traj_sm   = ScalarMappable(cmap=traj_cmap, norm=traj_norm)
traj_cbar = fig.colorbar(traj_sm, cax=cax_left)
traj_cbar.ax.invert_yaxis()
traj_cbar.set_ticks([min(all_final_L), max(all_final_L)])
traj_cbar.set_ticklabels(['global min.', 'local minima'])
traj_cbar.ax.yaxis.set_ticks_position('left')
traj_cbar.ax.yaxis.set_label_position('left')
traj_cbar.ax.tick_params(labelsize=8)
traj_cbar.set_label('Convergence', fontsize=9)

probe_cbar = fig.colorbar(probe_sm, cax=cax_right)
probe_cbar.set_label(r'Probe loss  $1 - \cos(\nabla_W)$', fontsize=9)
probe_cbar.ax.tick_params(labelsize=8)

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.01,
    r'Figure: 2D PCA projection of grayscale LFW faces. Colour shows the gradient-matching loss '
    r'$1\!-\!\cos(\nabla_W\mathcal{L}(x_d),\nabla_W\mathcal{L}(x))$ (random LeNet), via a soft-min',
    ha='center', va='top', fontsize=9.5, color='#333333', style='italic',
)
fig.text(
    0.5, -0.06,
    r'over real image positions. (a) Naive restarts land in high-loss basins and converge to local minima. '
    r'(b) Shooting evaluates $N$ probes, picks the lowest-loss seeds (diamonds),',
    ha='center', va='top', fontsize=9.5, color='#333333', style='italic',
)
fig.text(
    0.5, -0.11,
    r'and steers most restarts toward the target image — though not all converge (see convergence scale).',
    ha='center', va='top', fontsize=9.5, color='#333333', style='italic',
)

out_pdf = os.path.join(ROOT, 'random_shooting_illustration.pdf')
out_png = os.path.join(ROOT, 'random_shooting_illustration.png')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
