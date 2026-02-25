#!/usr/bin/env python
"""
Render side-by-side README GIF: CGSchNet (Baseline) vs FlashMD (6.5x).

Uses a single trajectory (FlashMD) and plays it at different speeds
for each panel based on the throughput ratio. Left panel (baseline)
advances slowly; right panel (FlashMD) advances 6.5x faster.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import splprep, splev
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import io
import argparse

# Colors from CLAUDE.md
CORAL = "#D87756"
BLUE = "#689BCC"
ROSE = "#C46686"

# GIF parameters
FPS = 15
DURATION_SEC = 8
N_FRAMES = FPS * DURATION_SEC  # 120
FLASHMD_TOTAL_STEPS = 100_000
BASELINE_THROUGHPUT = 438  # timestep*mol/s
FLASHMD_THROUGHPUT = 2861  # timestep*mol/s
SPEEDUP = FLASHMD_THROUGHPUT / BASELINE_THROUGHPUT  # ~6.53x
BASELINE_TOTAL_STEPS = int(FLASHMD_TOTAL_STEPS / SPEEDUP)  # ~15,310
WIDTH, HEIGHT = 960, 520


def get_ca_indices(pdb_path):
    """Get CA atom indices (0-based) from PDB file."""
    ca_indices = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                atom_idx = int(line[6:11].strip())
                ca_indices.append(atom_idx - 1)  # 0-based
            if line.startswith("ENDMDL"):
                break
    return ca_indices


def center_coordinates(pos):
    """Center coordinates at origin."""
    return pos - pos.mean(axis=0)


def smooth_backbone(ca_pos, n_points=200):
    """Smooth backbone with cubic spline."""
    tck, _ = splprep(
        [ca_pos[:, 0], ca_pos[:, 1], ca_pos[:, 2]], s=2, k=3
    )
    u_fine = np.linspace(0, 1, n_points)
    return np.array(splev(u_fine, tck)).T


def make_nc_colormap():
    """N->C terminus color gradient: coral -> white -> blue."""
    return LinearSegmentedColormap.from_list(
        "nc_gradient",
        [CORAL, "#F0E0D8", "#FFFFFF", "#D8E4F0", BLUE],
        N=256,
    )


def load_trajectory(data_dir, prefix="gif_flashmd"):
    """Load all coordinate npy files for a single trajectory."""
    files = sorted(Path(data_dir).glob(f"{prefix}_coords_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No coord files matching {prefix}_coords_*.npy in {data_dir}"
        )

    chunks = []
    for f in files:
        data = np.load(f)
        chunks.append(data)
    # shape: (n_sims, total_frames, n_atoms, 3)
    all_coords = np.concatenate(chunks, axis=1)
    print(f"Loaded {len(files)} files -> {all_coords.shape}")
    return all_coords


def rotation_matrix(elev_deg, azim_deg):
    """Build a 3D->2D rotation matrix from elevation and azimuth."""
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    # Rotation about z-axis (azimuth), then x-axis (elevation)
    caz, saz = np.cos(azim), np.sin(azim)
    cel, sel = np.cos(elev), np.sin(elev)
    Rz = np.array([[caz, -saz, 0], [saz, caz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cel, -sel], [0, sel, cel]])
    return (Rx @ Rz)[:2]  # project to 2D (drop z row)


def render_protein(
    ax, ca_pos, cmap, azim=45, elev=20, view_range=None,
    ghost_ca_positions=None,
):
    """Render protein backbone with N->C gradient via 2D projection.

    Parameters
    ----------
    ghost_ca_positions : list of arrays, optional
        CA positions from other replicas to render as faint ghost traces.
    """
    from matplotlib.collections import LineCollection

    R = rotation_matrix(elev, azim)

    # Draw ghost replicas first (behind the main one)
    if ghost_ca_positions is not None:
        for ghost_ca in ghost_ca_positions:
            try:
                ghost_smooth = smooth_backbone(ghost_ca, n_points=200)
            except Exception:
                continue
            ghost_proj = ghost_smooth @ R.T
            n_g = len(ghost_proj)
            ghost_colors = cmap(np.linspace(0, 1, n_g - 1))
            # Make very transparent
            ghost_colors[:, 3] = 0.12
            pts = ghost_proj.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc_g = LineCollection(
                segs, colors=ghost_colors, linewidths=1.8,
                capstyle="round", joinstyle="round", zorder=1,
            )
            ax.add_collection(lc_g)

    # Main backbone
    smooth = smooth_backbone(ca_pos, n_points=300)
    proj = smooth @ R.T  # (N, 2)

    n = len(proj)
    colors = cmap(np.linspace(0, 1, n - 1))

    # Draw backbone as colored segments
    points = proj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidths=4.5,
                        capstyle="round", joinstyle="round", zorder=5)
    ax.add_collection(lc)

    # Terminus markers (project CA endpoints)
    ca_proj = ca_pos @ R.T
    ax.scatter(
        ca_proj[0, 0], ca_proj[0, 1], s=160, c=CORAL,
        edgecolors="white", linewidths=1.5, zorder=10,
    )
    ax.scatter(
        ca_proj[-1, 0], ca_proj[-1, 1], s=160, c=BLUE,
        edgecolors="white", linewidths=1.5, zorder=10,
    )

    # Setup axes
    if view_range is not None:
        r = view_range
    else:
        center = proj.mean(axis=0)
        r = np.abs(proj - center).max() * 1.1
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        return
    center = proj.mean(axis=0)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")


def render_frame(
    flash_pos, baseline_pos, ca_indices, cmap, azim,
    flash_step, baseline_step, view_range=None, elev=25,
    ghost_flash_positions=None, ghost_baseline_positions=None,
):
    """Render a single GIF frame as a PIL Image."""
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")

    # Prepare ghost CA positions
    ghost_ca_baseline = None
    ghost_ca_flash = None
    if ghost_baseline_positions is not None:
        ghost_ca_baseline = [
            center_coordinates(gp[ca_indices])
            for gp in ghost_baseline_positions
        ]
    if ghost_flash_positions is not None:
        ghost_ca_flash = [
            center_coordinates(gp[ca_indices])
            for gp in ghost_flash_positions
        ]

    # Left panel - Baseline (2D axes)
    ax1 = fig.add_subplot(1, 2, 1)
    ca_baseline = center_coordinates(baseline_pos[ca_indices])
    render_protein(
        ax1, ca_baseline, cmap, azim=azim, elev=elev,
        view_range=view_range, ghost_ca_positions=ghost_ca_baseline,
    )

    # Right panel - FlashMD (2D axes)
    ax2 = fig.add_subplot(1, 2, 2)
    ca_flash = center_coordinates(flash_pos[ca_indices])
    render_protein(
        ax2, ca_flash, cmap, azim=azim, elev=elev,
        view_range=view_range, ghost_ca_positions=ghost_ca_flash,
    )

    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.85, bottom=0.20, wspace=0.06
    )

    # Render to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def add_overlays(img, flash_step, baseline_step, frame_idx, total_frames):
    """Add text overlays: titles, step counters, progress bars, speed."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size_title = 26
    font_size_info = 20
    font_size_small = 17
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size_title,
        )
        font_info = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size_info,
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size_small,
        )
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_info = ImageFont.load_default()
        font_small = ImageFont.load_default()

    half_w = w // 2
    bar_width = int(half_w * 0.65)
    bar_height = 14

    # Real-time speedup annotation:
    # Both sides represent the SAME wall clock duration.
    # FlashMD: 2861 timestep*mol/s for 128 replicas
    # Per-molecule: 2861/128 = 22.35 timesteps/s
    # 100k steps real wall time: 100000/22.35 = 4474 s ≈ 74.6 min
    # GIF covers this in 8 seconds -> 4474/8 ≈ 559x real time
    flash_per_mol = FLASHMD_THROUGHPUT / 128  # timesteps/s per molecule
    wall_clock_s = FLASHMD_TOTAL_STEPS / flash_per_mol
    gif_speedup = wall_clock_s / DURATION_SEC

    # === Left: CGSchNet ===
    left_cx = half_w // 2
    draw.text(
        (left_cx, 10), "CGSchNet",
        fill=CORAL, font=font_title, anchor="mt",
    )

    step_text = f"Step: {baseline_step:,}"
    draw.text(
        (left_cx, h - 82), step_text,
        fill="#333333", font=font_info, anchor="mt",
    )

    # Progress bar
    bar_x = left_cx - bar_width // 2
    bar_y = h - 56
    progress = min(baseline_step / FLASHMD_TOTAL_STEPS, 1.0)
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
        outline="#CCCCCC", width=1,
    )
    if progress > 0:
        fill_w = max(1, int(bar_width * progress))
        draw.rectangle(
            [bar_x, bar_y, bar_x + fill_w, bar_y + bar_height],
            fill=CORAL,
        )
    pct_text = f"{progress * 100:.0f}%"
    draw.text(
        (bar_x + bar_width + 8, bar_y - 2), pct_text,
        fill="#666666", font=font_small,
    )

    # === Right: FlashMD ===
    right_cx = half_w + half_w // 2
    draw.text(
        (right_cx, 10), "FlashMD",
        fill=BLUE, font=font_title, anchor="mt",
    )

    step_text = f"Step: {flash_step:,}"
    draw.text(
        (right_cx, h - 82), step_text,
        fill="#333333", font=font_info, anchor="mt",
    )

    bar_x = right_cx - bar_width // 2
    bar_y = h - 56
    progress = min(flash_step / FLASHMD_TOTAL_STEPS, 1.0)
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
        outline="#CCCCCC", width=1,
    )
    if progress > 0:
        fill_w = max(1, int(bar_width * progress))
        draw.rectangle(
            [bar_x, bar_y, bar_x + fill_w, bar_y + bar_height],
            fill=BLUE,
        )
    pct_text = f"{progress * 100:.0f}%"
    draw.text(
        (bar_x + bar_width + 8, bar_y - 2), pct_text,
        fill="#666666", font=font_small,
    )

    # Centered wall-clock speed label + replica count
    draw.text(
        (half_w, h - 28),
        f"128 replicas  |  {gif_speedup:.0f}x wall-clock speed",
        fill="#999999", font=font_small, anchor="mt",
    )

    # Divider line
    draw.line(
        [(half_w, 42), (half_w, h - 90)],
        fill="#E0E0E0", width=1,
    )

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render README GIF comparing CGSchNet vs FlashMD"
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/gif_outputs",
        help="Directory containing gif_flashmd_coords_*.npy files",
    )
    parser.add_argument(
        "--prefix",
        default="gif_flashmd",
        help="Filename prefix for the npy files",
    )
    parser.add_argument(
        "--pdb",
        default="/home/pingzhi/efficient-schnet/mlcg/data/"
        "simulating_a_trained_cg_model/1enh_5beads.pdb",
        help="PDB file for CA indices",
    )
    parser.add_argument(
        "--output",
        default="static/flashmd-demo.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--mol-index",
        type=int,
        default=0,
        help="Which molecule (replica) to visualize",
    )
    args = parser.parse_args()

    # Load data
    ca_indices = get_ca_indices(args.pdb)
    print(f"Found {len(ca_indices)} CA atoms")

    all_coords = load_trajectory(args.data_dir, prefix=args.prefix)

    # Pick main molecule and ghost replicas
    n_replicas = all_coords.shape[0]
    traj = all_coords[args.mol_index]  # (total_frames, 269, 3)
    n_traj_frames = traj.shape[0]
    print(f"Trajectory: {n_traj_frames} frames for mol {args.mol_index}")

    # Select ghost replicas (evenly spaced, excluding the main one)
    N_GHOSTS = 6
    ghost_indices = []
    for gi in np.linspace(0, n_replicas - 1, N_GHOSTS + 2).astype(int):
        if gi != args.mol_index and gi < n_replicas:
            ghost_indices.append(gi)
    ghost_indices = ghost_indices[:N_GHOSTS]
    ghost_trajs = [all_coords[gi] for gi in ghost_indices]
    print(f"Ghost replicas: {ghost_indices}")

    # The trajectory covers some number of steps.
    # We treat the full trajectory as the FlashMD side (100k steps).
    # The baseline side uses a subset: only 1/SPEEDUP of the frames.
    # FlashMD: frame indices span the full trajectory
    # Baseline: frame indices span only the first 1/SPEEDUP fraction
    flash_frame_indices = np.linspace(
        0, n_traj_frames - 1, N_FRAMES
    ).astype(int)

    baseline_max_frame = int(n_traj_frames / SPEEDUP)
    baseline_max_frame = min(baseline_max_frame, n_traj_frames - 1)
    baseline_frame_indices = np.linspace(
        0, baseline_max_frame, N_FRAMES
    ).astype(int)

    # Step counts for display
    flash_steps = np.linspace(0, FLASHMD_TOTAL_STEPS, N_FRAMES).astype(int)
    baseline_steps = np.linspace(
        0, BASELINE_TOTAL_STEPS, N_FRAMES
    ).astype(int)

    # Azimuth rotation: gentle sweep for 3D feel
    # azim=0 shows the protein lying horizontally (good for wide panels)
    azimuths = np.linspace(0, 30, N_FRAMES)
    elev = 15

    cmap = make_nc_colormap()

    # Compute a fixed view range across all used frames for stable framing
    print("Computing view range...")
    all_used_frames = set(flash_frame_indices.tolist())
    all_used_frames.update(baseline_frame_indices.tolist())
    max_spread = 0
    for fi in all_used_frames:
        ca = center_coordinates(traj[fi][ca_indices])
        spread = np.abs(ca).max()
        max_spread = max(max_spread, spread)
    view_range = max_spread * 1.1
    print(f"Fixed view range: {view_range:.1f}")

    # Render all frames
    frames = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for i in range(N_FRAMES):
        if i % 10 == 0:
            print(f"Rendering frame {i}/{N_FRAMES}...")

        fi = flash_frame_indices[i]
        bi = baseline_frame_indices[i]

        # Ghost positions for this frame
        ghost_flash = [gt[fi] for gt in ghost_trajs]
        ghost_baseline = [gt[bi] for gt in ghost_trajs]

        img = render_frame(
            traj[fi], traj[bi],
            ca_indices, cmap, azimuths[i],
            flash_steps[i], baseline_steps[i],
            view_range=view_range, elev=elev,
            ghost_flash_positions=ghost_flash,
            ghost_baseline_positions=ghost_baseline,
        )
        img = add_overlays(
            img, flash_steps[i], baseline_steps[i], i, N_FRAMES,
        )
        frames.append(img)

    # Assemble GIF
    print(f"Assembling GIF ({len(frames)} frames, {FPS} fps)...")
    frame_duration_ms = int(1000 / FPS)

    # Quantize to 256 colors for smaller file size
    frames_q = [f.quantize(colors=256, method=2) for f in frames]

    frames_q[0].save(
        args.output,
        save_all=True,
        append_images=frames_q[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Saved: {args.output} ({file_size:.1f} MB)")

    if file_size > 10:
        print(
            "WARNING: GIF > 10 MB. Consider reducing resolution "
            "or frame count."
        )


if __name__ == "__main__":
    main()
