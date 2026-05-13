"""
Standalone script to generate the three new figures for the paper revision.

For v2 (GFRP AP): loads correlation_matrix.pkl directly (~700 MB, manageable).
For v3 (GFRP flux): reads from existing processed video files to avoid loading
the full correlation pkl (~4-8 GB).

Outputs written to figures/:
  gfrp_ap_fft_phase_defect_vs_sound.png
  gfrp_flux_phase_map_bin10.png
  gfrp_flux_disc_sensitivity.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_V2 = os.path.join(PROJECT_ROOT, 'data', 'v2')
DATA_V3 = os.path.join(PROJECT_ROOT, 'data', 'v3')
FIGURES = os.path.join(PROJECT_ROOT, 'figures')


def load_pkl(path):
    print(f'Loading {path} ...')
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_video_pixel(video_path, row, col):
    """Read one pixel's intensity across all frames of a video."""
    cap = cv.VideoCapture(video_path)
    values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        values.append(float(gray[row, col]))
    cap.release()
    return np.array(values)


def read_video_frame(video_path, frame_index):
    """Read a single frame from a video as a float32 2D array."""
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Could not read frame {frame_index} from {video_path}')
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)


def generate_gfrp_ap_fft_phase():
    """FFT phase of GFRP AP correlation main lobe: defect vs sound."""
    CORRMATRIX = load_pkl(os.path.join(DATA_V2, 'correlation_matrix.pkl'))
    cropped = CORRMATRIX[400:610, :, :]  # main lobe: 210 frames

    # Vectorised FFT; only show positive-frequency half (bins 0..104)
    phase_3d = np.angle(np.fft.fft(cropped, axis=0))
    half = 105

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phase_3d[:half, 115, 115], label='defect (115, 115)')
    ax.plot(phase_3d[:half, 230, 230], label='defect (230, 230)')
    ax.plot(phase_3d[:half, 175, 175], label='sound (175, 175)', linestyle='--', color='red')
    ax.set_xlabel('Frequency bin $k$')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('FFT Phase of Correlation Main Lobe — GFRP AP Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, 'gfrp_ap_fft_phase_defect_vs_sound.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')


def generate_gfrp_flux_phase_map(bin_k=10):
    """Spatial 2D phase map at frequency bin bin_k, read from existing video."""
    video_path = os.path.join(DATA_V3, 'gfrp_flux_fft_phase_mainlobe_disc160.mp4')
    phase_map = read_video_frame(video_path, bin_k)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(phase_map, cmap='RdBu', aspect='auto')
    fig.colorbar(im, ax=ax, label='Phase (normalised, 0–255)')
    ax.set_title(f'FFT Phase Map — GFRP Flux — Frequency Bin {bin_k}')
    ax.set_xlabel('Column (pixel)')
    ax.set_ylabel('Row (pixel)')
    fig.tight_layout()
    out = os.path.join(FIGURES, f'gfrp_flux_phase_map_bin{bin_k}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')


def generate_disc_sensitivity():
    """
    FFT phase vs normalised frequency for defect pixel (115, 230) at three
    zero-padding levels. Phase values read from the existing per-N_d videos.
    """
    disc_videos = {
        160:  os.path.join(DATA_V3, 'gfrp_flux_fft_phase_mainlobe_disc160.mp4'),
        640:  os.path.join(DATA_V3, 'gfrp_flux_fft_phase_mainlobe_disc640.mp4'),
        1024: os.path.join(DATA_V3, 'gfrp_flux_fft_phase_mainlobe_disc1024.mp4'),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for nd, path in disc_videos.items():
        phase_series = read_video_pixel(path, row=115, col=230)
        half = len(phase_series) // 2
        freqs = np.linspace(0, 1, len(phase_series), endpoint=False)
        ax.plot(freqs[:half], phase_series[:half], label=f'$N_d = {nd}$')
        print(f'  N_d={nd}: {len(phase_series)} frames read')

    ax.set_xlabel('Normalised frequency')
    ax.set_ylabel('Phase (normalised, 0–255)')
    ax.set_title('Discretisation Sensitivity — Defect Pixel (115, 230), GFRP Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, 'gfrp_flux_disc_sensitivity.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    generate_gfrp_ap_fft_phase()
    generate_gfrp_flux_phase_map(bin_k=10)
    generate_disc_sensitivity()
    print('Done.')
