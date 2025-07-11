#!/usr/bin/env python3.11

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from scipy.fft import fft2, ifft2 # fourier transform library

# initial wavefunction
def psi0(x, y, x0, y0, sigma=0.5, k=15*np.pi):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.exp(-1j * k * x)


def simulate_double_slit_fourier(
    L=8.0, # length of the simulation box
    Dy=0.1, # grid spacing
    Nt=300,  # number of time steps (increased for smoother animation)
    w=0.6, # slit width
    s=0.8, # slit separation
    a=0.2, # wall height
    v0=200.0, # potential barrier height
):

    Nx = int(L / Dy) + 1
    Ny = Nx
    Dt = Dy**2 / 2  
    
    x_vals = np.linspace(0, L, Nx)
    y_vals = np.linspace(0, L, Ny)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="xy")
    
    # wavenumber grid for Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Dy)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, Dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    
    # kinetic energy operator in fourier space
    K = 0.5 * (kx_grid**2 + ky_grid**2)
    
    print(f"Grid size: {Nx} x {Ny} = {Nx*Ny} points")

    # barrier geometry indices
    j0 = int((L - w) / (2 * Dy))
    j1 = int((L + w) / (2 * Dy))
    i_upper = int((L - s) / (2 * Dy) - a / Dy)
    i_lower = int((L - s) / (2 * Dy))
    i_middle = int((L + s) / (2 * Dy))
    i_top = int((L + s) / (2 * Dy) + a / Dy)

    # build walls 
    V = np.zeros((Ny, Nx), dtype=float)
    V[0:i_upper, j0:j1] = v0
    V[i_lower:i_middle, j0:j1] = v0
    V[i_top:, j0:j1] = v0

    # initialize 
    x0 = L / 6  
    y0 = L / 2
    psi = psi0(x_grid, y_grid, x0, y0)

    # split-step fourier
    snapshots = []
    save_interval = max(1, Nt // 150)  # gif - more frequent snapshots
    
    print(f"starting time evolution for {Nt} steps...")
    for n in range(Nt):
        if n % 50 == 0:
            print(f"Step {n}/{Nt}")
                
        # half step potential in real space
        psi = np.exp(-1j * V * Dt / 2) * psi
        
        # full step kinetic in Fourier space
        psi_k = fft2(psi)
        psi_k = np.exp(-1j * K * Dt) * psi_k
        psi = ifft2(psi_k)
        
        # half step potential
        psi = np.exp(-1j * V * Dt / 2) * psi
        
        if n % save_interval == 0:
            snapshot = np.abs(psi)
            
            # color the wall
            snapshot[0:i_upper, j0:j1] = 0.3  # light gray
            snapshot[i_lower:i_middle, j0:j1] = 0.3
            snapshot[i_top:, j0:j1] = 0.3
            
            snapshots.append(snapshot)

    print(f"simulation complete. saved {len(snapshots)} snapshots.")
    return snapshots, L, Dy, j0, j1, i_upper, i_lower, i_middle, i_top


def create_gif_animation(snapshots, L, Dy, j0, j1, i_upper, i_lower, i_middle, i_top, filename="double_slit_simulation.gif"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_val = max(np.max(snap) for snap in snapshots)
    
    img = ax.imshow(
        snapshots[0],
        extent=[0, L, 0, L],
        cmap="hot",  
        vmin=0,
        vmax=max_val,
        interpolation="bilinear",
    )
    
    # colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('|ψ|² Probability Density', rotation=270, labelpad=15)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Double-Slit Split-Step Fourier Simulation')
    
    def update(frame):
        img.set_data(snapshots[frame])
        return (img,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=80, 
        blit=True,
        repeat=True,
    )
    
    # gif saving
    print(f"Saving animation to {filename}...")
    writer = PillowWriter(fps=24)  # fps
    ani.save(filename, writer=writer, dpi=150)  # dpi
    print(f"Animation saved to {filename}")
    
    plt.close() 


if __name__ == "__main__":
    print("Starting simulation using Fourier method...")
    snaps, L, Dy, j0, j1, i_up, i_low, i_mid, i_top = simulate_double_slit_fourier()
    print("Creating GIF animation...")
    create_gif_animation(snaps, L, Dy, j0, j1, i_up, i_low, i_mid, i_top)
