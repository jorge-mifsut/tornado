import torch
import matplotlib.pyplot as plt


# Per-variable colour maps for the ERA5 channels [t2m, u10, v10, z500]. The two
# wind components are signed and get a diverging map centred on zero; temperature
# and geopotential are read as magnitudes.
ERA5_CMAPS = {"t2m": "RdBu_r", "u10": "PuOr_r", "v10": "PuOr_r", "z500": "viridis"}
ERA5_DIVERGING = {"u10", "v10"}


def plot_result_map(u, rec=None, n_t=5, path=None, lat=None, lon=None,
                    var_names=None, times=None, max_channels=4):
    '''
    Plot ERA5-style results on a world map, with coastlines overlaid.

    One row per (timestep, channel); ground truth, prediction and their difference
    side by side. Colour limits are taken from the ground truth so prediction and
    truth are directly comparable, and the error panel uses a symmetric diverging
    scale so over- and under-prediction are distinguishable.

    Args:
        u (torch.Tensor): ground truth, shape (b, nt, nlat, nlon, c)
        rec (torch.Tensor): prediction, same shape, or None
        n_t (int): number of timesteps to plot
        path (str): where to save; None skips saving
        lat, lon: coordinate vectors in degrees. Defaults to an equiangular
            with-poles grid inferred from the field shape.
        var_names (list[str]): channel names, used for titles and colour maps
        times (array-like): hours since 1970-01-01 for each plotted timestep
    '''
    import numpy as np

    from common.geo import plot_map, pole_grid_lat

    u = u[0].detach().cpu().float().numpy()  # (nt, nlat, nlon, c)
    rec = rec[0].detach().cpu().float().numpy() if rec is not None else None

    nt, nlat, nlon, n_c = u.shape
    n_c = min(n_c, max_channels)
    if lat is None:
        lat = pole_grid_lat(nlat)
    if lon is None:
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    if var_names is None:
        var_names = ["t2m", "u10", "v10", "z500"][:n_c]

    n_t = max(1, min(n_t, nt))
    n_skip = max(1, nt // n_t) if n_t > 1 else 1
    steps = list(range(0, nt, n_skip))[:n_t]

    n_cols = 3 if rec is not None else 1
    n_rows = len(steps) * n_c
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 2.4 * n_rows),
                             squeeze=False)

    for ri, t_idx in enumerate(steps):
        for ci in range(n_c):
            row = ri * n_c + ci
            name = var_names[ci] if ci < len(var_names) else f"c{ci}"
            cmap = ERA5_CMAPS.get(name, "viridis")
            gt = u[t_idx, ..., ci]

            if name in ERA5_DIVERGING:
                lim = float(np.abs(gt).max())
                vmin, vmax = -lim, lim
            else:
                vmin, vmax = float(gt.min()), float(gt.max())

            stamp = ""
            if times is not None and t_idx < len(times):
                stamp = " " + str(
                    (np.datetime64("1970-01-01T00", "h")
                     + np.timedelta64(int(times[t_idx]), "h"))
                )

            im = plot_map(axes[row][0], gt, lat, lon, cmap=cmap, vmin=vmin, vmax=vmax,
                          title=f"{name} truth  t={t_idx}{stamp}")
            fig.colorbar(im, ax=axes[row][0], fraction=0.025, pad=0.01)

            if rec is not None:
                pr = rec[t_idx, ..., ci]
                im = plot_map(axes[row][1], pr, lat, lon, cmap=cmap,
                              vmin=vmin, vmax=vmax, title=f"{name} prediction")
                fig.colorbar(im, ax=axes[row][1], fraction=0.025, pad=0.01)

                err = pr - gt
                emax = float(np.abs(err).max())
                # A rollout snapshot copies the initial condition into slot 0, so
                # the first row's "error" is identically zero by construction, not
                # a perfect forecast. Say so rather than drawing a blank panel with
                # an arbitrary colour range.
                elim = emax if emax > 0 else 1.0
                err_title = (
                    f"{name} error (max |e| = {emax:.3g})"
                    if emax > 0
                    else f"{name}: initial condition (copied, no error)"
                )
                im = plot_map(axes[row][2], err, lat, lon, cmap="RdBu_r",
                              vmin=-elim, vmax=elim, title=err_title)
                fig.colorbar(im, ax=axes[row][2], fraction=0.025, pad=0.01)

    fig.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def entropy_spectrum(vorticity_batch: torch.Tensor):
    """
    Compute the azimuthally averaged entropy spectrum of 2D vorticity fields using PyTorch and einsum.

    Parameters
    ----------
    vorticity_batch : torch.Tensor
        Tensor of shape (batch, nx, ny, 1), real-valued vorticity fields.

    Returns
    -------
    spectra : torch.Tensor
        Tensor of shape (batch, k_max+1), the entropy spectrum for each sample.
    k_vals : torch.Tensor
        Tensor of shape (k_max+1,), the corresponding wavenumber bins.
    """
    batch, nx, ny, _ = vorticity_batch.shape
    device = vorticity_batch.device

    # Remove channel dimension for FFT
    vorticity = vorticity_batch[..., 0]  # (batch, nx, ny)

    # Compute FFT and power spectrum
    fft2 = torch.fft.fft2(vorticity, norm='forward')  # (batch, nx, ny)
    power = fft2.abs() ** 2  # (batch, nx, ny)

    # Frequency grid
    kx = torch.fft.fftfreq(nx, d=1.0, device=device).reshape(-1, 1).repeat(1, ny)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device).reshape(1, -1).repeat(nx, 1)
    k_mag = torch.sqrt((kx * nx) ** 2 + (ky * ny) ** 2)
    k_int = torch.round(k_mag).to(torch.int64)  # integer wavenumbers

    # Flatten spatial dimensions
    power_flat = power.reshape(batch, -1)             # (batch, nx*ny)
    k_int_flat = k_int.reshape(-1)                    # (nx*ny,)
    k_max = int(k_int_flat.max().item())
    k_vals = torch.arange(k_max + 1, device=device)   # (k_max+1,)

    # Build a mask matrix of shape (k_max+1, nx*ny): 1 if point belongs to shell k
    mask = (k_int_flat[None, :] == k_vals[:, None]).float()  # (k_max+1, nx*ny)

    # Use einsum to sum power within each shell: result is (batch, k_max+1)
    spectra = torch.einsum('bk,nk->bn', power_flat, mask)

    return spectra, k_vals


def plot_result_2d(u, rec=None, n_t=5, path=None, cmap="twilight_shifted"):
    '''
    Plot the results of a 2D PDE model.
    Args:
        u (torch.Tensor): ground truth velocity field, shape (b, nt, nx, ny, c)
        rec (torch.Tensor): predicted velocity field, shape (b, nt, nx, ny, c), can be None
        n_t (int): number of timesteps to plot
        path (str): path to save the plot, if None, the plot will not be saved
    Returns:
        None
    '''

    u = u[0, ..., 0].detach().cpu() # (nt nx ny)
    rec = rec[0, ..., 0].detach().cpu() if rec is not None else None
        
    vmin = torch.min(u)
    vmax = torch.max(u)

    if n_t == 1:
        u_downs = u # (1, nx, ny)
        n_skip = 1
    else:
        n_skip = max(1, u.shape[0] // n_t)
        u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 2, figsize=(8, 4*n_t))

        if n_t == 1:
            rec_downs = rec
            ax = [ax]
        else:
            rec_downs = rec[::n_skip]

        for j in range(2):
            for i in range(n_t):
                ax[i][j].set_axis_off()
                
                if j == 0:
                    velocity = u_downs[i] 
                else:
                    velocity = rec_downs[i]

                im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i][j].title.set_text(f'Timestep {i*n_skip}')
            ax[0][j].title.set_text(f'Ground Truth' if j == 0 else f'Prediction')
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))
        ax = [ax] if n_t == 1 else ax

        for i in range(n_t):
            ax[i].set_axis_off()
            velocity = u_downs[i] 

            im = ax[i].imshow(velocity, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[i].title.set_text(f'Timestep {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_entropy_2d(u, rec=None, n_t=5, path=None):
    '''
    Plot the results of a 2D PDE model.
    Args:
        u (torch.Tensor): ground truth velocity field, shape (b, nt, nx, ny, c)
        rec (torch.Tensor): predicted velocity field, shape (b, nt, nx, ny, c), can be None
        n_t (int): number of timesteps to plot
        path (str): path to save the plot, if None, the plot will not be saved
    Returns:
        None
    '''

    u = u[0, ..., 0].detach().cpu() # (nt nx ny)
    rec = rec[0, ..., 0].detach().cpu() if rec is not None else None

    if n_t == 1:
        u_downs = u # (1, nx, ny)
        n_skip = 1
    else:
        n_skip = max(1, u.shape[0] // n_t)
        u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))

        if n_t == 1:
            rec_downs = rec
            ax = [ax]
        else:
            rec_downs = rec[::n_skip]

        for i in range(n_t):
            sample_true = u_downs[i].unsqueeze(-1).unsqueeze(0)
            sample_pred = rec_downs[i].unsqueeze(-1).unsqueeze(0)

            entropy_true, k_vals = entropy_spectrum(sample_true)
            entropy_pred, _ = entropy_spectrum(sample_pred)

            ax[i].loglog(k_vals.cpu(), entropy_true[0].cpu(), label='Ground Truth', color='blue')
            ax[i].loglog(k_vals.cpu(), entropy_pred[0].cpu(), label='Prediction', color='orange', linestyle='--')
            ax[i].title.set_text(f'Timestep {i*n_skip}')
            ax[i].legend()
        
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))
        ax = [ax] if n_t == 1 else ax

        for i in range(n_t):
            sample_true = u_downs[i].unsqueeze(-1).unsqueeze(0)

            entropy_true, k_vals = entropy_spectrum(sample_true)

            ax[i].loglog(k_vals.cpu(), entropy_true[0].cpu(), label='Ground Truth', color='blue')
            ax[i].title.set_text(f'Timestep {i*n_skip}')
            ax[i].legend()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)