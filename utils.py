import numpy as np
import matplotlib.pyplot as plt
import torch

def ifft(img, dims=[-2,-1]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(img, dim=dims), dim=dims, norm = 'ortho'), dim=dims)
    
def fft(kspace, dims=[-2,-1]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(kspace, dim=dims), dim=dims, norm = 'ortho'), dim=dims)

def rssq(x, xp = torch):
    return xp.sqrt(xp.sum(xp.abs(x)**2, -3))

def shift_fov(x, fov_shifts, xp = torch):
    '''
    Apply or remove CAIPI shift from multi-band MRI data
    x should have:
    Multiband slice dimension = -3
    PE dimension = -2
    '''
    slices = []
    for i in range(x.shape[-3]):
        shift = xp.ceil(fov_shifts[i] * x.shape[-2])
        slice_i = xp.roll(x[..., i, :, :], shift, axis=-2)
        slices.append(slice_i)
    return xp.stack(slices, axis=-3)

def mask_generator(nx,ny,r,coil_dim = 0, first_line = 0):
    mask = np.zeros((r,ny))
    mask[0,:] = 1
    mask = np.tile(mask,(nx//r+1, 1))
    mask = np.roll(mask, first_line, 0)
    mask = mask[:nx,:ny]
    if coil_dim > 0:
        mask = np.tile(mask[np.newaxis,:,:], (coil_dim,1,1))
    return mask


def quick_plot(
    img,
    gs=None,
    no_axis=True,
    figs=(15, 4),
    aspect=1,
    title=None,
    title_list=None,
    path=None,
    cmap="gray",
    fs=12,
    scale=False,
    colorbar=False,
    vrange = None
):
    # Track whether original input was a list of images
    is_list = isinstance(img, list)

    # Force scale to True if a shared colorbar is requested
    if colorbar:
        scale = True

    if gs is None:
        if is_list:
            gs = (1, len(img))
        else:
            gs = (1, 1)
            img = [img]

    m, n = gs

    # Determine common color scale if requested
    if scale:
        vmin = min(im.min() for im in img)
        vmax = max(im.max() for im in img)
    else:
        vmin = vmax = None

    if vrange is not None:
        vmin, vmax = vrange

    fig, ax = plt.subplots(m, n, figsize=figs)

    # Variable to store the mappable image object for the colorbar
    im_mappable = None

    for i, current_ax in enumerate(fig.get_axes()):
        im_mappable = current_ax.imshow(
            img[i].T, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax
        )

        # Apply subtitles only if original input was a list of images
        if is_list and title_list and i < len(title_list):
            current_ax.set_title(title_list[i], fontsize=fs)

        if no_axis:
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            current_ax.tick_params(axis="x", length=0)
            current_ax.tick_params(axis="y", length=0)
            current_ax.axis("off")

    # Add the shared colorbar to the figure, scaling it to the axes
    if colorbar and im_mappable is not None:
        fig.colorbar(im_mappable, ax=fig.get_axes())

    if title is not None:
        fig.suptitle(title, fontsize=fs)

    if not colorbar:
        fig.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path)