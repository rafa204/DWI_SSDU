import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py

from PIL import Image
from configs import Config
from torch.utils.data import Dataset

import wandb
import io


def ifft(img, dims=[-2,-1]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(img, dim=dims), dim=dims, norm = 'ortho'), dim=dims)
    
def fft(kspace, dims=[-2,-1]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(kspace, dim=dims), dim=dims, norm = 'ortho'), dim=dims)

def fft_np(x, ax=(-2,-1), xp = np):
    return xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)

def ifft_np(x, ax=(-2,-1), xp = np):
    return xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)

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
        shift = int(fov_shifts[i] * x.shape[-2])
        slice_i = xp.roll(x[..., i, :, :], shift, -2)
        slices.append(slice_i)
    return xp.stack(slices, axis=-3)


def create_disjoint_masks(omega_mask, seed_val=1):
    '''
    Given a kspace mask, create two random disjoint subsets of it
    Uniform random undersampling
    omega_mask: (nx, ny) numpy array

    '''
    conf = Config().parse()
    samp_ratio = conf.lambda_ratio 

    temp_mask = np.abs(omega_mask) * samp_ratio

    # Set local seed and draw Bernoulli samples
    rng = np.random.default_rng(seed_val)
    lambda_mask = rng.binomial(1, temp_mask)

    theta_mask = omega_mask - lambda_mask

    theta_mask = torch.tensor(theta_mask).to(torch.complex64)
    lambda_mask = torch.tensor(lambda_mask).to(torch.complex64)

    return theta_mask, lambda_mask


class Zeroshot_dataset(Dataset):

    def __init__(self, data_path, device):
        self.device = device
        self.data_path = data_path
        self.conf = Config().parse()

        with h5py.File(data_path, 'r') as f:
            if not self.conf.qdim:
                nq = 1
            else:
                nq = 20

            ksp = f["nordic_nophasecor_ksp"][:, -nq:]
            coils = f["acs_maps"][:]
            size = f.attrs["size"]

        _, _, mb, nc, nx, ny  = size
        self.num_slabs = ksp.shape[0]
        nq = ksp.shape[1]
        self.size = (self.num_slabs, nq, mb, nc, nx, ny)

        # Move to torch
        self.ksp = torch.from_numpy(ksp).to(self.device).to(torch.complex64)
        self.coils = torch.from_numpy(coils).to(self.device).to(torch.complex64)

        phase_ramp = fft(torch.roll(fft(torch.ones((nx, ny), dtype = torch.complex64)), shifts = (ny - nx)//2, dims=-2))
        self.coils = self.coils * phase_ramp

        # Create a mask
        self.omega_mask = (self.ksp.abs().sum(dim = (0,1,2)) > 0).to(torch.complex64)
        #self.omega_mask = mask_generator(nx, ny, self.conf.ipa)
        #self.omega_mask = torch.from_numpy(self.omega_mask).to(self.device).to(torch.complex64)

    def __len__(self):
        return self.num_slabs

    def __getitem__(self, idx):

        idx = idx % self.num_slabs

        coil = self.coils[idx]
        ksp = self.ksp[idx]

        ksp = ksp * self.omega_mask
        norm_factor = ksp.abs().max()
        ksp = ksp / norm_factor

        return ksp, coil
    

class Zeroshot_dataloader(Dataset):
    """
    Class to help with multi-mask sampling of kspace data
    """
    def __init__(self, dataset, batch_size=1, device=None):
        self.conf = Config().parse()
        self.dataset = dataset
        self.device = device if device is not None else dataset.device
        self.batch_size = batch_size
        self.restart = True
        self.epoch = 0
        self.saved_recons = {}
        self.n_masks = self.conf.n_masks

        # Configure batch sizes and iteration counts
        self.n_slices = len(self.dataset) * self.n_masks
        self.n_iter = self.n_slices // self.batch_size

    def restart_count(self, epoch):
        self.epoch = epoch
        self.restart = True

    def __len__(self):
        return self.n_iter

    def _get_mask_batch(self, idx):
        slice_idx = self.slice_idx_list[idx]
        slice_data = list(self.dataset[slice_idx])
        samples = []
        for i in range(self.batch_size):
            masks = create_disjoint_masks(self.dataset.omega_mask.abs().detach().cpu().numpy(), slice_idx * self.batch_size + i)
            samples.append(slice_data + list(masks))

        return [list(items) for items in zip(*samples)]

    def __getitem__(self, idx):
        if self.restart:
            rng = np.random.default_rng(self.epoch)
            self.slice_idx_list = rng.permutation(self.n_iter)
            self.restart = False

        out_items = self._get_mask_batch(idx) 

        # Batch stack tensors to the designated device
        out_items = [
            torch.stack(items).to(self.device) if items[0] is not None else None
            for items in out_items
        ]

        return out_items
 
def L1_L2_norm(x, ref):
    """
    Compute L1-L2 norm loss
    """
    d = (2,3,4)
    L1 = torch.norm(ref-x, dim = d, p=1)/torch.norm(ref, dim = d, p=1)
    L2 = torch.norm(ref-x, dim = d, p=2)/torch.norm(ref, dim = d, p=2)

    return torch.mean(L1 + L2)


def mask_generator(nx,ny,r,coil_dim = 0, first_line = 0):
    mask = np.zeros((r,ny))
    mask[0,:] = 1
    mask = np.tile(mask,(nx//r+1, 1))
    mask = np.roll(mask, first_line, 0)
    mask = mask[:nx,:ny]
    if coil_dim > 0:
        mask = np.tile(mask[np.newaxis,:,:], (coil_dim,1,1))
    return mask


def plot_slices_zeroshot(dataloader, model, epoch = 0, n_slices = 1):
    conf = Config().parse() 
    model.eval()
    model.test = True
    dataloader.restart_count(0)
    with torch.no_grad():
        ksp, coils, theta_mask, lambda_mask = next(iter(dataloader))

        omega_mask = theta_mask + lambda_mask
        args = (ksp * omega_mask, coils, omega_mask)
        output, mu, reg_input, reg_output = model(args)

        mb = 5
        output = output[0,0].cpu().numpy()
        #output = shift_fov(output, -conf.fov_shifts)

        fig, ax = plt.subplots(1,mb,figsize=(3*mb,4))
        for j in range(mb):
            ax[j].imshow(np.abs(output[j].T), cmap = "gray")

        process_figure(fig, conf, title = f"Epoch {epoch}", label = f"Example recons")
        
        #Plot intermediate inputs and outputs once
        fig, ax = plt.subplots(conf.mb,len(reg_input),figsize=(4*conf.nb_unroll_blocks, 3*mb))
        for j in range(len(reg_input)):
            input = reg_input[j][0,0].abs().detach().cpu().numpy()

            #input = shift_fov(input.abs().detach().cpu().numpy(), -conf.fov_shifts)
            for i in range(mb):
                ax[i,j].imshow(input[i], cmap = "gray")

        process_figure(fig, conf, title = None, label = f"Network inputs")


        #Plot intermediate inputs and outputs once
        fig, ax = plt.subplots(conf.mb,len(reg_input),figsize=(4*conf.nb_unroll_blocks, 3*mb))
        for j in range(len(reg_input)):
            input = reg_output[j][0,0].abs().detach().cpu().numpy()

            #input = shift_fov(input.abs().detach().cpu().numpy(), -conf.fov_shifts)
            for i in range(mb):
                ax[i,j].imshow(input[i], cmap = "gray")

        process_figure(fig, conf, title = None, label = f"Network outputs")

    model.test = False

    return


def process_figure(fig, conf, title = None, label = "images"):
    for ax in fig.get_axes():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    conf = Config().parse()
    if conf.wandb:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        wandb.log(({label: wandb.Image(Image.open(buf))}))
    plt.close("all")
    return



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