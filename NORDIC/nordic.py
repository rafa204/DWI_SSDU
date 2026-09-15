from linear_recon.linear_utils import fft, ifft, quick_plot, rssq
import importlib
import linear_recon.NORDIC.nordic_utils as u
importlib.reload(u)
from linear_recon.NORDIC.nordic_utils import *
#from scipy.signal.windows import hann as filter_fun
from scipy.signal.windows import tukey as filter_fun
import sys

def nordic(ksp, k = (1,3,3), phase_correction = True, coil_comb = False, n_monte_carlo = 10, thresh = None, stride = 1, filter_exp = 1):
    xp = np

    ksp = xp.asarray(ksp)
    nx, ny = ksp.shape[-2:]

    #Shift kspace to eliminate the phase ramp due to partial-fourier, we will add it back later
    if phase_correction:
        pf_shift = nx//2 - ny//2
        ksp = xp.roll(ksp, shift=-pf_shift, axis=-2)

    img_data = ifft(ksp, xp = xp)
    #img_data = ksp

    if coil_comb:
        img_data = img_data[..., None,:,:,:]

    if (img_data.ndim == 4):
        vol_data = img_data[:,:,None,:,:] 
    else:
        vol_data = img_data 

    l = []
    for i in [0,10,20]:
        l.append(np.angle(vol_data[i,0,0,:,:]))
    quick_plot(l, gs = (1,3), figs = (10,10), cmap = "hsv")

    #Phase correction
    if phase_correction:
        print("Phase corr")
        filter2D = filter_fun(nx, xp=xp)[None,...].T @ filter_fun(ny, xp=xp)[None,...]
        filter2D = xp.asarray(filter2D)
        filter2D = filter2D[None, None, None, ...]

        mean_phase = xp.mean(vol_data, 0, keepdims=True)
        vol_data *= xp.exp(-1j * xp.angle(mean_phase))

        quick_plot(np.angle(mean_phase[0,0,0,:,:]), cmap = "hsv")
        l = []
        for i in [0,10,20]:
            l.append(np.angle(vol_data[i,0,0,:,:]))
        quick_plot(l, gs = (1,3), figs = (10,10), cmap = "hsv")

        #Remove smooth phase
        filter = filter2D**filter_exp
        smooth_phase = ifft(fft(vol_data, xp = xp) * (filter) , xp = xp)
        #smooth_phase[:46] = 1.0 
        vol_data *= xp.exp(-1j * xp.angle(smooth_phase))

        l = []
        for i in [0,10,20]:
            l.append(np.angle(smooth_phase[i,0,0,:,:]))
        quick_plot(l, gs = (1,3), figs = (10,10), cmap = "hsv")

        l = []
        for i in [0,10,20]:
            l.append(np.angle(vol_data[i,0,0,:,:]))
        quick_plot(l, gs = (1,3), figs = (10,10), cmap = "hsv")



    nq, nc, nz, nx, ny = vol_data.shape

    quick_plot(np.abs(vol_data[0,0,0,:,:]))
    quick_plot(np.abs(filter[0,0,0,:,:]), title="Filter")
    quick_plot(np.angle(smooth_phase[0,0,0,:,:]), cmap = "hsv")
    quick_plot(np.angle(vol_data[0,0,0,:,:]), cmap = "hsv")
    sys.exit(0)

    print("Calculating noise threshold")
    if thresh is None:
        t = get_noise_thresh(k[0] * k[1] * k[2], nq, n = n_monte_carlo, xp = xp, sigma = 1/np.sqrt(2))
    else:
        t = thresh

    #vol_data = np.pad(vol_data, ((0,0), (0,0),(k[0]//2,k[0]//2),(k[1]//2,k[1]//2),(k[2]//2,k[2]//2),))
    patches = np.lib.stride_tricks.sliding_window_view(vol_data, k, axis=(-3,-2,-1))

    #denoised_img = denoise_patches(patches, k, t, stride = stride)
    denoised_img = denoise_patches_par(patches, k, t, stride = stride)


    print("DONE with LLR")

    if phase_correction:
        denoised_img *= np.exp(1j*np.angle(smooth_phase))
        denoised_img *= np.exp(1j*np.angle(mean_phase))
        denoised_ksp = np.roll(fft(denoised_img), shift=pf_shift, axis=-2)
    else:
        denoised_ksp = fft(denoised_img)

    if coil_comb:
        denoised_ksp = denoised_ksp[..., 0, :, :]

    return denoised_ksp
