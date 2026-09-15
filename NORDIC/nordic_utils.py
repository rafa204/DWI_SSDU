import numpy as np
import time
#import cupy as cp
from numba import njit
from itertools import product
import sys

def get_noise_thresh(x, y, n = 10, xp = np, sigma = 1/np.sqrt(2)):
    noise_real = xp.random.normal(size = (n, x, y)) * sigma
    noise_imag = xp.random.normal(size = (n, x, y)) * sigma
    noise = noise_real + 1j*noise_imag
    U, S, VH = xp.linalg.svd(noise, full_matrices = False)

    mean_thresh = xp.mean(S[:,0])
    return mean_thresh

def denoise_patches_coils(patches, k, t, stride=1):
    kz, kx, ky = k[0], k[1], k[2]
    nq, nc, nz, nx, ny = patches.shape[:5]

    pz = kz // 2
    px = kx // 2
    py = ky // 2

    NZ = nz + 2 * pz
    NX = nx + 2 * px
    NY = ny + 2 * py

    M = kz * kx * ky  # Spatial size per patch
    N = nq * nc           # Number of queries

    denoised_vol = np.zeros((nq, nc, NZ, NX, NY), dtype=np.complex64)
    indx_tracker = np.zeros((nq, nc, NZ, NX, NY), dtype=np.complex64)

    c_indices = list(range(0, nx, stride))
    d_indices = list(range(0, ny, stride))
    nc_c = len(c_indices)
    nd = len(d_indices)
    batch_size = nc_c * nd
    t_val = float(t)

    for b in range(0, nz, stride):
        patch_slice = np.copy(patches[:, :, b, 0:nx:stride, 0:ny:stride, :, :, :]).astype(np.complex64)
        patch_slice = patch_slice.transpose(2, 3, 4, 5, 6, 0, 1)
        patch_slice = patch_slice.reshape(batch_size, M, N)


        U, S, Vh = np.linalg.svd(patch_slice, full_matrices=False)
        S = np.where(S < t_val, np.zeros_like(S), S)
        US = U * S[:, None, ...]
        A = US @ Vh

        A = np.swapaxes(A, -2, -1).reshape(nc_c, nd, nq, nc, kz, kx, ky)

        for ic, c in enumerate(c_indices):
            for id_, d in enumerate(d_indices):
                denoised_vol[:, :, b:b+kz, c:c+kx, d:d+ky] += A[ic, id_]
                indx_tracker[:, :, b:b+kz, c:c+kx, d:d+ky] += 1.0

    # Division and exact cropping per original bounds
    indx_tracker[indx_tracker <= 1e-8]  = 1
    denoised_vol /= indx_tracker
    x = denoised_vol[:, :, pz:NZ-pz, px:NX-px, py:NY-py]

    return x


'''

def denoise_patches(patches, k, t, stride=1):
    kz, kx, ky = k[0], k[1], k[2]
    nq, nc, nz, nx, ny = patches.shape[:5]

    pz = kz // 2
    px = kx // 2
    py = ky // 2

    NZ = nz + 2 * pz
    NX = nx + 2 * px
    NY = ny + 2 * py

    M = kz * kx * ky  # Spatial size per patch
    N = nq           # Number of queries

    denoised_vol = np.zeros((nq, nc, NZ, NX, NY), dtype=np.complex64)
    indx_tracker = np.zeros((nq, nc, NZ, NX, NY), dtype=np.uint8)

    t_val = float(t)

    patches = patches[:, :, 0:nz:stride, 0:nx:stride, 0:ny:stride, :, :, :]
    s = patches.shape

    batch_div = 10 
    stack_size = np.prod(s[1:5])
    batch_size = s[]

    sys.exit(0)

    for b in range(n_batches):

        patch_slice = np.copy(patches[:, a, b, 0:nx:stride, 0:ny:stride, :, :, :]).astype(np.complex64)
        patch_slice = patch_slice.transpose(1, 2, 3, 4, 5, 0).reshape(batch_size, M, N)

        print(f"SVD INPUT SHAPE: ", patch_slice.shape)
        U, S, Vh = np.linalg.svd(patch_slice, full_matrices=False)
        S = np.where(S < t_val, np.zeros_like(S), S)
        US = U * S[:, None, ...]
        A = US @ Vh

        A = np.swapaxes(A, -2, -1).reshape(nc_c, nd, nq, kz, kx, ky)

        for ic, c in enumerate(c_indices):
            for id_, d in enumerate(d_indices):
                denoised_vol[:, a, b:b+kz, c:c+kx, d:d+ky] += A[ic, id_]
                indx_tracker[:, a, b:b+kz, c:c+kx, d:d+ky] += 1.0

    # Division and exact cropping per original bounds
    indx_tracker[indx_tracker <= 1e-8]  = 1
    denoised_vol /= indx_tracker
    x = denoised_vol[:, :, pz:NZ-pz, px:NX-px, py:NY-py]

    return x
'''

import os
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

@torch.no_grad()
def denoise_patches_par(patches, k, t, stride=1, max_workers=32):
    kz, kx, ky = k[0], k[1], k[2]
    nq, nc, nz, nx, ny = patches.shape[:5]

    pz, px, py = kz // 2, kx // 2, ky // 2
    NZ, NX, NY = nz + 2 * pz, nx + 2 * px, ny + 2 * py

    M = kz * kx * ky  # Spatial size per patch
    N = nq           # Number of queries
    t_val = float(t)

    # Accumulators allocated directly in CPU RAM
    denoised_vol = torch.zeros((nq, nc, NZ, NX, NY), dtype=torch.complex64)
    indx_tracker = torch.zeros((nq, nc, NZ, NX, NY), dtype=torch.int32)

    x_indices = list(range(0, nx, stride))
    y_indices = list(range(0, ny, stride))
    batch_size = len(x_indices) * len(y_indices) 

    # Prevent OpenMP thread contention by dividing hardware threads per worker
    total_cores = os.cpu_count() or 4
    print("TOTAL CORES: ", total_cores)
    threads_per_worker = max(1, total_cores // max_workers)
    print("Threads per worker: ", threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    accumulation_lock = threading.Lock()

    def process_zc_slice(z, c):
        print("COIL ", c)
        # 1. Slice and shape patch batch in RAM
        patch_slice = np.copy(patches[:, c, z, 0:nx:stride, 0:ny:stride, :, :, :]).astype(np.complex64)
        patch_slice = patch_slice.transpose(1, 2, 3, 4, 5, 0).reshape(batch_size, M, N)
        patch_slice_cpu = torch.from_numpy(patch_slice)

        # 2. LAPACK CPU SVD (uses multithreaded MKL/OpenMP 'gesdd')
        U, S, Vh = torch.linalg.svd(patch_slice_cpu, full_matrices=False)
        S = torch.where(S < t_val, torch.zeros_like(S), S)
        US = U * S.unsqueeze(-2)
        A = US @ Vh

        A = torch.swapaxes(A, -2, -1).reshape(len(x_indices), len(y_indices), nq, kz, kx, ky)

        # 3. Thread-safe accumulation into shared memory array
        with accumulation_lock:
            for ix, x in enumerate(x_indices):
                for iy, y in enumerate(y_indices):
                    denoised_vol[:, c, z:z+kz, x:x+kx, y:y+ky] += A[ix, iy]
                    indx_tracker[:, c, z:z+kz, x:x+kx, y:y+ky] += 1

    # Dispatch tasks across CPU worker pool
    task_pairs = [(z, c) for z in range(0, nz, stride) for c in range(nc)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_zc_slice, z, c) for z, c in task_pairs]
        for future in futures:
            future.result()

    # Final normalization and cropping
    indx_tracker[indx_tracker == 0] = 1
    denoised_vol /= indx_tracker
    #x = denoised_vol[:, :, pz:NZ-pz, px:NX-px, py:NY-py]
    x = denoised_vol

    return x.numpy()


import numpy as np

def denoise_patches(patches, k, t, stride=1):
    kz, kx, ky = k[0], k[1], k[2]
    nq, nc, nz, nx, ny = patches.shape[:5]

    pz = kz // 2
    px = kx // 2
    py = ky // 2

    NZ = nz + 2 * pz
    NX = nx + 2 * px
    NY = ny + 2 * py

    M = kz * kx * ky  # Spatial size per patch
    N = nq           # Number of queries

    denoised_vol = np.zeros((nq, nc, NZ, NX, NY), dtype=np.complex64)
    indx_tracker = np.zeros((nq, nc, NZ, NX, NY), dtype=np.complex64)

    x_indices = list(range(0, nx, stride))
    y_indices = list(range(0, ny, stride))

    batch_size = len(x_indices) * len(y_indices) 
    t_val = float(t)

    print(nz)
    print(patches.shape)

    for z in range(0, nz, stride):
        z = 3
        for c in range(nc):

            print(f"COIL {c} / {nc}")
            print(f"Z VALUE {z} / {nz}")
            patch_slice = np.copy(patches[:, c, z, 0:nx:stride, 0:ny:stride, :, :, :]).astype(np.complex64)

            patch_slice = patch_slice.transpose(1, 2, 3, 4, 5, 0)
            print(f"pathc INPUT SHAPE: ", patch_slice.shape)
            patch_slice = patch_slice.reshape(len(x_indices), len(y_indices), M, N)

            print(f"SVD INPUT SHAPE: ", patch_slice.shape)
            U, S, Vh = np.linalg.svd(patch_slice, full_matrices=False)

            return S, t, patch_slice

            S = np.where(S < t_val, np.zeros_like(S), S)
            US = U * S[:, None, ...]
            A = US @ Vh

            A = np.swapaxes(A, -2, -1).reshape(len(x_indices), len(y_indices), nq, kz, kx, ky)

            for ix, x in enumerate(x_indices):
                for iy, y in enumerate(y_indices):
                    denoised_vol[:, c, z:z+kz, x:x+kx, y:y+ky] += A[ix, iy]
                    indx_tracker[:, c, z:z+kz, x:x+kx, y:y+ky] += 1

    # Division and exact cropping per original bounds
    indx_tracker[indx_tracker == 0]  = 1
    denoised_vol /= indx_tracker
    x = denoised_vol[:, :, pz:NZ-pz, px:NX-px, py:NY-py]

    return x




'''
    kz, kx, ky = k[0], k[1], k[2]
    nq, nc, nz, nx, ny = patches.shape[:5]

    pz = kz // 2
    px = kx // 2
    py = ky // 2

    NZ = nz + 2 * pz
    NX = nx + 2 * px
    NY = ny + 2 * py

    M = kz * kx * ky  # Spatial size per patch
    N = nq           # Number of queries

    # Host accumulators on CPU to save GPU memory
    denoised_vol = torch.zeros((nq, nc, NZ, NX, NY), dtype=torch.complex64)
    indx_tracker = torch.zeros((nq, nc, NZ, NX, NY), dtype=torch.complex64)

    c_indices = list(range(0, nx, stride))
    d_indices = list(range(0, ny, stride))
    nc_c = len(c_indices)
    nd = len(d_indices)
    batch_size = nc_c * nd
    t_val = float(t)

    for a in range(nc):
        for b in range(0, nz, stride):
            # 1. Sample all (c, d) patches for the current (a, b) slice
            #    Shape: (nq, nc_c, nd, kz, kx, ky)
            patch_slice = np.copy(patches[:, a, b, 0:nx:stride, 0:ny:stride, :, :, :])

            # 2. Transfer the (c, d) batch slice to GPU
            if isinstance(patch_slice, np.ndarray):
                gpu_chunk = torch.from_numpy(patch_slice).to(device=device, dtype=torch.complex64)
            else:
                gpu_chunk = patch_slice.to(device=device, dtype=torch.complex64)

            # 3. Permute and reshape to matrix batch:
            #    (nq, nc_c, nd, kz, kx, ky) -> (nc_c, nd, kz, kx, ky, nq) -> (batch_size, M, N)
            gpu_chunk = gpu_chunk.permute(1, 2, 3, 4, 5, 0).reshape(batch_size, M, N)

            # 4. Batched SVD on GPU across all combined (c, d) patches
            #    U:  (batch_size, M, K)
            #    S:  (batch_size, K)
            #    Vh: (batch_size, K, N)   where K = min(M, N)
            U, S, Vh = torch.linalg.svd(gpu_chunk, full_matrices=False)

            # 5. Singular Value Thresholding
            S = torch.where(S < t_val, torch.zeros_like(S), S)

            # 6. Reconstruct A = (U * S) @ Vh using batch matrix multiplication
            #    (batch_size, M, K) * (batch_size, 1, K) -> (batch_size, M, K)
            US = U * S.unsqueeze(1)
            #    (batch_size, M, K) @ (batch_size, K, N) -> (batch_size, M, N)
            A = torch.bmm(US, Vh)

            # 7. Reshape back to patch layout
            #    (batch_size, M, N) -> transpose last dims -> (batch_size, N, M) -> (nc_c, nd, nq, kz, kx, ky)
            A_reshaped = A.transpose(-2, -1).reshape(nc_c, nd, nq, kz, kx, ky)

            # 8. Transfer batch back to CPU for accumulation
            A_cpu = A_reshaped.cpu()

            # 9. Accumulate on CPU over c and d indices
            for ic, c in enumerate(c_indices):
                for id_, d in enumerate(d_indices):
                    denoised_vol[:, a, b:b+kz, c:c+kx, d:d+ky] += A_cpu[ic, id_]
                    indx_tracker[:, a, b:b+kz, c:c+kx, d:d+ky] += 1.0

    # Division and exact cropping per original bounds
    denoised_vol /= indx_tracker
    x = denoised_vol[:, :, pz:NZ-pz, px:NX-px, py:NY-py]

    return x.numpy() if isinstance(patches, np.ndarray) else x
'''

'''
#@njit
def denoise_patches(patches, k, t, stride = 1):

    kz = k[0]
    kx = k[1]
    ky = k[2]
    nq = patches.shape[0]
    nc = patches.shape[1]
    nz = patches.shape[2]
    nx = patches.shape[3]
    ny = patches.shape[4]

    NZ = nz + 2*(k[0]//2)
    NX = nx + 2*(k[1]//2)
    NY = ny + 2*(k[2]//2)
    denoised_vol = np.zeros((nq, nc, NZ, NX, NY), dtype = patches.dtype)
    indx_tracker = np.zeros((nq, nc, NZ, NX, NY), dtype = patches.dtype)

    for a in range(nc):
        for b in range(0,nz,stride):
            for c in range(0,nx,stride):
                for d in range(0,ny,stride):
                    input_patches = np.ascontiguousarray(patches[:,a,b,c,d,:,:,:])
                    input_patches = input_patches.reshape(nq, kx*ky*kz).T
                    U, S, VH = np.linalg.svd(input_patches, full_matrices = False)
                    S[S < t] = 0
                    A = (U * S[..., None, :]) @ VH
                    A = A.T.reshape(nq, kz, kx, ky)

                    denoised_vol[:, a, b:b+k[0], c:c+k[1], d:d+k[2]] += A
                    indx_tracker[:, a, b:b+k[0], c:c+k[1], d:d+k[2]] += 1

    denoised_vol /= indx_tracker
    pz = k[0]//2
    px = k[1]//2
    py = k[2]//2
    x = denoised_vol[:,:, pz : NZ-pz, px : NX-px, py : NY-py]

    return x

'''

@njit
def patch_average_numba(patches):
    # Input: N-D patches (..., nz, nx, ny, kz, kx, ky)
    nz, nx, ny, kz, kx, ky = patches.shape[-6:]
    batch_shape = patches.shape[:-6]

    # Compute total number of flattened batch items
    n_batch = 1
    for dim in batch_shape:
        n_batch *= dim

    # Calculate original padded dimensions
    Nz = nz + kz - 1
    Nx = nx + kx - 1
    Ny = ny + ky - 1

    # Flatten leading dimensions for linear iteration in Numba
    patches_flat = np.ascontiguousarray(patches).reshape((n_batch, nz, nx, ny, kz, kx, ky))

    # Initialize volume and overlap counts
    volume = np.zeros((n_batch, Nz, Nx, Ny), dtype=np.float64)
    counts = np.zeros((Nz, Nx, Ny), dtype=np.float64)

    # 1. Calculate overlap counts once (spatial geometry is identical across batches)
    for z in range(nz):
        for x in range(nx):
            for y in range(ny):
                for dz in range(kz):
                    for dx in range(kx):
                        for dy in range(ky):
                            counts[z + dz, x + dx, y + dy] += 1.0

    # 2. Accumulate patch values into the reconstructed volume
    for b in range(n_batch):
        for z in range(nz):
            for x in range(nx):
                for y in range(ny):
                    for dz in range(kz):
                        for dx in range(kx):
                            for dy in range(ky):
                                volume[b, z + dz, x + dx, y + dy] += patches_flat[b, z, x, y, dz, dx, dy]

    # 3. Divide by counts
    for b in range(n_batch):
        for z in range(Nz):
            for x in range(Nx):
                for y in range(Ny):
                    if counts[z, x, y] > 0:
                        volume[b, z, x, y] /= counts[z, x, y]

    # 4. Crop padding and restore full preceding batch dimensions
    pz = kz // 2
    px = kx // 2
    py = ky // 2

    cropped_volume = volume[:, pz : Nz - pz, px : Nx - px, py : Ny - py]
    
    out_shape = batch_shape + (Nz - 2 * pz, Nx - 2 * px, Ny - 2 * py)
    return np.ascontiguousarray(cropped_volume).reshape(out_shape)


#from numba import njit, prange
#
#@njit(parallel=True, fastmath=True)
#def _llr_numba_core(flat_patches, threshold, out):
#    num_matrices = flat_patches.shape[0]
#    
#    for i in prange(num_matrices):
#        # Pass False positionally to satisfy Numba
#        u, s, vh = np.linalg.svd(flat_patches[i], False)
#        
#        m, k = u.shape
#        # In-place scaling of U columns to avoid broadcasting typing issues
#        for col in range(k):
#            val = s[col] if s[col] >= threshold else 0.0
#            for row in range(m):
#                u[row, col] *= val
#                
#        # Matrix multiplication (@) is supported in Numba nopython mode
#        out[i] = u @ vh
#
#def llr(img_patches, threshold):
#    # Ensure input is a CPU NumPy array and C-contiguous
#    patches = np.ascontiguousarray(img_patches)
#    orig_shape = patches.shape
#    M, N = orig_shape[-2], orig_shape[-1]
#    
#    flat_patches = patches.reshape(-1, M, N)
#    out = np.empty_like(flat_patches)
#    
#    _llr_numba_core(flat_patches, float(threshold), out)
#    return out.reshape(orig_shape)

def llr(img_patches, threshold, xp = np):
    img_patches = xp.asarray(img_patches)
    U, S, VH = xp.linalg.svd(img_patches, full_matrices = False)
    S_thresh = xp.copy(S)
    S_thresh[S < threshold] = 0
    A = (U * S_thresh[..., None, :]) @ VH

    return A

'''
import torch

def llr(img_patches, threshold, target_vram_usage=0.5):
    """
    Fastest possible LLR implementation for massive stacks.
    Streams micro-batches to GPU to guarantee zero Out-Of-Memory (OOM) errors.
    
    :param img_patches: NumPy array or PyTorch Tensor (Real or Complex)
    :param threshold: Cutoff threshold for singular values
    :param target_vram_usage: Fraction of FREE GPU memory to use per chunk (0.5 = 50%)
    """
    # 1. Convert input array to CPU Tensor with Pinned Memory for ultra-fast PCIe transfers
    if isinstance(img_patches, np.ndarray):
        host_tensor = torch.from_numpy(img_patches)
    else:
        host_tensor = img_patches.cpu()

    orig_shape = host_tensor.shape
    M, N = orig_shape[-2], orig_shape[-1]
    
    # Reshape to (Total_Matrices, M, N) without memory duplication
    flat_patches = host_tensor.reshape(-1, M, N)
    total_matrices = flat_patches.shape[0]
    
    # Pin memory to enable asynchronous non-blocking CPU -> GPU transfers
    if not flat_patches.is_pinned():
        flat_patches = flat_patches.pin_memory()
        
    out_flat = torch.empty_like(flat_patches)

    # 2. Calculate optimal batch size based on currently available VRAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_vram, _ = torch.cuda.mem_get_info(device)
    
    # Calculate bytes per matrix across inputs + U, S, Vh intermediate tensors (~6x input size)
    bytes_per_element = 8 if flat_patches.dtype in (torch.complex64, torch.float32) else 16
    bytes_per_matrix = M * N * bytes_per_element * 6
    
    # Target batch size to use only a safe fraction of free VRAM
    usable_vram = free_vram * target_vram_usage
    batch_size = int(usable_vram // bytes_per_matrix)
    batch_size = max(1, min(batch_size, total_matrices))

    # 3. Stream micro-batches through GPU
    # inference_mode completely disables gradient engine to save huge amounts of RAM
    with torch.inference_mode():
        for start_idx in range(0, total_matrices, batch_size):
            end_idx = min(start_idx + batch_size, total_matrices)
            
            # Non-blocking transfer to GPU
            chunk = flat_patches[start_idx:end_idx].to(device, non_blocking=True)
            
            # Batched GPU SVD
            U, S, Vh = torch.linalg.svd(chunk, full_matrices=False)
            
            # Threshold singular values
            S_thresh = torch.where(S >= threshold, S, 0.0)
            
            # Reconstruct: (U * S) @ Vh
            A_chunk = (U * S_thresh.unsqueeze(-2)) @ Vh
            
            # Transfer result back to host CPU buffer
            out_flat[start_idx:end_idx].copy_(A_chunk.cpu(), non_blocking=True)

    # Synchronize GPU streams before returning final array
    torch.cuda.synchronize()
    
    final_out = out_flat.reshape(orig_shape)
    return final_out.numpy() if isinstance(img_patches, np.ndarray) else final_out

'''