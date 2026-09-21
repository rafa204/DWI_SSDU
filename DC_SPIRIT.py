import torch
from utils import fft, ifft

import torch.nn.functional as F

class DC_SPIRIT(torch.nn.Module):
    '''
    Data consistency class for multi-band data reconstruction with CG-SENSE
    '''
    def __init__(self, mu=0.05):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad = True)
        self.ks = (7,7)
        self.mb = 5
   
    def E(self, x, mask):
            return x.sum(1) * mask
        
    def EH(self, x, mask):
        return torch.stack([x*mask] * self.mb, axis=1)

    def EHE(self, image, mask):
        return self.EH(self.E(image, mask), mask)

    #SPIRIT KSPACE CONVOLUTION OPERATOR
    def G(self, x, filter):
        kx, ky = self.ks
        k1, k2 = kx // 2, ky // 2
        n, a, c, X, Y = x.shape
        b = filter.shape[2]

        x = x.reshape(n * a * c, X, Y).contiguous()
        x = F.pad(x, (k2, k2, k1, k1)) #, mode='circular')
        weight = filter.reshape(n * a * b, c, filter.shape[-2], filter.shape[-1]).contiguous()
        out = F.conv2d(x, weight, groups=n * a)
        return out.reshape(n, a, b, out.shape[-2], out.shape[-1]).contiguous()

    #SPIRIT KSPACE HERMITIAN CONVOLUTION OPERATOR
    def GH(self, y, filter):
        kx, ky = self.ks
        k1, k2 = kx // 2, ky // 2
        n, a, b, X, Y = y.shape
        c = filter.shape[3]

        y = y.reshape(n * a * b, X, Y).contiguous()
        y = F.pad(y, (k2, k2, k1, k1)) #, mode='circular')
        filter_adj = filter.conj().flip(-1, -2)
        weight = filter_adj.transpose(2, 3).reshape(n * a * c, b, filter.shape[-2], filter.shape[-1]).contiguous()
        out = F.conv2d(y, weight, groups=n * a)
        return out.reshape(n, a, c, out.shape[-2], out.shape[-1]).contiguous()

    def A(self, x, filter, mask):
        x = fft(x)
        x1 = self.G(x, filter) - x
        x2 = self.GH(x1, filter) - x1
        out = x2 + 1*self.EHE(x,mask)
        out = ifft(out)
        return out
    

    def forward(self, zerofilled, coil, mask, denoiser=None, x0=None, CG_iter=None):
        """
        Perform regularized LS with CG method across multiple batches.
        Note that input and output are already complex tensors.
        """
        if denoiser is None:
            denoiser = torch.zeros_like(zerofilled)
            mu = 0.0
        else:
            mu = self.mu

        # Dimensions to reduce over (all dimensions except batch dim 0)
        reduce_dims = tuple(range(1, zerofilled.ndim))

        # Calculate initial residual r0 = rhs - A(x0)
        rhs = zerofilled + mu * denoiser  # E^H*y + mu*z

        # Compute initial estimate if an x0 is provided
        if x0 is None:
            b_approx = torch.zeros_like(zerofilled)
            p_now = rhs.clone()
        else:
            b_approx = x0.clone()
            p_now = rhs - self.A(b_approx, coil, mask) - mu * b_approx

        r_now = p_now.clone()

        # Compute initial squared residual norm
        r_sq_now = torch.real(torch.sum(r_now * torch.conj(r_now), dim=reduce_dims, keepdim=True))

        for _ in range(CG_iter):
            # q = A * p = T(p) + mu * p
            q = self.A(p_now, coil, mask) + mu * p_now

            p_q = torch.real(torch.sum(torch.conj(p_now) * q, dim=reduce_dims, keepdim=True))
            alpha = r_sq_now / (p_q + 1e-12)
            b_approx = b_approx + alpha * p_now
            r_next = r_now - alpha * q
            r_sq_next = torch.real(torch.sum(r_next * torch.conj(r_next), dim=reduce_dims, keepdim=True))
            beta = r_sq_next / (r_sq_now + 1e-12)
            p_now = r_next + beta * p_now
            r_now = r_next
            r_sq_now = r_sq_next

        return b_approx

