import torch
from configs import Config
from utils import fft, ifft

class DC_SENSE(torch.nn.Module):
    '''
    Data consistency class for multi-band data reconstruction with CG-SENSE
    '''
    def __init__(self, mu=0.05):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad = True)
   
    def E(self, x, coil, mask):
        x = torch.einsum('nqmxy, nmcxy -> nqcxy', x, coil) #Apply coils (c dimension) and sum over multi-band slices (m dimension)
        return fft(x, [-2,-1]) * mask
        
    def EH(self, x, coil, mask):
        image = ifft(x*mask, [-2,-1])
        return torch.einsum('nqcxy, nmcxy -> nqmxy', image, torch.conj(coil)) #Sum over conjugate coils (c dimension) and expand over multi-band slices (m dimension)

    def EHE(self, image, coil, mask):
        return self.EH(self.E(image, coil, mask), coil, mask)

    def forward(self, zerofilled, coil, mask, denoiser=None, x0=None, CG_iter=10):
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
        reduce_dims = tuple(range(2, zerofilled.ndim))

        # Calculate initial residual r0 = rhs - A(x0)
        rhs = zerofilled + mu * denoiser  # E^H*y + mu*z

        # Compute initial estimate if an x0 is provided
        if x0 is None:
            b_approx = torch.zeros_like(zerofilled)
            p_now = rhs.clone()
        else:
            b_approx = x0.clone()
            p_now = rhs - self.EHE(b_approx, coil, mask) - mu * b_approx

        r_now = p_now.clone()

        # Compute initial squared residual norm
        r_sq_now = torch.real(torch.sum(r_now * torch.conj(r_now), dim=reduce_dims, keepdim=True))

        for _ in range(CG_iter):
            # q = A * p = T(p) + mu * p
            q = self.EHE(p_now, coil, mask) + mu * p_now

            p_q = torch.real(torch.sum(torch.conj(p_now) * q, dim=reduce_dims, keepdim=True))
            alpha = r_sq_now / (p_q)
            b_approx = b_approx + alpha * p_now
            r_next = r_now - alpha * q
            r_sq_next = torch.real(torch.sum(r_next * torch.conj(r_next), dim=reduce_dims, keepdim=True))
            beta = r_sq_next / (r_sq_now)
            p_now = r_next + beta * p_now
            r_now = r_next
            r_sq_now = r_sq_next

        return b_approx, mu

