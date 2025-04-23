
import torch
import numpy as np

def gvars(n1, n2, L1, L2):
    X = np.linspace(-L1/2, L1/2 - L1/n1, n1)
    Y = np.linspace(-L2/2, L2/2 - L2/n2, n2)

    x, y = np.meshgrid(X, Y, indexing='ij')

    kx1 = np.concatenate((torch.arange(0, n1//2), 
                     torch.arange(-n1//2, 0))) * (2*np.pi/L1)
    ky1 = np.concatenate((torch.arange(0, n2//2), 
                     torch.arange(-n2//2, 0))) * (2*np.pi/L2)

    kx, ky = np.meshgrid(kx1, ky1, indexing = 'ij')

    return x, y, kx, ky


def dissipation_torch(w):
    return torch.mean(w**2, axis = 0)

def dealiasing_indices(kx, ky, n1, n2):
    """
    2/3 dealiasing in torch
    """
    ksq = kx**2 + ky**2
    k0 = (n1/3)**2
    k_cutoff = (ksq <= k0)  # This will give you a tensor of 1s and 0s, where condition holds

    return k_cutoff


def dealiase_torch(ff, kx, ky, n1, n2):
    """
    2/3 dealiasing in torch
    """

    # if real take to Fourier domain
    RorC = torch.is_complex(ff)
    if not RorC:
        ff = torch.fft.fftn(ff)

    # Chandler & Kerswell use this
    ksq = kx**2 + ky**2
    k0 = (n1/3)**2
    k_cutoff = (ksq <= k0)  # This will give you a tensor of 1s and 0s, where condition holds

    ff = k_cutoff * ff

    # take back to physical domain
    if not RorC:
        ff = torch.fft.ifftn(ff).real  # 'symmetric' means we're expecting a real output

    return ff


def rhs_torch_fourier_tf(w_flat, kx, ky, nu, x, y, n1, n2):
    # This implementation is based on the matlab code of Mohammad Farazmand.
    # See M. Farazmand, A. K. Saibaba (2023) Tensor-based flow reconstruction from optimally
    # located sensor measurements. JFM 962

    fw = w_flat.reshape((n1, n2))
    ksq = kx**2 + ky**2

    #fw = torch.fft.fftn(w)

    # compute \nabla\omega
    fwx = 1j*kx*fw; wx = torch.fft.ifftn(fwx).real
    fwy = 1j*ky*fw; wy = torch.fft.ifftn(fwy).real
    ksq2 = ksq.clone().detach()
    ksq2[0,0] = 1e17 # to circumwent in-place modification of w
    
    
    fpsi = fw / ksq2

    fu1 =  1j*ky*fpsi; u1 = torch.fft.ifftn(fu1).real
    fu2 = -1j*kx*fpsi; u2 = torch.fft.ifftn(fu2).real
    # add NL term in physical domain
    rhs_w =  -(u1*wx + u2*wy)
    # add forcing
    rhs_w2 = (rhs_w - 4*torch.cos(4*y))

    # add dissipation term in Fourier domain
    frhsw = torch.fft.fftn(rhs_w2)
    frhsw2 = frhsw - nu*ksq*fw

    # dealiase
    frhsw = dealiase_torch(frhsw, kx, ky, n1, n2)

    # take back to physical domain
    #rhs_w = torch.fft.ifftn(frhsw2).real

    return frhsw2.ravel()

def newton_method(F, u0, tol=1e-6, max_iter=1000):
    """Simple Newton's method."""
    u = u0.clone().detach().requires_grad_(True)
    sol = {}
    sol['u0'] = u0
    for _ in range(max_iter):
        f_val = F(u)
        jacobian = torch.autograd.functional.jacobian(F, u) # derivative wrt u
        du = -torch.linalg.lstsq(jacobian, f_val).solution

        u  = u +  du.squeeze()

        if torch.linalg.norm(du) < tol:
            sol['x'] = u
            break
    sol['x'] = u

    if torch.linalg.norm(F(u)) < tol:
        sol['success'] = 1
    return sol