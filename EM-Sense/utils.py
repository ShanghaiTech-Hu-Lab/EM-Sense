import torch
import torch.fft as fft
from einops import rearrange
import numpy as np
"""

args:
    B (slice) H W coils complex 
"""


def ktoi(m_kspace, dim=(-2, -3)):
    """
        args:
        m_kspace: B ... H W coils complex
        m_img: B ... H W  coils complex
    """
    if torch.is_complex(m_kspace):
        m_kspace_complex = m_kspace
    else:
        m_kspace_complex = torch.view_as_complex(m_kspace) # B ... H W coils
    temp_shift = fft.ifftshift(m_kspace_complex, dim=dim)
    temp_img = fft.ifftn(temp_shift, dim=dim)
    m_img_complex = fft.fftshift(temp_img, dim=dim)
    m_img = torch.view_as_real(m_img_complex)

    return m_img


def itok(m_img, dim=(-2, -3)):
    """
        args:
        m_kspace: B ... H W coils complex
        m_img: B ... H W  coils complex
    """
    if torch.is_complex(m_img):
        m_img_complex = m_img
    else:
        m_img_complex = torch.view_as_complex(m_img)

    temp_shift = fft.ifftshift(m_img_complex, dim=dim)
    temp_kspace = fft.fftn(temp_shift, dim=dim)
    m_kspace_complex = fft.fftshift(temp_kspace, dim=dim)

    m_kspace = torch.view_as_real(m_kspace_complex)

    return m_kspace


def rss(m_img, keepdim=True):
    """
    m_img: B ... H W coils complex
    """

    s_img = torch.sqrt(torch.sum(torch.square(torch.abs(torch.view_as_complex(m_img))), dim=-1, keepdim=keepdim))

    return s_img


def complex_conj(data):
    real = data[...,0]
    imag = -data[...,1]
    data = torch.stack([real, imag], dim=-1)

    return data


def complex_mutmul(a, b):

    c = torch.view_as_complex(a.contiguous()) * torch.view_as_complex(b.contiguous())

    return torch.view_as_real(c)


def expand_op(s_img, coils):
    
    """
    s_img: B (slice) H W complex
    coils: B (slice) H W coils complex
    """
    
    m_img = complex_mutmul(s_img.unsqueeze(-2), coils)

    return m_img


def reduce_op(m_img, coils):
    """
    m_img: B (slice) H W coils complex
    coils: B (slice) H W coils complex
    """
    s_img = torch.sum(complex_mutmul(m_img, complex_conj(coils)), dim=-2)

    return s_img



class A:
    def __init__(self, mask):

        """
        mask ... H W 1 1
        """
        self.mask = mask
    
    def A_op(self, m_img):
        """
        img: B H W coils complex 
        """
        m_kspace = itok(m_img)  # B H W coils complex

        mask_m_kspace = m_kspace * self.mask

        return mask_m_kspace

    def A_star_op(self, m_kspace):
        """
            m_kspace: B H W coils complex
        """
        m_kspace = m_kspace * self.mask

        return ktoi(m_kspace)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True