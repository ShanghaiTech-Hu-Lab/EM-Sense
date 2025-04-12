import torch
import torch.nn as nn
from einops import rearrange
from utils import ktoi, rss
from .baseblock import resnet, SpatialGaussianKernel, ResnetConfig



class csm_init(nn.Module):
    def __init__(self, 
                Resnetconfig=ResnetConfig,
                shape=(24, 24), # the center of the k-space 
                ):
        super().__init__()

        
        self.csm_init = resnet(Resnetconfig)
        self.h_size = shape[0]
        self.w_size = shape[1]
     
        self.spatial1 = SpatialGaussianKernel()
        self.spatial2 = SpatialGaussianKernel()

    def forward(self, m_kspace):

        """
        args:
            m_kspace : B H W coils complex
        """
        # get the shape 
        _, h, w, coils, com = m_kspace.shape

        h_start = (h - self.h_size) // 2
        w_start = (w - self.w_size) // 2

        ACS_mask = torch.zeros_like(m_kspace)        
        # get acs mask
        ACS_mask[...,h_start:h_start + self.h_size, w_start:w_start + self.w_size, :,:] = 1
        m_kspace_ACS = m_kspace * ACS_mask 
       
        m_img = ktoi(m_kspace_ACS)
        m_img = m_img / rss(m_img, keepdim=True).unsqueeze(-1)

        m_img = rearrange(m_img, 'B H W coils complex -> B (coils complex) H W')
        m_img = self.spatial1(m_img)
        csm = self.csm_init(m_img)
        csm = rearrange(csm, "B  (coils complex) H W -> B H W coils complex", complex=com, coils=coils).contiguous()
              
        csm = csm / rss(csm, keepdim=True).unsqueeze(-1)
        csm = rearrange(csm, 'B H W coils complex -> B (coils complex) H W')
        csm = self.spatial2(csm)
        csm = rearrange(csm, "B  (coils complex) H W -> B H W coils complex", complex=com, coils=coils).contiguous()
        return csm
    

