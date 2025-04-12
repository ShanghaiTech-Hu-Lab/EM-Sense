import torch
import torch.nn as nn
from einops import rearrange
from .baseblock import SpatialGaussianKernel, resnet, ResnetConfig
from utils import expand_op, rss, complex_mutmul, complex_conj, ktoi, itok,reduce_op, A
from .csm import csm_init


"""
For all the ADMM modules, you can find the original function from the eq(22) of the paper
"""



class ADMM_Net(nn.Module):
    def __init__(self,
                Resnetconfig_csm:ResnetConfig,
                Resnetconfig_img:ResnetConfig,
                share:bool =True,
                iterations:int =5,
                rho_prime:bool =True,
                ):
        
        super().__init__()


        self.share = share
        self.iterations = iterations
        self.rho = nn.Parameter(torch.ones(1) * 0.5)

        self.s_img_updata = nn.ModuleList()
        self.csm_updata = nn.ModuleList()
        self.m_img_updata = nn.ModuleList()
        self.beta_updata = nn.ModuleList()
        if self.share:
            updata_beta = beta_updata()
            updata_s_img = s_img_update(Resnetconfig=Resnetconfig_img)
            updata_m_img = m_img_update(rho_prime=rho_prime)
            updata_csm = csm_updata(Resnetconfig=Resnetconfig_csm)
            for i in range(self.iterations):
                self.s_img_updata.insert(i, updata_s_img)
                self.csm_updata.insert(i, updata_csm)
                self.m_img_updata.insert(i, updata_m_img)
                self.beta_updata.insert(i, updata_beta)
        else:
            for i in range(self.iterations):
                self.s_img_updata.insert(i, s_img_update(Resnetconfig=Resnetconfig_img))
                self.csm_updata.insert(i, csm_updata(Resnetconfig=Resnetconfig_csm))
                self.m_img_updata.insert(i, m_img_update(rho_prime=rho_prime))
                self.beta_updata.insert(i, beta_updata())
        self.csm_init = csm_init(Resnetconfig=Resnetconfig_csm)

    def forward(self, kspace0, mask, csm_kspace, mask2=None):
        csm = self.csm_init(csm_kspace) 
        m_img = ktoi(kspace0) 
        s_img = reduce_op(m_img, csm)
        beta = torch.zeros_like(m_img)
        A_op = A(mask)
        A_op2 = A(mask2) if mask2 is not None else None
        for i, (updata_m_img, updata_s_img, updata_beta, updata_csm) in enumerate(zip(self.m_img_updata, self.s_img_updata, self.beta_updata, self.csm_updata)):
        # updata s_img
            s_img = updata_s_img(m_img, s_img, csm, beta, self.rho)
        # updata csm
            csm = updata_csm(m_img, s_img, csm, beta, self.rho) 
        # updata m_img
            m_img = updata_m_img(m_img, s_img, csm, beta, self.rho, A_op, kspace0, A_op2)
        # updata u
            beta = updata_beta(m_img, s_img, csm, beta)

        m_img = expand_op(s_img, csm)
        
        return m_img, itok(m_img), csm




"""
args:
    s_img: B H W complex
    m_img: B H W coils complex
    csm: B H W coils complex
    mu: B H W coils complex
"""

class csm_updata(nn.Module):

    def __init__(self, Resnetconfig=ResnetConfig):
        super().__init__()
    
        self.prox = resnet(Resnetconfig=Resnetconfig)
        self.spatial1 = SpatialGaussianKernel()
        self.spatial2 = SpatialGaussianKernel()
        self.lam = nn.Parameter(torch.ones(1))
    def update(self, m_img, s_img, csm, beta, rho):
        coils, complex = m_img.shape[-2], m_img.shape[-1]
        
        csm_next = self.grad(m_img, s_img, csm, beta, rho) # B H W coils complex
        csm_next = rearrange(csm_next, 'B H W coils complex -> B (coils complex) H W').contiguous()
        csm_next = self.spatial1(csm_next)
        csm_next = self.prox(csm_next)
        csm_next = rearrange(csm_next, 'B (coils complex) H W -> B H W coils complex', coils=coils, complex=complex).contiguous()
        csm_next = csm_next / rss(csm_next).unsqueeze(-1)        
        csm_next = rearrange(csm_next, 'B H W coils complex -> B (coils complex) H W')
        csm_next = self.spatial2(csm_next)
        csm_next = rearrange(csm_next, "B (coils complex)  H W -> B H W coils complex", complex=complex, coils=coils).contiguous()
        return csm_next
    def forward(self, m_img, s_img, csm, beta, rho):
    
        csm_next = self.update(m_img, s_img, csm, beta, rho)

        return csm_next
    
    def grad(self, m_img, s_img, csm, beta, rho):
        """
        rho/2 (||csm * s_img - m_img + beta ||^2)
        m_img: B H W coils complex
        s_img: B H W complex 
        csm: B H W coils complex
        u: B H W coils complex

        return:
        csm_next B H W coils complex
        
        """
        grad_csm  = rho * complex_mutmul(expand_op(s_img, csm) - m_img + beta, complex_conj(s_img).unsqueeze(-2))
        grad_csm = grad_csm / (torch.square(torch.abs(torch.view_as_complex(s_img.contiguous()))).unsqueeze(-1).unsqueeze(-1))  #FIXME
        csm_next = csm - self.lam * grad_csm
        csm_next = csm_next / rss(csm_next).unsqueeze(-1)

        return csm_next


class beta_updata(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.ones(1))
    
    def forward(self, m_img, s_img, csm, beta):
    
        beta_next = self.update(m_img, s_img, csm, beta)

        return beta_next
    
    def update(self, m_img, s_img, csm, beta):
        """
        beta = beta + mu * (csm * s_img - m_img)
        """
        beta_next = beta + self.lam * (expand_op(s_img, csm) - m_img)

        return beta_next


class m_img_update(nn.Module):

    def __init__(self, rho_prime=True):
        super().__init__()
        """
        (The predicted k-space section and the acquired k-space section have different noise levels)
        For here we use different parameters(rho and rho_prime) for the acquired k-space region and the predicted k-space region.
        However, dividing the k-space into distinct regions and applying different regularization parameters (rho) to each section 
        introduces compatibility issues with the Conjugate Gradient (CG) method employed in MoDL (https://arxiv.org/pdf/1712.02862). 
        For here, we recommend to use Variabel split method (https://arxiv.org/abs/1907.10033) 
        or the gradient methods (https://arxiv.org/pdf/2004.06688) as the data consistency method.
        If you also want to use the CG method, you can use the rho_prime as zero. 
        The learnable rho will be not influenced by the noise of the generated k-space region. 
        """
        if  rho_prime:
            self.rho_prime = nn.Parameter(torch.ones(1)) * 0.5 
        else:
            self.rho_prime = None

    
    def update(self, m_img, s_img, csm, beta, rho, A, kspace0, A2):
        m_img_next = self.grad(m_img, s_img, csm, beta, rho, A, kspace0, A2)

        return m_img_next
    
    def forward(self, m_img, s_img, csm, beta, rho, A, kspace0, A2=None):
    

        m_img_next = self.update(m_img, s_img, csm, beta, rho, A, kspace0, A2)
        
        return m_img_next

    def grad(self, m_img, s_img, csm, beta, rho, A, kspace0, A2=None):

        """
        (A^*A + rho * I)^{-1} (A^*kspace0 + rho * (S x + beta))
        m_img: B H W coils complex
        s_img: B H W complex
        csm: B H W coils complex
        u: B H W coils complex

        return:
            s_img_next B H W complex
        """
        kspace_next_un_mask = itok(expand_op(s_img, csm) + beta) * (1 - A.mask)
        if A2 is not None:
            rho2 = self.rho_prime * rho
            kspace_acquired  = (kspace0 / (1 + rho) + rho/(1 + rho) * itok(expand_op(s_img, csm) + beta)) * A.mask * A2.mask
            kspace_generate = (kspace0 / (1 + rho2) + rho2/(1 + rho2) * itok(expand_op(s_img, csm) + beta)) * A.mask * (1 - A2.mask)
            kspace_next_mask = kspace_acquired + kspace_generate
        else:
            kspace_next_mask = (kspace0 / (1 + rho) + rho/(1 + rho) * itok(expand_op(s_img, csm) + beta)) * A.mask
        m_img = ktoi(kspace_next_mask + kspace_next_un_mask)

        return m_img


class s_img_update(nn.Module):

    def __init__(self,
                Resnetconfig=ResnetConfig,
                ):
        
        super().__init__()
        
        self.lam = nn.Parameter(torch.ones(1))
        
        self.prox = resnet(Resnetconfig=Resnetconfig)
    
    def update(self, m_img, s_img, csm, beta, rho):
        s_img_next = self.grad(m_img, s_img, csm, beta, rho)
        # FIXME
        s_img_next= rearrange(s_img_next, 'B H W complex -> B complex H W')
        s_img_next = self.prox(s_img_next.contiguous())
        s_img_next = rearrange(s_img_next,'B complex H W -> B H W complex')

        return s_img_next

    
    def forward(self, m_img, s_img, csm, beta, rho):
     
        s_img_next = self.update(m_img, s_img, csm, beta, rho)
        return s_img_next
       
    def grad(self, m_img, s_img, csm, beta, rho):
        """
        rho/2 (|| csm * s_img + beta - m_img ||^2)
        m_img: B H W coils complex
        s_img: B H W complex
        csm: B H W coils complex
        u: B H W coils complex

        return:
            s_img_next B H W complex
        """
        s_img_grad = rho * torch.sum(complex_mutmul((expand_op(s_img, csm) + beta - m_img), complex_conj(csm)), dim=-2)
        s_img_next = s_img - self.lam * s_img_grad 

        return s_img_next


