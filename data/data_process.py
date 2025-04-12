import torch
import gaussian
from typing import NamedTuple





class cinedata(NamedTuple):
    trn_mask: torch.Tensor
    loss_mask: torch.Tensor
    loss_kspace: torch.Tensor
    trn_kspace: torch.Tensor
    val_kspace: torch.Tensor
    val_mask: torch.Tensor
    gt_kspace : torch.Tensor
    mask_generator: callable


def data_process(kspace: torch.Tensor,
                mask_generator: callable,
                seed:int,
                number=None,
                simulate_mask=False,
                ):
    """
    kspace_shape: B H W coil complex 
    """
    if torch.is_complex(kspace):
        kspace = torch.view_as_real(kspace)

    val_mask, trn_mask, loss_mask = mask_generator(kspace, seed, number, simulate_mask= simulate_mask)
   
    val_kspace = kspace * val_mask
    trn_kspace = kspace * trn_mask
    loss_kspace = kspace * loss_mask
    

    return cinedata(
        trn_mask= trn_mask,
        loss_mask=loss_mask,
        trn_kspace=trn_kspace,
        loss_kspace=loss_kspace,
        val_mask=val_mask,
        val_kspace = val_kspace,
        gt_kspace=kspace,
        mask_generator=mask_generator
    )

if __name__ == "__main__":

    import torch
    a = torch.randn((1, 3, 10, 200,210, 2))
    a_complex = torch.view_as_complex(a)

    out = data_process(a_complex)
    print(out.trn_kspace.shape, out.trn_mask.shape, out.loss_mask.shape, out.loss_kspace.shape, out.mask.shape, out.gt_kspace.shape)
