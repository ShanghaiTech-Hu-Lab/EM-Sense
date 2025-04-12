import gaussian
import torch 
import numpy as np

class Mask_generator: 
    def __init__(self, 
                mask_type:str='gaussion_1d',
                acs_lines:int=24,
                ssdu_flag:bool=False,
                ssdu_rho:float=0.4,
                ssdu_acs_block:int=8,
                ssdu_accelerate:int=8,
                ):
        
        self.mask_type = mask_type
        self.acs_lines = acs_lines
        self.ssdu_flag = ssdu_flag
        self.ssdu_rho = ssdu_rho
        self.ssdu_acs_block = ssdu_acs_block
        self.ssdu_accelerate= ssdu_accelerate

    
    def __call__(self, kspace, seed=42, number=None, simulate_mask=False):
        """
        args: B H W coils complex
        return H W 1
        """
        _, h, w, _, _ = kspace.shape

        fun1 = self.gaussion_1d

        mask = np.ones((h, w), dtype=np.float64)
        mask_trn = np.ones((h, w),dtype=np.float64)
        mask_loss = np.ones((h, w), dtype=np.float64)
        if simulate_mask:
          mask = fun1((h,w), seed,  number=None)
        else:
          mask = fun1((h,w), 42,  number=number)
  
        mask_val = torch.as_tensor(mask).to(kspace.device).to(torch.float32).unsqueeze(-1).unsqueeze(-1)
        mask_trn = torch.as_tensor(mask_trn).to(kspace.device).to(torch.float32).unsqueeze(-1).unsqueeze(-1)
        mask_loss = torch.as_tensor(mask_loss).to(kspace.device).to(torch.float32).unsqueeze(-1).unsqueeze(-1)
        
        
        return mask_val, mask_trn, mask_loss
        

    def gaussion_1d(self, shape, seed,  number=None):
        """
        return numpy(1, h)
        """
        h = shape[0]
        h_start = (h - self.acs_lines) // 2
        mask = np.zeros(h, dtype=np.float64)       
        nonzero_count = int(np.round(h / self.ssdu_accelerate - self.acs_lines))
        mask[h_start: h_start + self.acs_lines] = 1

      
        if number is not None:
            gaussian.gaussian_mask_1d(
                nonzero_count, h, h // 2,  6 * np.sqrt(h // 2), mask,  seed + number)
        else:
            gaussian.gaussian_mask_1d(
                nonzero_count, h, h // 2, 6 * np.sqrt(h // 2), mask,  seed) 
        mask = np.expand_dims(mask, axis=0)
        return mask
    