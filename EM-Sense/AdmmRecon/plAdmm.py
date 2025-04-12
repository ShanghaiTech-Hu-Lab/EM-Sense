import torch
import torch.nn as nn
import lightning as L
from .AdmmBlock import ADMM_Net
from .baseblock import ResnetConfig
from data import data_process, Mask_generator
from utils import  ktoi, reduce_op, setup_seed
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection
from pprint import pprint



class ADMM(L.LightningModule):
    def __init__(self,
                 shared_weight=True,
                 iters = 5,
                 seed = 42,
                 loss_type="L1",
                 in_channels=2,
                 out_channels=2,
                 middle_channels=64,
                 expand_channels=16,
                 attention=True,
                 img_iters=5,
                 csm_iters=3,
                 csm_numbers=15,
                 rho_prime=True,
                 learning_rate=0.0001,
                 step_size=50,
                 gamma=0.5,
                 **kwargs,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.seed = seed
        self.loss_type = loss_type
        self.iters = iters
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma  = gamma

        img_config = ResnetConfig(in_channels=in_channels,
                                  out_channels=out_channels,
                                  hidden_channels=middle_channels,
                                  C=expand_channels,
                                  attention=attention,
                                  iters=img_iters
                                  )
        csm_config = ResnetConfig(in_channels=csm_numbers,
                                  out_channels=csm_numbers,
                                  hidden_channels=middle_channels,
                                  C=expand_channels,
                                  attention=attention,
                                  iters=csm_iters
                                 )

        setup_seed(seed)
        if self.loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif self.loss_type=="L1":
            self.loss = nn.L1Loss()
        

        self.iters = iters
        self.metric = MetricCollection([PeakSignalNoiseRatio(), StructuralSimilarityIndexMeasure()])
        
        
    
        self.mask_generator = Mask_generator(mask_type='gaussion_1d', 
                                            acs_lines=24,
                                            ssdu_flag=False, 
                                            ssdu_accelerate=4, 
                                            ssdu_rho=0.4, 
                                            ssdu_acs_block=8
                                            )
       
        self.admm = ADMM_Net(share=shared_weight, 
                            iterations=iters,
                            Resnetconfig_csm=csm_config,
                            Resnetconfig_img=img_config,
                            rho_prime=rho_prime
                            )

     
    def forward(self, kspace0, mask, csm_kspace, model=None):
        m_img, m_kspace, csm = model(kspace0, mask, csm_kspace)
        return m_img, m_kspace, csm
        
    def training_step(self, batch, batch_idx):
        number, kspace, csm = batch # B H W coils
        # # E-step 
        with torch.no_grad():
            self.admm.eval()
            cine_data = data_process(kspace, self.mask_generator, seed=self.seed, number=number, simulate_mask=False)
            m_img, simulate_kspace, csm  = self(cine_data.val_kspace, cine_data.val_mask, cine_data.val_kspace, self.admm)
            simulate_kspace = simulate_kspace.detach()
        # Noise Compensation
        simulate_kspace = simulate_kspace *(1 -  cine_data.val_mask) + cine_data.val_kspace  
        # M-step 
        cine_data_simulate = data_process(simulate_kspace, self.mask_generator, seed=self.seed + number + self.global_step, simulate_mask=True)
        m_img, m_kspace, csm_2 = self( cine_data_simulate.val_kspace, cine_data_simulate.val_mask,  cine_data_simulate.val_kspace, self.admm)
    
        loss2 = self.loss(m_kspace * cine_data.val_mask , cine_data.val_kspace)
        self.log("loss", loss2, prog_bar=True)
        
        return loss2
    
    def validation_step(self, batch, batch_idx):

        (i, kspace, csm_gt) = batch # B T Coil H W
        cine_data = data_process(kspace, self.mask_generator, seed=self.seed)
        m_img, m_kspace, csm = self(cine_data.val_kspace, cine_data.val_mask, cine_data.val_kspace, self.admm)
        
        csm[csm_gt==0] = 0 
        pred = torch.view_as_complex(reduce_op(m_img, csm_gt)).abs()
        gt = torch.view_as_complex(reduce_op(ktoi(cine_data.gt_kspace), csm)).abs()

        self.metric.update(pred.unsqueeze(1) / pred.unsqueeze(1).max(), gt.unsqueeze(1) / gt.unsqueeze(1).max())

    def on_validation_epoch_end(self):
       # print("val end")
        out_put = self.metric.compute()
        self.log_dict(out_put, prog_bar=True, sync_dist=True)
        #pprint(out_put)
        # for key, values in out_put.items():
        #     self.log("{}".format(key), values, prog_bar=True)
        self.metric.reset()
        
        
    def test_step(self, batch, batch_idx):

        (i, kspace, csm_gt) = batch # B Coil H W
        cine_data = data_process(kspace, self.mask_generator, seed=self.seed + i)
        m_img, m_kspace, csm = self(cine_data.val_kspace, cine_data.val_mask, cine_data.val_kspace, self.admm)
        gt = torch.view_as_complex(reduce_op(ktoi(cine_data.gt_kspace), csm_gt)).abs()

        csm[csm_gt==0] = 0 

        pred = torch.view_as_complex(reduce_op(m_img, csm)).abs()
        self.metric.update(pred.unsqueeze(1) / pred.unsqueeze(1).max(), gt.unsqueeze(1) / gt.unsqueeze(1).max())
        
    def on_test_epoch_end(self):
        print("test end")
        pprint(self.metric.compute())
        self.metric.reset()

    def configure_optimizers(self):
        admm_opt = torch.optim.Adam(self.admm.parameters(), lr=self.learning_rate)
        admm_opt_scheduler = torch.optim.lr_scheduler.StepLR(
            admm_opt, step_size=self.step_size, gamma=self.gamma
        )
        return [admm_opt], [admm_opt_scheduler]

            
