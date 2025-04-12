import torch
import pickle
from torch.utils.data import DataLoader
import lightning as L


class DataModule(L.LightningDataModule):
    def __init__(self, 
                batch_size: int = 1,
                numberworks: int=20,
                train_path:str = '/mnt/e/workspace/dataset/T2_train.pkl',
                train_csm_path:str = '/mnt/e/workspace/dataset/T2_train_csm.pkl',
                val_path:str = '/mnt/e/workspace/dataset/T2_val.pkl',
                val_csm_path:str = '/mnt/e/workspace/dataset/T2_val_csm.pkl',
                test_path:str = '/mnt/e/workspace/dataset/T2_test.pkl',
                test_csm_path:str = '/mnt/e/workspace/dataset/T2_test_csm.pkl'
                ):
        super().__init__()
        self.batch_size = batch_size
        self.numberworks = numberworks
        self.train_path = train_path
        self.train_csm_path = train_csm_path
        self.val_path = val_path
        self.val_csm_path = val_csm_path
        self.test_path = test_path
        self.test_csm_path = test_csm_path
        

    def setup(self, stage: str):
        if stage == 'fit':
            with open(self.train_path, 'rb') as f:
                self.train = pickle.load(f)
        
            with open(self.train_csm_path, 'rb') as f:
                self.csm_train = pickle.load(f)

            self.trains = [(i, kspace, csm) for i, (kspace, csm) in enumerate(zip(self.train, self.csm_train))]
            with open(self.val_path, 'rb') as f:
                    self.val = pickle.load(f)
            
            with open(self.val_csm_path, 'rb') as f:
                self.csm_val = pickle.load(f)

            self.vals = [(i, kspace, csm) for i, (kspace, csm) in enumerate(zip(self.val, self.csm_val))]
        elif stage == 'test':
            # with open(self.test_path, 'rb') as f:
            with open(self.test_path, 'rb') as f:
                    self.test = pickle.load(f)
            
            with open(self.test_csm_path, 'rb') as f:
                self.csm_test = pickle.load(f)

            self.tests = [(i, kspace, csm) for i, (kspace, csm) in  enumerate(zip(self.test, self.csm_test))]
        
        
    def train_dataloader(self):
        return DataLoader(self.trains, batch_size=self.batch_size, num_workers=self.numberworks)

    def val_dataloader(self):
        return DataLoader(self.vals, batch_size=self.batch_size, num_workers=self.numberworks)

    def test_dataloader(self):
        return DataLoader(self.tests, batch_size=self.batch_size, num_workers=self.numberworks)