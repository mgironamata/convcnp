from torch.utils.data import DataLoader, Dataset
#from multiprocessing import Manager
import numpy as np



class HydroDataset(Dataset):
    
    def __init__(self,gen,num_tasks_epoch):#,min_train_points=100,max_train_points=165):

        self.gen = gen
        self.num_tasks_epoch = num_tasks_epoch
        
        self.min_train_points = np.random.randint(50, 60, self.num_tasks_epoch+1)
        self.max_train_points = np.random.randint(140, 150, self.num_tasks_epoch+1)
                
    def __getitem__(self,index):
         
        self.gen.min_train_points = self.min_train_points[index]
        self.gen.max_train_points = self.max_train_points[index]

        return self.gen.generate_task(index)
        
    def __len__(self):
        return self.num_tasks_epoch

    def batch_size(self):
        return self.gen.batch_size

class HydroTestDataset(Dataset):

    def __init__(self,gen,years,basin):
        self.gen = gen
        self.years = years
        self.basin = basin

    def __getitem__(self,index):
        year = self.years[index]
        return self.gen.generate_test_task(year,self.basin)

    def __len__(self):
        return len(self.years)