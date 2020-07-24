from torch.utils.data import DataLoader, Dataset
#from multiprocessing import Manager
import numpy as np


class HydroDataset(Dataset):
    
    def __init__(self,gen,num_tasks_epoch):
        #pass
        #self.task = gen.generate_task()
        #self.n_samples = self.task['x'].shape[0]
        #self.task['x'].shape[0]
        self.gen = gen
        self.num_tasks_epoch = num_tasks_epoch
        
        self.max_train_points = np.random.randint(self.gen.max_train_points,self.gen.max_train_points+1,1000)
        self.min_train_points = np.random.randint(self.gen.min_train_points,self.gen.min_train_points+1,1000)
                
    def __getitem__(self,index):
        """if index not in self.shared_dict:
            print('Adding {} to shared_dict'.format(index))
            self.shared_dict[index] = self.gen.generate_task(index) #torch.tensor(index)
        return self.shared_dict[index]"""
        
        """task_dict = {}
        
        for k in self.task.keys():
            #exec("%s = task[k]" % k)
            task_dict.update( {str(k) : self.task[k][index]} )
        return gen.generate_task"""
        self.gen.min_train_points = self.min_train_points[index]
        self.gen.max_train_points = self.max_train_points[index]

        return self.gen.generate_task(index)
        
    def __len__(self):
        return self.num_tasks_epoch

    def batch_size(self):
        return self.gen.batch_size