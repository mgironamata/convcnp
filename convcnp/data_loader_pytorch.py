from torch.utils.data import DataLoader, Dataset

class HydroDataset(Dataset):
    
    def __init__(self,gen,num_tasks_epoch):
        pass
        #self.task = gen.generate_task()
        #self.n_samples = self.task['x'].shape[0]
        #self.task['x'].shape[0]
        self.gen = gen
        self.num_tasks_epoch = num_tasks_epoch
        
    def __getitem__(self,index):
        """task_dict = {}
        
        for k in self.task.keys():
            #exec("%s = task[k]" % k)
            task_dict.update( {str(k) : self.task[k][index]} )
        return gen.generate_task"""
        return self.gen.generate_task()
        
    def __len__(self):
        return self.num_tasks_epoch

    def batch_size(self):
        return self.gen.batch_size