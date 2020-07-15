from torch.utils.data import DataLoader, Dataset

class HydroDataset(Dataset):
    
    def __init__(self):
        pass
        #self.task = gen.generate_task()
        #self.n_samples = self.task['x'].shape[0]
        #self.task['x'].shape[0]
        
    def __getitem__(self,index):
        """task_dict = {}
        for k in self.task.keys():
            #exec("%s = task[k]" % k)
            task_dict.update( {str(k) : self.task[k][index]} )
        return gen.generate_task"""
        return gen.generate_task()
        
    def __len__(self):
        return 1 #self.n_samples

start = time.time()
dataset = HydroDataset()
elapsed = time.time() - start
print(elapsed)
len(dataset)
#print(dataset)
    
dataloader = DataLoader(dataset=HydroDataset(), batch_size=4, shuffle=False, num_workers=2)
#dataiter = iter(dataloader)
#data = dataiter.next()
#print(len(dataset))