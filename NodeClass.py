from Functions import *

class Node:
    def __init__(self,identity, model,indices, whole_dataset, neighbor, lag, delay_dist, stream, device, criterion, num_worker, batch_size):
        self.delay_shift = 1
        self.lag = lag
        # self.model = model
        self.indices = indices
        self.neighbor = neighbor
        self.x = copy.deepcopy(model)
        # self.x_tau = [copy.deepcopy(model) for st in range(stream)]
        self.x_pre = copy.deepcopy(model)
        self.t_pre = 0
        self.tau = [0 for st in range(stream)]
        self.count = [0 for i in range(len(neighbor))]
        self.delay_dist = delay_dist
        self.traverse_parent = [None for st in range(stream)]
        self.async_agg = []
        self.async_H = 0
        # self.u = copy.deepcopy(model)
        # self.u_other= copy.deepcopy(model)
        self.device = device
        self.criterion = criterion
        self.c = copy.deepcopy(model)
        self.c.load_state_dict(aggrigate([model, model], [1, -1]))
        # self.u = copy.deepcopy(model)
        if  identity[0]=="LLM":
            self.dataset = whole_dataset.select(list(range(indices[0],indices[1])))
            self.data_size = len(self.dataset)
            self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=min(batch_size, len(self.dataset)),
                                                            shuffle=True, num_workers=num_worker)
            self.dataloader_iterator = iter(self.train_loader)
            self.train_batch = []
            self.optimizer = AdamW(self.x.parameters(), lr=1e-4)

        else:
            self.dataset = torch.utils.data.Subset(whole_dataset,list(range(indices[0],indices[1])))
            self.data_size = len(self.dataset)
            self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=min(batch_size, len(self.dataset)), shuffle=True, num_workers=num_worker)
            self.dataloader_iterator = iter(self.train_loader)
            self.train_batch = []
            self.optimizer = torch.optim.SGD(self.x.parameters(), .1,
                            momentum=0.9,
                            weight_decay=1e-4)

    def load_data(self):
        try:
            batch_data = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.train_loader)
            batch_data = next(self.dataloader_iterator)
        self.train_batch = batch_data

    def observe_delay(self, a_index):
        return self.delay_shift + np.random.exponential(self.delay_dist[a_index])

    def local_sgd(self, identity, total_data, t, lr, tokenizer):
        if identity[0] == "LLM":
            self.load_data()
            i_feature = self.train_batch['text']
            i_label = None
        else:
            self.load_data()
            i_feature, i_label = self.train_batch
        self.x, gr = logistic_regression(identity, self.x, i_feature, i_label, lr, self.optimizer, self.device, self.criterion, tokenizer)
        return gr


