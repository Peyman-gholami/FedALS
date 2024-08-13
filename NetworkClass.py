from Functions import *


class Network:
    def __init__(self, model, all_node, edge, action, pre_action):
        self.all_node = all_node
        self.edge = edge
        self.action = action
        self.pre_action = pre_action
        self.visited = []
        self.x_agg = copy.deepcopy(model)
        self.c_agg = copy.deepcopy(model)
        self.q = np.ones(len(self.all_node))
        self.p = np.ones(len(self.all_node)) / len(self.all_node)

    def final(self,identity,total_data):
        self.x_agg.load_state_dict(aggrigate([node.x for node in self.all_node],[node.data_size/total_data for node in self.all_node]))
        return self.x_agg

    def final_c(self,identity,total_data):
        self.c_agg.load_state_dict(aggrigate([node.c for node in self.all_node],[node.data_size/total_data for node in self.all_node]))
        return self.c_agg
