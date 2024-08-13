# FedALS



Abstract: This paper focuses on reducing the communication cost of federated learning by exploring generalization bounds and representation learning. We first characterize a tighter generalization bound for one-round federated learning based on local clients' generalizations and heterogeneity of data distribution (non-iid scenario). We also characterize a generalization bound in R-round federated learning and its relation to the number of local updates (local stochastic gradient descents (SGDs)). Then, based on our generalization bound analysis and our representation learning interpretation of this analysis, we show for the first time that less frequent aggregations, hence more local updates, for the representation extractor (usually corresponds to initial layers) leads to the creation of more generalizable models, particularly for non-iid scenarios. We design a novel Federated Learning with Adaptive Local Steps (FedALS) algorithm based on our generalization bound and representation learning analysis.
FedALS employs varying aggregation frequencies for different parts of the model, so reduces the communication cost. The paper is followed with experimental results showing the effectiveness of FedALS.

## Code Organization

- **main_fed_layer.py** is the main script to simulate all algorithms. There is a variable called ```mode``` that can be set to simulate different scenarios. ```mode = "cons""``` generates the result of average consensus distance across layers for ResNet-20 and specified data-set in ```identity```.
The rest of the configuration for each experiment can be set at the beginning of the **main_fed_layer.py** as shown in the following:
```python
inp = [None, "2", "iid", "100000", "1", "0", "0", "sgd"]
###############config_variables###############
config_file_name = inp[1]  # network topology
num_of_node = int(config_file_name)
sampling_f = 100  # record frequency of training loss and test accuracy to measure
iteration = 2 * 10 ** 4
repeat_simulation = 10
child_process = False  # if True creates `repeat_simulation`s parallel threads to run all simultaneously
mps = True
gpu = int(inp[6])  # gpu core number to use
num_worker = 0  # number of workers to lead data samples (torch.num_worker)
batch_size = 64
cte = True  # set true for constant leraning rate, set [100,10] for decaying learning rate (Check Functions.learning_rate)
# cte = [100,10]
lr_exp = [
    -7]  # log of the different values for the learning rate - If decaying learning rate is set this weill determine the initial value
iid = inp[2] == "iid"
identity = [
    "cifar10", ]  # task identity: chose between "cifar10", "cifar100", "SVHN", "Mnist", "ImageNet", "LLM"
H = int(inp[3])
alpha = int(inp[4])
L = int(inp[5])
mode = inp[
    7]  # to be "sca" for SCAFFOLD, "sgd" for SGD, "gen" to measure generalization (works with identity = ["synthetic",]), "cons" to measure consensus distance over layers.
inp_dim = 10  # feature dimension just in identity = ["synthetic",]
out_dim = 1  # nn.output dimension just in identity = ["synthetic",]
out_class = 10  # number of classes just in identity = ["synthetic",]
each_class_train_data = 5000  # just in identity = ["synthetic",]
each_class_test_data = 1000  # just in identity = ["synthetic",]
mean = torch.FloatTensor([1 for i in range(
    inp_dim)])  # distance of the mean for two cunsecuative cleasses, can be set different valuess for different dimentions, larger means more margine between classes, just in identity = ["synthetic",]
cov = torch.eye(inp_dim)  # cov matrix of the gussian features in identity = ["synthetic",]
simulation_result_file_name = mode + "_learning_rate_cte" + str(cte) + config_file_name + "_nodes" + identity[
    0] + "_iid" + str(iid) + "_H" + str(H) + "_alpha" + str(alpha) + "_L" + str(L) + "_iter" + str(
    iteration) + "_batch_size" + str(batch_size) + "lr" + str(len(lr_exp))
###############config_variables###############
```

- **config_file_name** identifies the numer of nodes in the network of the federated learning setting.
- Once you run **main_fed_layer.py** completely, the simulation result will be saved at **result/simulation_result_file_name**.
- For the purpose of visualization use **visualization_difference.py** where you first set the config as follows.
```python
###############config_variables###############
data_file_name1 = 'result/fed_simulation_result_5nodes_cifar10_iidFalse_H50_alpha1'
data_file_name2 = 'result/fed_simulation_result_5nodes_cifar10_iidTrue_H50_alpha1'
final_figure_name = "consensus_error_across_layers"
###############config_variables###############
```
Here you can input multiple files to see the result.

Finally the figure will be saved at **figures/final_figure_name.pdf**.

-Use ``mode = sgd or sca`` in **main_fed_layer.py** to generates the result of convergence behaviour and generalization (test accuracy) for SGD and SCAFFOLD, respectively.
The configuration of the experiment can be set at the beginning of the **main_fed_layer.py** as shown before.

- Once you run **main_fed_layer.py** completely, the simulation result will be saved at **result/simulation_result_file_name**.
- For the purpose of visualization use **visualization.py** where you first set the config as follows.

```python
###############config_variables###############
data_file_name1 = 'result/simulation_result_5nodes_cifar100_iidFalse_H50_alpha1_iter10000_batch_size64'
data_file_name2 = 'result/simulation_result_5nodes_cifar100_iidFalse_H50_alpha10_iter10000_batch_size64'
data_file_name3 = 'result/scaffold_simulation_result_5nodes_cifar100_iidFalse_H50_alpha1_iter20000_batch_size64'
data_file_name4 = 'result/scaffold_simulation_result_5nodes_cifar100_iidFalse_H50_alpha10_iter20000_batch_size64'
final_figure_name = "iidFalse_cifar10_scaffold"
iteration =   10 **4
frequency = 100
###############config_variables###############
```
Finally the figure will be saved at **figures/final_figure_name.pdf**.


-Use ``mode = gen`` in **main_fed_layer.py** to generates the result of generalization behaviour for SGD.
The configuration of the experiment can be set at the beginning of the **main_fed_layer.py** as shown before.
If you want to have access to the data distribution that the data is coming from use `identity = synthetic`. For instance, you can verify the result of first main theorem in the paper related to one-round federated learning using a synthetic dataset by setting `identity = synthetic`.
Note that as we are willing to measure generalization we need to have access to the distribution that the data is coming from. This is the reason that we are using a synthetic dataset.
In order to visualize the result for this case, use **visualization_gen.py**.
