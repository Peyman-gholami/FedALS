import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

###############config_variables###############
data_file_name1 = 'result/sgd_not_cte_lr_alpha_final_simulation_result_25nodes_cifar10_iidFalse_H5_alpha1_L19_iter20000_batch_size64lr8'
data_file_name2 = 'result/sgd_not_cte_lr_alpha_final_simulation_result_25nodes_cifar10_iidFalse_H5_alpha5_L19_iter20000_batch_size64lr8'
data_file_name3 = 'result/not_cte_lr_alpha_final_simulation_result_5nodes_cifar100_iidFalse_H5_alpha10_L12_iter20000_batch_size64lr8'
data_file_name4 = 'result/not_cte_lr_alpha_final_simulation_result_5nodes_cifar100_iidFalse_H5_alpha10_L8_iter20000_batch_size64lr8'
data_file_name5 = 'result/not_cte_lr_alpha_final_simulation_result_5nodes_cifar100_iidFalse_H5_alpha10_L4_iter20000_batch_size64lr8'
data_file_name6 = 'result/not_cte_lr_alpha_final_simulation_result_5nodes_cifar100_iidFalse_H5_alpha10_L0_iter20000_batch_size64lr8'
data_file_name7 = 'result/not_cte_lr_alpha_final_simulation_result_5nodes_SVHN_iidFalse_H5_alpha10_L0_iter20000_batch_size64lr8'
final_figure_name = "L_range_iidFalse_cifar100"
iteration =     2 * 10 ** 4
frequency = 100
num_of_lr = 8
final_av_window = 10
###############config_variables###############

def extract_data(loss_data):
    l = []
    a = []
    t = []
    c = []
    for i in range(len(loss_data)):
        l.append([[loss_data[i][j][0][k][0] for k in range(len(loss_data[i][j][0]))] for j in range(len(loss_data[i]))])
        a.append([[loss_data[i][j][0][k][1] for k in range(len(loss_data[i][j][0]))] for j in range(len(loss_data[i]))])
        t.append([loss_data[i][j][1] for j in range(len(loss_data[i]))])
        c.append([loss_data[i][j][2] for j in range(len(loss_data[i]))])
    return l, a, t, c


def linear_estimate(data, t, ind):
    res = []
    for i in ind:
        for j in range(len(t)):
            if t[j] >= i:
                # print(j,i)
                res.append(data[j - 1] + (i - t[j - 1]) / (t[j] - t[j - 1]) * (data[j] - data[j - 1]))
                break
    return res


with open("./result/"+data_file_name1, "r") as fp:
    data = json.load(fp)
with open("./result/"+data_file_name2, "r") as fp:
    data2 = json.load(fp)
data += data2
# with open("./result/"+data_file_name3, "r") as fp:
#     data2 = json.load(fp)
# data += data2
# with open("./result/"+data_file_name4, "r") as fp:
#     data2 = json.load(fp)
# data += data2
# with open("./result/"+data_file_name5, "r") as fp:
#     data2 = json.load(fp)
# data += data2
# with open("./result/"+data_file_name6, "r") as fp:
#     data2 = json.load(fp)
# data += data2

#%%
iter_loss, iter_accuracy ,iter_t, iter_comm = extract_data(data)
#%%
t_loss = []
t_accuracy = []
t_comm = []
time = range(0, iteration, frequency)
for i in range(len(iter_loss)):
    print(i)
    t_loss.append([linear_estimate(iter_loss[i][j], iter_t[i][j], time) for j in range(len(iter_loss[i]))])
    print("d")
    t_accuracy.append([linear_estimate(iter_accuracy[i][j], iter_t[i][j], time) for j in range(len(iter_loss[i]))])
    t_comm.append([linear_estimate(iter_comm[i][j], iter_t[i][j], time) for j in range(len(iter_comm[i]))])

#%%
av_iter_loss = []
std_iter_loss = []
av_t_loss = []
std_t_loss = []
av_iter_accuracy = []
std_iter_accuracy = []
av_t_accuracy = []
std_t_accuracy = []
av_iter_comm = []
std_iter_comm = []
av_t_comm = []
std_t_comm = []
for i in range(len(iter_loss)):
    av_iter_loss.append(np.average(iter_loss[i], axis=0))
    std_iter_loss.append(np.std(iter_loss[i], axis=0))
    av_t_loss.append(np.average(t_loss[i], axis=0))
    std_t_loss.append(np.std(t_loss[i], axis=0))
    av_iter_accuracy.append(np.average(iter_accuracy[i], axis=0))
    std_iter_accuracy.append(np.std(iter_accuracy[i], axis=0))
    av_t_accuracy.append(np.average(t_accuracy[i], axis=0))
    std_t_accuracy.append(np.std(t_accuracy[i], axis=0))
    av_iter_comm.append(np.average(iter_comm[i], axis=0))
    std_iter_comm.append(np.std(iter_comm[i], axis=0))
    av_t_comm.append(np.average(t_comm[i], axis=0))
    std_t_comm.append(np.std(t_comm[i], axis=0))
#%%
av_iter_loss = np.array([av_iter_loss[i][:iteration//frequency] for i in range(len(av_iter_loss))])
std_iter_loss = np.array([std_iter_loss[i][:iteration//frequency] for i in range(len(std_iter_loss))])
av_iter_comm = np.array([av_iter_comm[i][:iteration//frequency] for i in range(len(av_iter_comm))])
std_iter_comm = np.array([std_iter_comm[i][:iteration//frequency] for i in range(len(std_iter_comm))])
av_iter_accuracy = np.array([av_iter_accuracy[i][:iteration//frequency] for i in range(len(av_iter_accuracy))])
std_iter_accuracy = np.array([std_iter_accuracy[i][:iteration//frequency] for i in range(len(std_iter_accuracy))])

av_t_loss = np.array([av_t_loss[i][:iteration//frequency] for i in range(len(av_t_loss))])
std_t_loss = np.array([std_t_loss[i][:iteration//frequency] for i in range(len(std_t_loss))])
av_t_comm = np.array([av_t_comm[i][:iteration//frequency] for i in range(len(av_t_comm))])
std_t_comm = np.array([std_t_comm[i][:iteration//frequency] for i in range(len(std_t_comm))])
av_t_accuracy = np.array([av_t_accuracy[i][:iteration//frequency] for i in range(len(av_t_accuracy))])
std_t_accuracy = np.array([std_t_accuracy[i][:iteration//frequency] for i in range(len(std_t_accuracy))])


#%%
indices = []
for i in range(len(av_iter_accuracy)//num_of_lr):
    max_value = 0
    index = -1
    for j in range(i * num_of_lr,(i+1) * num_of_lr):
        value = np.average(av_iter_accuracy[j][-final_av_window:])
        if value >= max_value:
            max_value = value
            index = j
    print(index)
    print(max_value)
    print(np.average(std_iter_accuracy[index][-final_av_window:]))
    indices.append(index)
#%%
av_iter_loss = av_iter_loss[indices,:]
std_iter_loss = std_iter_loss[indices,:]
av_iter_comm = av_iter_comm[indices,:]
std_iter_comm = std_iter_comm[indices,:]
av_iter_accuracy = av_iter_accuracy[indices,:]
std_iter_accuracy = std_iter_accuracy[indices,:]

av_t_loss = av_t_loss[indices,:]
std_t_loss = std_t_loss[indices,:]
av_t_comm = av_t_comm[indices,:]
std_t_comm = std_t_comm[indices,:]
av_t_accuracy = av_t_accuracy[indices,:]
std_t_accuracy = std_t_accuracy[indices,:]


#%%
fig,ax = plt.subplots(figsize=(24,10),nrows=1, ncols=2)
plt.rcParams.update({'font.size': 19})
plt.xticks(fontsize = 19)
plt.yticks(fontsize = 19)
colors = matplotlib.cm.tab20(range(20))
b=0
c=0
markers=["o","X","P","^","v","s","h","<",">","d","*"]
every=[5,5,5,5,5,5,5,5,5,5,5]
order = [10,0,9,5,5,6,5,6,6]

###############loss_iteration###############
y = range(c,iteration//frequency)
ax[0].plot(y,av_iter_loss[b:,c:].T)
for i,line in enumerate(ax[0].get_lines()):
    line.set_marker(markers[i])
    line.set_markevery(5)
    # line.set_color(colors[i])
    line.set_markersize(10)
for i in range(b,len(av_iter_loss)):
    ax[0].fill_between(y, av_iter_loss[i,c:].T - 1*std_iter_loss[i,c:].T, av_iter_loss[i,c:].T + 1*std_iter_loss[i,c:].T,
                  alpha=0.2)
ax[0].set_ylabel('Training global loss')
ax[0].set_xlabel('Iteration($10^2$)')
# ax[0].legend([r"FedAvg/ $\tau$=50",r"FedALS/ $\tau$=50/ $\alpha$ = 10", r"FedAvg/ $\tau$=500"])
ax[0].legend([r"FedAvg/ $\tau$=5",r"FedALS/ $\tau$=5/ $\alpha$ = 5",])# r"SCAFFOLD/ $\tau$=5",r"FedALS + SCAFFOLD/ $\tau$=5/ $\alpha$ = 10",])
ax[0].grid(True,which="both")
# ax[0].set_ylim(1,3)
# ax[0].legend([r"$\alpha$ = 1",r"$\alpha$ = 5",r"$\alpha$ = 10",r"$\alpha$ = 25",r"$\alpha$ = 50",r"$\alpha$ = 100"])
# ax[0].legend(["$L$ = 20",r"$L$ =16","$L$ = 12","$L$ = 8","$L$ = 4","$L$ = 0"])
ax[1].plot(y,av_iter_accuracy[b:,c:].T)
for i,line in enumerate(ax[1].get_lines()):
    line.set_marker(markers[i])
    line.set_markevery(5)
    # line.set_color(colors[i])
    line.set_markersize(10)
for i in range(b,len(av_iter_loss)):
    ax[1].fill_between(y, av_iter_accuracy[i,c:].T - 1*std_iter_accuracy[i,c:].T, av_iter_accuracy[i,c:].T + 1*std_iter_accuracy[i,c:].T,
                  alpha=0.2)
ax[1].set_ylabel('Test accuracy')
ax[1].set_xlabel('Iteration ($10^2$)')
ax[1].legend([r"FedAvg/ $\tau$=5",r"FedALS/ $\tau$=5/ $\alpha$ = 5",])# r"SCAFFOLD/ $\tau$=5",r"FedALS + SCAFFOLD/ $\tau$=5/ $\alpha$ = 10",])
# ax[1].legend([r"FedAvg/ $\tau$=50",r"FedALS/ $\tau$=50/ $\alpha$ = 10", r"FedAvg/ $\tau$=500"])
ax[1].grid(True,which="both")
# ax[1].set_ylim(1,1.5)
# ax[1].legend([r"$\alpha$ = 1",r"$\alpha$ = 5",r"$\alpha$ = 10",r"$\alpha$ = 25",r"$\alpha$ = 50",r"$\alpha$ = 100"])
# ax[1].legend(["$L$ = 20",r"$L$ =16","$L$ = 12","$L$ = 8","$L$ = 4","$L$ = 0"])
###############loss_iteration###############

# plt.savefig("./figures/"+final_figure_name+".pdf",dpi =600,bbox_inches='tight',format='pdf')



