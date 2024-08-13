import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

###############config_variables###############
data_file_name1 = 'result/fed_simulation_result_5nodes_cifar10_iidFalse_H200_alpha1'
# data_file_name2 = 'result/fed_simulation_result_5nodes_cifar10_iidTrue_H50_alpha1'
final_figure_name = "consensus_error_across_layers"
###############config_variables###############
avg = []
std = []
with open("./result/"+data_file_name1, "r") as fp:
    data = json.load(fp)
d = [data[0][j][-1] for j in range(len(data[0]))]
d = np.array(d)
avg.append(np.average(d,axis=(0, 1)))
std.append(np.std(d,axis=(0, 1)))

# with open("./result/"+data_file_name2, "r") as fp:
#     data = json.load(fp)
# d = [data[0][j][-1] for j in range(len(data[0]))]
# d = np.array(d)
# avg.append(np.average(d,axis=(0, 1)))
# std.append(np.std(d,axis=(0, 1)))

avg = np.array(avg)
std = np.array(std)

#%%
fig,ax = plt.subplots(figsize=(12,10))#,nrows=1, ncols=2)
plt.rcParams.update({'font.size': 19})
plt.xticks(fontsize = 19)
plt.yticks(fontsize = 19)
colors = matplotlib.cm.tab20(range(20))
b=0
markers=["o","X","P","^","v","s","h","<",">","d","*"]

y = range(1,len(avg[0])+1)
ax.plot(y,avg[b:,0:].T)
for i,line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
    # line.set_color(colors[i])
    line.set_markersize(10)
for i in range(b,len(avg)):
    ax.fill_between(y, avg[i,0:].T - 1*std[i,0:].T, avg[i,0:].T + 1*std[i,0:].T,
                  alpha=0.2)
ax.set_ylabel('Average consensus distance')
ax.set_xlabel('Model layer')
ax.legend([r"FedAvg/ $\tau$=50 / non-iid",r"FedAvg/ $\tau$=50 / iid"])
ax.grid(True,which="both")
plt.xlim(0,21)
# plt.ylim(0,2)


# plt.savefig("./figures/"+final_figure_name+".pdf",dpi =600,bbox_inches='tight',format='pdf')


