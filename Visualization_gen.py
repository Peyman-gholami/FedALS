import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%
###############config_variables###############
data_file_name1 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_2nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name2 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_3nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name3 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_4nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name4 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_5nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name5 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_6nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name6 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_7nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name7 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_8nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name8 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_9nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name9 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_10nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name10 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_15nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name11 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_20nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name12 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_25nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name13 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_30nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name14 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_35nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name15 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_40nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name16 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_45nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
data_file_name17 = 'result/gen_train_size_500_not_cte_lr_alpha_final_simulation_result_50nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size64lr1'
# data_file_name5 = 'result/gen_not_cte_lr_alpha_final_simulation_result_50nodes_synthetic_iidTrue_H500000_alpha1_L0_iter20000_batch_size16lr1'
final_figure_name = "gen"

###############config_variables###############
avg = []
std = []
with open("./result/"+data_file_name1, "r") as fp:
    data = json.load(fp)
with open("./result/"+data_file_name2, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name3, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name4, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name5, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name6, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name7, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name8, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name9, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name10, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name11, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name12, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name13, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name14, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name15, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name16, "r") as fp:
    data2 = json.load(fp)
data += data2
with open("./result/"+data_file_name17, "r") as fp:
    data2 = json.load(fp)
data += data2

def process(data):
    global_gen = []
    local_gen = []
    noniid = []
    for i in range(len(data)):
        instance_global_gen = []
        instance_noniid = []
        instance_local_gen = []
        for j in range(len(data[i])):
            instance_global_gen.append(abs(data[i][j][0][0][1] - data[i][j][0][0][0]))
            temp_local_gen = []
            for k in range(len(data[i][j][-2][0])):
                temp_local_gen.append(abs(data[i][j][-2][0][k][1]-data[i][j][-2][0][k][0]))
            instance_local_gen.append(temp_local_gen)
            temp_noniid = []
            for k in range(len(data[i][j][-1][0])):
                temp_noniid.append(abs(data[i][j][-1][0][k][0] - data[i][j][-2][0][k][0]))
                # print(temp_noniid[-1],data[i][j][-1][0][k][0], - data[i][j][-2][0][k][0] )
            instance_noniid.append(temp_noniid)
        global_gen.append(np.average(instance_global_gen))
        local_gen.append(np.average(np.array(instance_local_gen),axis=0))
        noniid.append(np.average(np.array(instance_noniid),axis=0))
    return global_gen, local_gen, noniid

global_gen, local_gen, noniid = process(data)
bound = [local_gen[i]/local_gen[i].shape[-1]**2 + 2/local_gen[i].shape[-1] *np.sqrt(noniid[i] * local_gen[i]) for i in range(len(local_gen))]
bound2 = [local_gen[i]/local_gen[i].shape[-1] for i in range(len(local_gen))]
bound = [np.average(bound[i]) for i in range(len(bound))]
bound2 = [np.average(bound2[i]) for i in range(len(bound))]

#%%
fig,ax = plt.subplots(figsize=(12,10))#,nrows=1, ncols=2)
plt.rcParams.update({'font.size': 19})
plt.xticks(fontsize = 19)
plt.yticks(fontsize = 19)
colors = matplotlib.cm.tab20(range(20))
b=0
markers=["o","X","P","v","s","h","<",">","d","^","*"]

def f1(x,y_two):
    # return y_two/(x/2)**2
    return 1/(x)**2
def f2(x,y_two):
    # return y_two/(x/2)
    return 1/(x)
# xfp = np.arange(1, 50.5, 0.2)
# y1fp = f1(xfp,avg[0])
# y2fp = f2(xfp,avg[0])

y = list(range(2,11)) + [15,20,25,30,35,40,45,50]

# ax.plot(y,avg[b:].T,'bo',markersize=15,)
ax.plot(y,bound[b:])
# ax.plot(y,bound2[b:])
ax.plot(y,global_gen[b:])

for i,line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
    line.set_color(colors[i])
    line.set_markersize(15)
# for i in range(b,len(avg)):
#     ax.fill_between(y, avg[i,0:].T - 0*std[i,0:].T, avg[i,0:].T + 0*std[i,0:].T,
#                   alpha=0.2)
# ax.plot(xfp, y1fp)
# ax.plot(xfp, y2fp)
# ax.set_ylabel(r'Global generalization/ Average local generalization')
ax.set_xlabel('# of Clients')
ax.legend(["Generalization upper bound","Experiment generalization results"])
ax.grid(True,which="both")
# plt.xlim(0,21)
# plt.ylim(0,2)


plt.savefig("./figures/"+final_figure_name+".pdf",dpi =600,bbox_inches='tight',format='pdf')


