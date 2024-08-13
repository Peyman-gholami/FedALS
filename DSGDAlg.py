from Functions import *

def parallel_sgd_measure(identity, total_data, whole_dataset, exp, cte, H, iters, sampling_f, device, criterion,
                         num_worker, batch_size, test_loader, alpha, tokenizer, network):
    all_layers = list(network.all_node[0].x.state_dict().keys())
    if identity[0] == "LLM":
        #just using .01 of the dataset to compute training loss
        whole_dataset = whole_dataset.train_test_split(test_size=0.01)
        whole_dataset = whole_dataset['test']
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=32)
        specific_layers = [all_layers[i] for i in range(len(all_layers)) if all_layers[i].endswith("k_proj.weight") or
                           "tokens" in all_layers[i] or "lm_head" in all_layers[i]]
    else:
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_worker)
        specific_layers = list(
            [all_layers[i] for i in range(len(all_layers)) if "conv" in all_layers[i] or "fc.weight" in all_layers[i] or 'linear.weight' in all_layers[i]])
    layer_map = []
    group = None
    for layer in all_layers:
        if layer in specific_layers:
            if group is not None:
                layer_map.append(group)
            group = [(layer, network.all_node[0].x.state_dict()[layer].numel())]
            continue
        group.append((layer, network.all_node[0].x.state_dict()[layer].numel()))
    layer_map.append(group)
    loss_over_t = []
    real_time_over_t = []
    real_time = 0
    comm_over_t = []
    comm = 0
    difference_by_layer_over_t = []
    for t in range(iters):
        if t % sampling_f == 0:
            final_x = network.final(identity, total_data)
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append(loss(identity, whole_dataset, final_x, criterion, device, test_loader, tokenizer, True))
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(identity, total_data, sgd_round, learning_rate(sgd_round, exp, cte), tokenizer)
        if t % H == 0:
            temp = network.final(identity, total_data)
            difference_by_layer_over_t.append(differ([node.x for node in network.all_node], temp,
                                                     [node.data_size / total_data for node in network.all_node],
                                                     layer_map))
            print(difference_by_layer_over_t[-1])
            for node in network.all_node:
                node.x.load_state_dict(temp.state_dict())
        real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t, difference_by_layer_over_t]


def parallel_sgd_layer(identity, total_data, whole_dataset, exp, cte, H, iters, sampling_f, device, criterion,
                       num_worker, batch_size, test_loader, alpha, L, tokenizer, network):
    all_layers = list(network.all_node[0].x.state_dict().keys())
    if identity[0] == "LLM":
        #just using .01 of the dataset to compute training loss
        whole_dataset = whole_dataset.train_test_split(test_size=0.01)
        whole_dataset = whole_dataset['test']
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=32)
        specific_layers = [all_layers[i] for i in range(len(all_layers)) if all_layers[i].endswith("k_proj.weight") or
                           "tokens" in all_layers[i] or "lm_head" in all_layers[i]]
    else:
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_worker)
        specific_layers = list(
            [all_layers[i] for i in range(len(all_layers)) if "conv" in all_layers[i] or "fc.weight" in all_layers[i] or 'linear.weight' in all_layers[i]])
    layer_map = []
    group = None
    for layer in all_layers:
        if layer in specific_layers:
            if group is not None:
                layer_map.append(group)
            group = [layer]
            continue
        group.append(layer)
    layer_map.append(group)
    Hs = [H * alpha for i in range(L)] + [H for i in range(len(layer_map) - L)]
    print(Hs)
    loss_over_t = []
    real_time_over_t = []
    real_time = 0
    comm_over_t = []
    comm = 0
    difference_by_layer_over_t = []
    for t in range(iters):
        if t % sampling_f == 0:
            final_x = network.final(identity, total_data)
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append(loss(identity, whole_dataset, final_x, criterion, device, test_loader, tokenizer, True))
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(identity, total_data, sgd_round, learning_rate(sgd_round, exp, cte), tokenizer)
        to_be_updated_layers = []
        for i in range(len(Hs)):
            if t % Hs[i] == 0:
                to_be_updated_layers += layer_map[i]
        if len(to_be_updated_layers) != 0:
            temp = network.final(identity, total_data)
            new_model = temp.state_dict()
            for node in network.all_node:
                model_dict = node.x.state_dict()
                for layer in to_be_updated_layers:
                    model_dict[layer] = new_model[layer]
                node.x.load_state_dict(model_dict)
        real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t, difference_by_layer_over_t]


def parallel_scaffold_layer(identity, total_data, whole_dataset, exp, cte, H, iters, sampling_f, device, criterion,
                            num_worker, batch_size, test_loader, alpha, L, tokenizer, network):
    all_layers = list(network.all_node[0].x.state_dict().keys())
    if identity[0] == "LLM":
        #just using .01 of the dataset to compute training loss
        whole_dataset = whole_dataset.train_test_split(test_size=0.01)
        whole_dataset = whole_dataset['test']
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=32)
        specific_layers = [all_layers[i] for i in range(len(all_layers)) if all_layers[i].endswith("k_proj.weight") or
                           "tokens" in all_layers[i] or "lm_head" in all_layers[i]]
    else:
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_worker)
        specific_layers = list(
            [all_layers[i] for i in range(len(all_layers)) if "conv" in all_layers[i] or "fc.weight" in all_layers[i] or 'linear.weight' in all_layers[i]])
    layer_map = []
    server_c = copy.deepcopy(network.all_node[0].c)
    group = None
    for layer in all_layers:
        if layer in specific_layers:
            if group is not None:
                layer_map.append(group)
            group = [layer]
            continue
        group.append(layer)
    layer_map.append(group)
    Hs = [H * alpha for i in range(L)] + [H for i in range(len(layer_map) - L)]
    loss_over_t = []
    real_time_over_t = []
    real_time = 0
    comm_over_t = []
    comm = 0
    difference_by_layer_over_t = []
    for t in range(iters):
        if t % sampling_f == 0:
            final_x = network.final(identity, total_data)
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append(loss(identity, whole_dataset, final_x, criterion, device, test_loader, tokenizer, True))
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(identity, total_data, sgd_round, learning_rate(sgd_round, exp, cte), tokenizer)
                node.x.load_state_dict(aggrigate([node.x, node.c, server_c],
                                                 [1, -learning_rate(sgd_round, exp, cte),
                                                  learning_rate(sgd_round, exp, cte)]))
        to_be_updated_layers = []
        for i in range(len(Hs)):
            if t % Hs[i] == 0:
                to_be_updated_layers += layer_map[i]
        if len(to_be_updated_layers) != 0:
            temp = network.final(identity, total_data)
            new_model = temp.state_dict()
            server_c_dict = server_c.state_dict()
            coeff = 1 / learning_rate(sgd_round, exp, cte) / Hs[i]
            for node in network.all_node:
                model_dict = node.x.state_dict()
                node_c_dict = node.c.state_dict()
                new_node_c = aggrigate([node.c, server_c, node.x_pre, node.x], [1, -1, coeff, -coeff])
                for layer in to_be_updated_layers:
                    model_dict[layer] = new_model[layer]
                    node_c_dict[layer] = new_node_c[layer]
                node.x.load_state_dict(model_dict)
                node.x_pre = copy.copy(node.x)
                node.c.load_state_dict(node_c_dict)
            temp_c = network.final_c(identity, total_data)
            new_server_c = temp_c.state_dict()
            for layer in to_be_updated_layers:
                server_c_dict[layer] = new_server_c[layer]
            server_c.load_state_dict(server_c_dict)
        real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t, difference_by_layer_over_t]

def parallel_gen_layer(identity, total_data, whole_dataset, exp, cte, H, iters, sampling_f, device, criterion,
                       num_worker, batch_size, test_loader, alpha, L, tokenizer, network):
    all_layers = list(network.all_node[0].x.state_dict().keys())
    if identity[0] == "LLM":
        #just using .01 of the dataset to compute training loss
        whole_dataset = whole_dataset.train_test_split(test_size=0.01)
        whole_dataset = whole_dataset['test']
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=32)
        specific_layers = [all_layers[i] for i in range(len(all_layers)) if all_layers[i].endswith("k_proj.weight") or
                           "tokens" in all_layers[i] or "lm_head" in all_layers[i]]
    else:
        whole_dataset = torch.utils.data.DataLoader(whole_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_worker)
        specific_layers = list(
            [all_layers[i] for i in range(len(all_layers)) if "conv" in all_layers[i] or "fc.weight" in all_layers[i] or 'linear.weight' in all_layers[i]])
    layer_map = []
    group = None
    # print(all_layers)
    for layer in all_layers:
        if layer in specific_layers:
            if group is not None:
                layer_map.append(group)
            group = [layer]
            continue
        group.append(layer)
    layer_map.append(group)
    Hs = [H * alpha for i in range(L)] + [H for i in range(len(layer_map) - L)]
    print(Hs)
    loss_over_t = []
    real_time_over_t = []
    loss_over_t_nodes = []
    noniid_over_t_nodes = []
    real_time = 0
    comm_over_t = []
    comm = 0
    for t in range(iters):
        print(t)
        if (t+1) % sampling_f == 0:
            # Each_node_model
            loss_over_nodes = []
            for node in network.all_node:
                final_x = copy.deepcopy(node.x)
                loss_over_nodes.append(
                    loss(identity, node.train_loader, final_x, criterion, device, test_loader, tokenizer, False))
            loss_over_t_nodes.append(loss_over_nodes)
            # Final_aggrigated_model
            final_x = network.final(identity, total_data)
            # print(final_x.state_dict())
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append(loss(identity, whole_dataset, final_x, criterion, device, test_loader, tokenizer, False))
            # Each_node_noniidness
            noniid_over_nodes = []
            for node in network.all_node:
                noniid_over_nodes.append(
                    loss(identity, node.train_loader, final_x, criterion, device, test_loader, tokenizer, False))
            noniid_over_t_nodes.append(noniid_over_nodes)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
            print(loss_over_t_nodes[-1])
            print(noniid_over_t_nodes[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(identity, total_data, sgd_round, learning_rate(sgd_round, exp, cte), tokenizer)
        to_be_updated_layers = []
        for i in range(len(Hs)):
            if (t + 1) % Hs[i] == 0:
                to_be_updated_layers += layer_map[i]
        if len(to_be_updated_layers) != 0:
            temp = network.final(identity, total_data)
            new_model = temp.state_dict()
            for node in network.all_node:
                model_dict = node.x.state_dict()
                for layer in to_be_updated_layers:
                    model_dict[layer] = new_model[layer]
                node.x.load_state_dict(model_dict)
        real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t, loss_over_t_nodes, noniid_over_t_nodes]
