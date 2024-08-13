import pandas as pd
import multiprocessing as mp
from tensorflow.keras.datasets import mnist as mnist_dataset
from NetworkClass import *
from NodeClass import *
from DSGDAlg import *
from Functions import *
import os
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# inp = sys.argv
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


# load the topology from ./config
with open("config/%s" % config_file_name, "r") as f:
    fp = f.readlines()
    node_connection = {int(k): v for k, v in json.loads(fp[0]).items()}
    connection_delay = {int(k): v for k, v in json.loads(fp[1]).items()}
    gap = json.loads(fp[2])
num_of_node = len(node_connection)
# setting Task:
if mps:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if identity[0] == "synthetic":
    criterion = nn.CrossEntropyLoss()
    tokenizer = None
    model = LogisticRegression(inp_dim, out_class).to(device)
    train_x = torch.empty((0, inp_dim), dtype=torch.float32)
    test_x = torch.empty((0, inp_dim), dtype=torch.float32)
    train_y = torch.empty((0), dtype=torch.int)
    test_y = torch.empty((0), dtype=torch.int)
    for i, c in enumerate(range(-out_class // 2, out_class // 2)):
        mvn = MultivariateNormal(loc=mean * c, covariance_matrix=cov)
        train_x = torch.cat((train_x, mvn.sample([each_class_train_data])))
        test_x = torch.cat((test_x, mvn.sample([each_class_test_data])))
        train_y = torch.cat((train_y, torch.ones([each_class_train_data], dtype=torch.long) * i))
        test_y = torch.cat((test_y, torch.ones([each_class_test_data], dtype=torch.long) * i))
    # train_y = F.one_hot(train_y)
    # test_y = F.one_hot(test_y)
    train_dataset = DummyDataset(train_x, train_y)
    test_dataset = DummyDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(out_class):
            boolArr = np.array(train_dataset.y) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == "cifar10":
    criterion = nn.CrossEntropyLoss()
    model = ResNet(ResidualBlock, [3, 3, 3]).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=False,
                                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    tokenizer = None
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(10):
            boolArr = np.array(train_dataset.targets) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == "cifar100":
    criterion = nn.CrossEntropyLoss()
    model = ResNet(ResidualBlock, [3, 3, 3], num_classes=100).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                  train=True,
                                                  transform=transform,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                 train=False,
                                                 transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    tokenizer = None
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(100):
            boolArr = np.array(train_dataset.targets) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == "SVHN":
    criterion = nn.CrossEntropyLoss()
    model = ResNet(ResidualBlock, [3, 3, 3], num_classes=10).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.SVHN(root='./data/',
                                              split='train',
                                              transform=transform,
                                              download=True)
    test_dataset = torchvision.datasets.SVHN(root='./data/',
                                             split='test',
                                             download=True,
                                             transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    tokenizer = None
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(10):
            boolArr = np.array(train_dataset.labels) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == "Mnist":
    criterion = nn.CrossEntropyLoss()
    model = ResNet(ResidualBlock, [3, 3, 3]).to(device)
    normalize = transforms.Normalize(mean=[0.485, ],
                                     std=[0.229, ])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=transform,
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    tokenizer = None
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        # data_split.append((0,total_data))
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(10):
            boolArr = np.array(train_dataset.targets) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == "ImageNet":
    criterion = nn.CrossEntropyLoss()
    model = torchvision.models.resnet50().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageNetKaggle("<YOUR_FOLDER>", "train", train_transform)
    test_dataset = ImageNetKaggle("<YOUR_FOLDER>", "val", test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False,
                                              num_workers=num_worker)
    tokenizer = None
    total_data = len(train_dataset)
    indices = np.arange(total_data)
    indices = np.random.permutation(indices)
    data_split = []
    dataset = torch.utils.data.Subset(train_dataset, indices)
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))
    if not iid:
        indices = []
        for n in range(1000):
            boolArr = np.array(train_dataset.targets) == n
            indices += list(np.where(boolArr)[0])
        dataset = torch.utils.data.Subset(train_dataset, indices)

elif identity[0] == 'LLM':
    criterion = nn.CrossEntropyLoss()
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        load_in_8bit=True,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_dataset("multi_nli")
    dataset = dataset.map(
        form_training_prompts,
        remove_columns=["promptID", "pairID", "premise_binary_parse", "premise_parse"
            , "hypothesis_binary_parse", "hypothesis_parse", "hypothesis", "premise", "label"],
        load_from_cache_file=False,
        desc="Generating text prompt",
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["validation_matched"]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    total_data = len(train_dataset)
    if not iid:
        dataset = train_dataset.sort('genre')
    else:
        dataset = train_dataset.shuffle(seed=42)
    data_split = []
    guid_to_split = range(0, total_data + 1, total_data // num_of_node)
    for node in range(num_of_node):
        data_split.append((guid_to_split[node], guid_to_split[node + 1]))


# %%
def my_func(f, stream, args):
    network = Network(model, [], node_connection, 0, 0)
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    for node in range(num_of_node):
        network.all_node.append(Node(identity, model, data_split[node], dataset, node_connection[node],
                                     gap[node], connection_delay[node], stream, device, criterion,
                                     num_worker, batch_size))
    return f(*args, network)


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    result = []

    stream = 1
    if mode == "cons":
        g = parallel_sgd_measure
    if mode == "sca":
        g = parallel_scaffold_layer
    elif mode == "gen":
        g = parallel_gen_layer
    else:
        g = parallel_sgd_layer
    for exp in lr_exp:
        input_arg = (g, stream, (identity, total_data, dataset, exp, cte, H, iteration,
                                 sampling_f, device, criterion, num_worker, batch_size, test_loader, alpha, L,
                                 tokenizer))
        if not child_process:
            res = []
            for run in range(repeat_simulation):
                res.append(my_func(*input_arg))
        else:
            with mp.Pool(repeat_simulation) as pool:
                res = pool.starmap_async(my_func, [input_arg for run in range(repeat_simulation)]).get()
        result.append(res)
        print("All done -", "Layer-wise - exp =", exp, repeat_simulation, "times at",
              datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))


    with open("result/"+simulation_result_file_name, 'w') as f:
        f.write(json.dumps(result))

