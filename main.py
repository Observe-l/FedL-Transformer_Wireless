import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

import os
import json
import argparse
import logging
import datetime
import scipy.io
from pathlib import Path
from collections import defaultdict

from util.data_process import codeword_generate, packet_diffusion, encoder_udp, packet_aggregation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

def get_args():
    parser = argparse.ArgumentParser()
    # General Parameters
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--samples_per_round", type=float, default=0.2)
    parser.add_argument("--comm_round", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--log_file_name", type=str, default=None)
    parser.add_argument("--algo", type=str, default="fedavg")
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--test_round", type=int, default=0)

    # Coding Parameters
    parser.add_argument("--coding", type=bool, default=False)
    parser.add_argument("--cordeword_len", type=int, default=1024)
    parser.add_argument("--rate", type=float, default=0.5)

    # Communication Parameters
    parser.add_argument("--loss_rate", type=float, default=0.01)
    return parser.parse_args()

def init_nets(num_nodes):
    from models.vit_small import ViT
    nets = {net_i: None for net_i in range(num_nodes)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for net_i in range(num_nodes):
        net = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=64,
            depth=6,
            heads=8,
            mlp_dim=32,
            dropout=0.1,
            emb_dropout=0.1
        )
        nets[net_i] = net.to(device)
    return nets

def get_divided_dataloader(fds:FederatedDataset, num_nodes):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def train_transforms(batch):
        transforms = transform_train
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch

    def test_transforms(batch):
        transforms = transform_test
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch
    
    train_loader={net_i: None for net_i in range(num_nodes)}
    test_loader={net_i: None for net_i in range(num_nodes)}
    for net_i in range(num_nodes):
        partition_train_test = fds.load_partition(net_i, "train").train_test_split(0.1)
        partition_train = partition_train_test["train"].with_transform(train_transforms)
        partition_test = partition_train_test["test"].with_transform(test_transforms)
        train_loader[net_i] = DataLoader(partition_train, batch_size=128, shuffle=True, num_workers=4)
        test_loader[net_i] = DataLoader(partition_test, batch_size=128, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def train_net(net, train_loader, test_loader, epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
    '''Evaluate the model'''
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / float(total)
    return test_acc

def local_train_net_per(nets, selected, fds, epochs=5, logger=None):
    train_loader, test_loader = get_divided_dataloader(fds, len(nets))
    total_num = sum([len(train_loader[net_i]) for net_i in selected])
    net_freq = {net_i: None for net_i in selected}
    for net_i in selected:
        test_acc = train_net(nets[net_i], train_loader[net_i], test_loader[net_i], epochs)
        net_freq[net_i] = len(train_loader[net_i].dataset) / total_num
        logger.info(f"Node {net_i} Test Accuracy: {test_acc}")
    return net_freq
    
def fed_avg(nets, selected, global_model, loss_rate, info_ni, freeze_ni, shuffle_ni, net_freq):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_para = global_model.state_dict()
    udp_packet = {net_i: None for net_i in selected}

    # Generate the udp packet
    for net_i in selected:
        weight_dict = {name: param.cpu().detach().numpy() for name, param in nets[net_i].state_dict().items()}
        udp_packet[net_i], codeword_idx, bit_array_len = encoder_udp(weight_dict, info_ni, 8, shuffle_ni)
    
    # Data transfer with packet loss
    restore_dict = {net_i: None for net_i in selected}
    for net_i in selected:
        received_packet = [packet for packet in udp_packet[net_i] if np.random.rand() > loss_rate]
        restore_array = packet_aggregation(received_packet, shuffle_ni, 8, info_ni, freeze_ni, codeword_idx, bit_array_len)
        for i, name in enumerate(global_para):
            restore_dict[net_i][name] = torch.tensor(restore_array[i].reshape(weight_dict[name].shape)).to(device)
    
    # Aggregate the model
    for idx, net_i in enumerate(selected):
        if idx == 0:
            for key in global_para:
                global_para[key] = restore_dict[net_i][key] * net_freq[net_i]
        else:
            for key in global_model.state_dict():
                global_para[key] += restore_dict[net_i][key] * net_freq[net_i]
    
    global_model.load_state_dict(global_para)
        
def broadcast_parameters(nets, global_model, selected, loss_rate, info_ni, freeze_ni, shuffle_ni):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Generate the udp packet from the global model
    global_para = global_model.state_dict()
    weight_dict = {name: param.cpu().detach().numpy() for name, param in global_para.items()}
    udp_packet, codeword_idx, bit_array_len = encoder_udp(weight_dict, info_ni, 8, shuffle_ni)

    # Data transfer with packet loss
    for net_i in selected:
        received_packet = [packet for packet in udp_packet if np.random.rand() > loss_rate]
        restore_array = packet_aggregation(received_packet, shuffle_ni, 8, info_ni, freeze_ni, codeword_idx, bit_array_len)

        restore_dict = {name: torch.tensor(restore_array[i].reshape(weight_dict[name].shape)).to(device) for i, name in enumerate(global_para)}
        nets[net_i].load_state_dict(restore_dict)

def compute_accuracy_loss(nets, fds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_divided_dataloader(fds, len(nets))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_i in nets:
        criterion = nn.CrossEntropyLoss().to(device)
        correct, total, total_loss, batch_count = 0, 0, 0, 0
        nets[net_i].eval()
        with torch.no_grad():
            for data in test_loader[net_i]:
                inputs, labels = data["img"].to(device), data["label"].to(device)
                outputs = nets[net_i](inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                batch_count += 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_results[net_i]["loss"] = total_loss / batch_count
        test_results[net_i]["correct"] = correct
        test_results[net_i]["total"] = total
    
    test_total_correct = sum([test_results[net_i]["correct"] for net_i in test_results])
    test_total_samples = sum([test_results[net_i]["total"] for net_i in test_results])
    test_avg_loss = np.mean([test_results[net_i]["loss"] for net_i in test_results])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [test_results[net_i]["correct"] / test_results[net_i]["total"] for net_i in test_results]
    return test_results, test_avg_loss, test_avg_acc, test_all_acc

def main():
    args = get_args()
    logging.info(f"Number of Nodes: {args.num_nodes}")
    logging.info(f"Samples per Round: {args.samples_per_round}")
    logging.info(f"Communication Round: {args.comm_round}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Coding: {args.coding}")

    num_nodes = args.num_nodes
    samples_per_round = args.samples_per_round
    epochs = args.epochs
    comm_round = args.comm_round
    loss_rate = args.loss_rate
    codeword_len = args.codeword_len
    rate = args.rate

    k = round(codeword_len * rate)
    c_1024 = np.load("c_1024.npy")
    coding_list = scipy.io.loadmat("1024-3db-d=2-mean.mat")["count_number"]
    coding_index = np.argsort(coding_list[:,1])
    info_idx = coding_index[:k]
    freeze_idx = coding_index[k:]

    info_ni = np.sort(info_idx)
    freeze_ni = np.sort(freeze_idx)

    # Save the log file
    os.makedirs(args.logdir, exist_ok=True)
    if args.log_file_name is None:
        args.log_file_name = f"{args.algo}_n{num_nodes}_s{samples_per_round}_r{comm_round}_e{epochs}_loss{loss_rate}-{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M-%S')}.log"
    log_path = os.path.join(args.logdir, args.log_file_name)
    logging.basicConfig(filename=log_path, 
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filemode='w',
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("#"*50)
    
    logger.info("Partitioning the dataset")
    fds = FederatedDataset(
        dataset="cifar10",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_nodes,
                partition_by="label",
                alpha=0.1,
                seed=42,
                min_partition_size=0,
            ),
        },
    )

    save_path = Path(f"results/{args.algo}")
    save_path.mkdir(parents=True, exist_ok=True)

    result_dict = defaultdict(list)
    eval_freq = args.eval_freq
    test_round = args.test_round

    if args.algo == "fedavg":
        logger.info("Using FedAvg Algorithm")

        nets = init_nets(num_nodes)
        global_model = init_nets(1)[0]
        for net_i in nets:
            nets[net_i].load_state_dict(global_model.state_dict())

        for round in range(comm_round):
            # Generate selected data
            arr = np.arange(num_nodes)
            np.random.shuffle(arr)
            selected = arr[:int(num_nodes * samples_per_round)]
            global_para = global_model.state_dict()

            # Train the local model
            net_freq = local_train_net_per(nets, selected, fds, epochs, logger=logger)

            # Transfer and aggregate the model
            fed_avg(nets, selected, global_model, loss_rate, info_ni, freeze_ni, c_1024, net_freq)

            # Evaluate the model
            if (round+1) >= test_round and (round+1) % eval_freq == 0:
                broadcast_parameters(nets, global_model, arr, loss_rate, info_ni, freeze_ni, c_1024)
                test_results, test_avg_loss, test_avg_acc, test_all_acc = compute_accuracy_loss(nets, fds)
                logger.info(f">>>>>> Round {round} <<<<<<")
                logger.info(f">> Global Model Test accuracy: {test_avg_acc}")
                logger.info(f">> Global Model Test avg loss: {test_avg_loss}")

                result_dict["test_acc"].append(test_avg_acc)
                result_dict["test_loss"].append(test_avg_loss)
                result_dict["test_all_acc"].append(test_all_acc)
        
        # Save the results
        file_name = f"n{num_nodes}_s{samples_per_round}_r{comm_round}_e{epochs}_loss{loss_rate}.json"
        with open(str(save_path / file_name), "w") as f:
            json.dump(result_dict, f, indent=4)
 
if __name__ == 'main':
    main()
