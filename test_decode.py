import socket
import struct
import time
import scipy.io
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision.transforms import ToTensor
from flwr_datasets.visualization import plot_label_distributions
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cython_decoder import cython_sc_decoding
from decoder_cuda import decoder_cuda
import gc

num_nodes = 5

fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=50,
            partition_by="label",
            alpha=0.1,
            seed=42,
            min_partition_size=0,
        ),
    },
)

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

train_loader=[]
test_loader=[]
for i in range(50):
    partition_train_test = fds.load_partition(i, "train").train_test_split(0.1)
    partition_train = partition_train_test["train"].with_transform(train_transforms)
    partition_test = partition_train_test["test"].with_transform(test_transforms)
    # centralized_dataset = fds.load_split("test").with_transform(test_transforms)
    train_loader.append(DataLoader(partition_train, batch_size=256, shuffle=True, num_workers=16))
    test_loader.append(DataLoader(partition_test, batch_size=128, shuffle=False, num_workers=16))

from models.vit_small import ViT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = []
optimizer = []
scheduler = []
criterion = []
scaler = []
for i in range(num_nodes):
    net.append(ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 32,
        depth = 6,
        heads = 8,
        mlp_dim = 32,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device))


    optimizer.append(optim.Adam(net[i].parameters(), lr=0.001))
    scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], 5))
    criterion.append(nn.CrossEntropyLoss())
    scaler.append(torch.cuda.amp.GradScaler(enabled=True))

server_net = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 32,
    depth = 6,
    heads = 8,
    mlp_dim = 32,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                device: torch.device, 
                scaler: torch.cuda.amp.GradScaler, 
                optimizer: torch.optim.Optimizer,
                epoch: int,
                nodes: int):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in train_loader:
        inputs = batch["img"].to(device)
        labels = batch["label"].to(device)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_samples += labels.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
    print(f"Nodes: {nodes}, Epoch: {epoch},Train Loss: {total_loss / total_samples:.4f}, Train Accuracy: {total_correct / total_samples:.4f}")

def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader, 
                   criterion: nn.Module, 
                   device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_samples += labels.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
    print(f"Validation Loss: {total_loss / total_samples:.4f}, Validation Accuracy: {total_correct / total_samples:.4f}\n\t")

for cli in range(1):
    start_time = time.time()
    for i in range(10):
        train_model(net[cli], train_loader[cli], criterion[cli], device, scaler[cli], optimizer[cli], i, cli)
        # evaluate_model(net[cli], test_loader[cli], criterion[cli], device)
        scheduler[cli].step()
    print(f"Training time: {time.time()-start_time}")
    scheduler[cli] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[cli], 5)

def get_gn(length:int, device):
    n = int(np.log2(length))
    f=np.array([[1,0],[1,1]], dtype=bool)
    g_n = f
    for _ in range(n-1):
        g_n = np.kron(g_n, f)
    g_order = rvsl(np.arange(length))
    g_n = g_n[:, g_order]
    g_n = torch.tensor(g_n, dtype=torch.float32).to(device)
    return g_n

def rvsl(y: np.ndarray) -> np.ndarray:
    N = y.shape[0]
    if N == 2:
        return y
    else:
        return np.concatenate((rvsl(y[0:N:2]), rvsl(y[1:N:2])))

def data_process(array, data_idx):
    bit_array = np.frombuffer(array.tobytes(), dtype=np.uint8)
    info_len = int(len(data_idx) / 8)
    array_len = bit_array.shape[0] * 8
    if bit_array.shape[0] % info_len != 0:
        padding = 255 * np.ones((info_len - bit_array.shape[0] % info_len), dtype=bit_array.dtype)
        bit_array = np.concatenate((bit_array, padding))
    bit_array = np.unpackbits(bit_array)
    bit_array = bit_array.reshape(-1, len(data_idx))
    current_array = np.zeros((bit_array.shape[0], 1024), dtype=np.uint8)
    current_array[:, data_idx] = bit_array
    current_idx = bit_array.shape[0]
    return current_array, current_idx, array_len

def udp_packet_to_numpy_array(packet, codeword_num):
    bit_array = np.frombuffer(packet, dtype=np.uint8)
    bit_array = np.unpackbits(bit_array)
    bit_array = bit_array[:codeword_num]
    return bit_array

def codeword_generate(array_dict, data_idx, gn):
    split_bit = []
    codeword_idx = []
    bit_array_len = []
    codeword_idx.append(0)
    for name in array_dict:
        current_array, current_idx, array_len = data_process(array_dict[name], data_idx)
        split_bit.extend(current_array)
        codeword_idx.append(codeword_idx[-1] + current_idx)
        bit_array_len.append(array_len)
    split_bit = torch.tensor(np.array(split_bit), dtype=torch.float32).to("cuda:0")
    tmp_codeword = torch.matmul(split_bit, gn) % 2
    codeword = np.array(tmp_codeword.cpu().detach().numpy(), dtype=np.int8)
    del split_bit, tmp_codeword
    gc.collect()
    torch.cuda.empty_cache()
    return codeword, codeword_idx, bit_array_len

def packet_diffusion(codeword):
    if codeword.shape[0] % 8 != 0:
        padding = np.ones((8 - codeword.shape[0] % 8, 1024), dtype=codeword.dtype)
        udp_numpy = np.concatenate((codeword, padding)).T
    udp_numpy = np.packbits(udp_numpy.flatten()).reshape(1024,-1)
    udp_packet = [struct.pack("I", idx) + udp_numpy[idx].tobytes() for idx in range(udp_numpy.shape[0])]
    return udp_packet

def encoder_udp(array_dict, data_idx, gn):
    codeword, codeword_idx, bit_array_len = codeword_generate(array_dict, data_idx, gn)
    udp_packet = packet_diffusion(codeword)
    return udp_packet, codeword_idx, bit_array_len, codeword.shape[0]

def packet_aggregation(udp_packet, codeword_num, data_idx, freeze_idx, codeword_idx, bit_array_len):
    udp_idx = []
    packet_del = []
    for tmp_packet in udp_packet:
        udp_idx.append(struct.unpack("I", tmp_packet[:4])[0])
        packet_del.append(udp_packet_to_numpy_array(tmp_packet[4:], codeword_num))
    packet_del = np.array(packet_del)
    packet_data = np.ones((1024, codeword_num)) * 0.5
    packet_data[udp_idx] = packet_del

    restore_codeword = packet_data.T
    decode_partial = partial(decoding, freeze_idx=freeze_idx, data_idx=data_idx)
    with ProcessPoolExecutor() as executor:
        decoding_data = np.array(list(executor.map(decode_partial, restore_codeword)),dtype=np.int8)
    del executor, decode_partial
    restore_array = []
    for i, array_len in enumerate(bit_array_len):
        tmp_array = np.concatenate(decoding_data[codeword_idx[i]:codeword_idx[i+1]])[:array_len]
        bit_array = np.packbits(tmp_array)
        restore_array.append(np.frombuffer(bit_array.tobytes(), dtype=np.float32))
    return restore_array


def decoding(bit_array, freeze_idx, data_idx):
    # Prepare the necessary arrays and values
    bit_array = 1-2*bit_array
    lr0 = np.exp(-(bit_array - 1)**2)
    lr1 = np.exp(-(bit_array + 1)**2)
    lr0_post = lr0 / (lr0 + lr1)
    lr1_post = lr1 / (lr0 + lr1)
    delete_num = 1024 - len(bit_array)
    hd_dec = np.zeros(1024, dtype=np.float64)
    frozen_val = np.zeros(len(freeze_idx), dtype=np.float64)
    pro_prun = np.zeros((1, 2 * 1024 + 1), dtype=np.float64)

    # Call the optimized Cython function
    i_scen_sum, hd_dec_result = cython_sc_decoding(
        lr0_post, lr1_post, freeze_idx.astype(np.float64),
        hd_dec, 1024, 10, len(freeze_idx), frozen_val, delete_num, 0, pro_prun
    )

    # Extract the output for data_idx from hd_dec_result
    data_out = hd_dec_result[data_idx]
    return data_out

def decoding_cuda(udp_packet, codeword_num, data_idx, freeze_idx, codeword_idx, bit_array_len, chunk_size=10000):
    udp_idx = []
    packet_del = []
    for tmp_packet in udp_packet:
        udp_idx.append(struct.unpack("I", tmp_packet[:4])[0])
        packet_del.append(udp_packet_to_numpy_array(tmp_packet[4:], codeword_num))
    packet_del = np.array(packet_del)
    packet_data = np.ones((1024, codeword_num)) * 0.5
    packet_data[udp_idx] = packet_del
    restore_codeword = packet_data.T

    codeword_torch = torch.tensor(restore_codeword, dtype=torch.float32).to("cuda:0")
    bit_array = 1 - 2 * codeword_torch.flatten()
    lr0 = torch.exp(-(bit_array - 1)**2)
    lr1 = torch.exp(-(bit_array + 1)**2)
    lr0_post = lr0 / (lr0 + lr1)
    lr1_post = lr1 / (lr0 + lr1)

    lr0_post = lr0_post.cpu().detach().numpy().astype(np.float64)
    lr1_post = lr1_post.cpu().detach().numpy().astype(np.float64)
    delete_num = 1024 - restore_codeword.shape[1]
    hd_dec = np.zeros_like(bit_array.cpu().detach().numpy()).astype(np.float64)
    frozen_val = np.zeros(len(freeze_idx), dtype=np.float64)
    del codeword_torch, bit_array, lr0, lr1
    gc.collect()
    torch.cuda.empty_cache()
    hd_dec_result = decoder_cuda(lr0_post, lr1_post, freeze_idx.astype(np.float64), hd_dec, 1024, 10, len(freeze_idx), frozen_val, delete_num, 0, restore_codeword.shape[0], chunk_size)

    hd_dec_result = hd_dec_result.reshape(restore_codeword.shape).astype(np.int8)
    hd_dec_result = hd_dec_result[:,data_idx]

    restore_array = []
    for i, array_len in enumerate(bit_array_len):
        tmp_array = np.concatenate(hd_dec_result[codeword_idx[i]:codeword_idx[i+1]])[:array_len]
        bit_array = np.packbits(tmp_array)
        restore_array.append(np.frombuffer(bit_array.tobytes(), dtype=np.float32))
    return restore_array


# Export all weights to numpy arrays
weights_dict = {name: param.cpu().detach().numpy() for name, param in net[0].state_dict().items()}
N = 1024
n = 10
rate = 1/2
K = round(N*rate)
c_1024 = np.load('c_1024.npy')
coding_list = scipy.io.loadmat("1024-3db-d=2-mean.mat")["count_number"]
coding_index = np.argsort(coding_list[:,1])
info_idx = coding_index[:K]
freeze_idx = coding_index[K:]

# sort the final index
info_ni = np.sort(info_idx)
freeze_ni = np.sort(freeze_idx)
gn = get_gn(1024, "cuda:0")

time1 = time.time()
udp_packet, codeword_idx, bit_array_len, codeword_num = encoder_udp(weights_dict, info_ni, gn)
time2 = time.time()
print(f"Encode time: {time2-time1}")

cuda_array = decoding_cuda(udp_packet, codeword_num, info_ni, freeze_ni, codeword_idx, bit_array_len,10000)
time3 = time.time()
print(f"GPU decode time: {time3-time2}")

# restore_array = packet_aggregation(udp_packet, codeword_num, info_ni, freeze_ni, codeword_idx, bit_array_len)
# time4 = time.time()
# print(f"CPU decode time: {time4-time3}")