import struct
import gc
import torch
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from decoder_cuda import decoder_cuda

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

def codeword_generate(array_dict, data_idx, gn, device):
    split_bit = []
    codeword_idx = []
    bit_array_len = []
    codeword_idx.append(0)
    for name in array_dict:
        current_array, current_idx, array_len = data_process(array_dict[name], data_idx)
        split_bit.extend(current_array)
        codeword_idx.append(codeword_idx[-1] + current_idx)
        bit_array_len.append(array_len)
    split_bit = torch.tensor(np.array(split_bit), dtype=torch.float32).to(device)
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

def encoder_udp(array_dict, data_idx, gn, device):
    codeword, codeword_idx, bit_array_len = codeword_generate(array_dict, data_idx, gn, device)
    udp_packet = packet_diffusion(codeword)
    return udp_packet, codeword_idx, bit_array_len, codeword.shape[0]

def decoding(udp_packet, codeword_num, data_idx, freeze_idx, codeword_idx, bit_array_len, device, chunk_size=10000):
    udp_idx = []
    packet_del = []
    for tmp_packet in udp_packet:
        udp_idx.append(struct.unpack("I", tmp_packet[:4])[0])
        packet_del.append(udp_packet_to_numpy_array(tmp_packet[4:], codeword_num))
    packet_del = np.array(packet_del)
    packet_data = np.ones((1024, codeword_num)) * 0.5
    packet_data[udp_idx] = packet_del
    restore_codeword = packet_data.T

    codeword_torch = torch.tensor(restore_codeword, dtype=torch.float32).to(device)
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
