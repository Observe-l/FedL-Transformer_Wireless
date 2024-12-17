import struct
import torch
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from cython_decoder import cython_sc_decoding

def get_gn(length:int, device) -> np.ndarray:
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
    array_len = bit_array.shape[0] * 8
    if bit_array.shape[0] % 64 != 0:
        padding = 255 * np.ones((64 - bit_array.shape[0] % 64), dtype=bit_array.dtype)
        bit_array = np.concatenate((bit_array, padding))
    bit_array = np.unpackbits(bit_array)
    bit_array = bit_array.reshape(-1, 512)
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
        hd_dec, 1024, 10, 512, frozen_val, delete_num, 0, pro_prun
    )

    # Extract the output for data_idx from hd_dec_result
    data_out = hd_dec_result[data_idx]
    return data_out
