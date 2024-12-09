
import struct
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from numba import njit, jit
from cython_decoder import cython_sc_decoding

def numpy_to_bit(array: np.ndarray) -> np.ndarray:
    tmp_byte=np.frombuffer(array.tobytes(),dtype=np.uint8)
    bit_stream = ''.join(format(byte, '08b') for byte in tmp_byte)
    bit_array = np.array([int(bit) for bit in bit_stream], dtype=np.int8)
    return bit_array

def bit_to_numpy(bit_array: np.ndarray) -> np.ndarray:
    bit_stream = ''.join(str(int(bit)) for bit in bit_array)
    byte_array_back = np.array([int(bit_stream[i:i+8], 2) for i in range(0, len(bit_stream), 8)], dtype=np.uint8)
    int_array_back = np.frombuffer(byte_array_back.tobytes(), dtype=np.float32)
    return int_array_back

'''
Encoding and decoding function
'''
@jit(nopython=True)
def encode(u: np.ndarray) -> np.ndarray:
    N = u.shape[0]  # Get the length of u
    n = int(np.log2(N))  # Calculate the log base 2 of N

    if n == 1:
        x = np.array([(u[0] + u[1]) % 2, u[1]],dtype=np.int8)
        return x
    else:
        x1 = encode(np.mod(u[:N//2] + u[N//2:], 2))
        x2 = encode(u[N//2:])
        x = np.concatenate((x1, x2))
        return x

@jit(nopython=True)
def rvsl(y: np.ndarray) -> np.ndarray:
    N = y.shape[0]
    if N == 2:
        return y
    else:
        return np.concatenate((rvsl(y[0:N:2]), rvsl(y[1:N:2])))

def data_process(array):
    bit_array = numpy_to_bit(array)
    array_len = len(bit_array)
    current_array = []
    for i in range(0, array_len, 512):
        sub_array = bit_array[i:i+512]
        if len(sub_array) < 512:
            padding = np.ones((512 - len(sub_array)), dtype=bit_array.dtype)
            sub_array = np.concatenate((sub_array, padding))
        current_array.append(sub_array)
    current_idx = i // 512 + 1
    return current_array, current_idx, array_len

def numpy_array_to_udp_packet(bit_array):
    # Convert the numpy array of bits to a string of bits
    bit_string = ''.join(str(int(bit)) for bit in bit_array)
    # Convert the bit string to bytes
    byte_array = bytearray(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
    return byte_array

def udp_packet_to_numpy_array(packet):
    # Convert the byte array back to a bit string
    bit_string = ''.join(format(byte, '08b') for byte in packet)
    # Convert the bit string to a numpy array of floats
    bit_array = np.array([int(bit) for bit in bit_string], dtype=np.int8)
    return bit_array

def data_generate(bit_array:np.ndarray, data_idx:np.ndarray) -> np.ndarray:
    u=np.zeros(1024,dtype=np.int8)
    u[data_idx] = bit_array
    x = encode(u)
    x = rvsl(x)
    return x

def codeword_generate(array_dict, data_idx):
    split_bit = []
    codeword_idx = []
    bit_array_len = []
    codeword_idx.append(0)
    array_process = partial(data_process)
    with ProcessPoolExecutor() as executor:
        for current_array, current_idx, array_len in executor.map(array_process, [array_dict[name] for name in array_dict]):
            split_bit.extend(current_array)
            codeword_idx.append(codeword_idx[-1] + current_idx)
            bit_array_len.append(array_len)
    executor.shutdown(wait=True)
    del executor, array_process
    encode_partial = partial(data_generate, data_idx=data_idx)
    with ProcessPoolExecutor() as executor:
        codeword = list(executor.map(encode_partial, split_bit))
    executor.shutdown(wait=True)

    del executor, encode_partial
    return np.array(codeword,dtype=np.int8), codeword_idx, bit_array_len

def packet_diffusion(codeword, block_len, packet_idx):
    codeword_len = codeword[0].shape[0]
    udp_packet = []
    for idx, i in enumerate(range(0, codeword_len, block_len)):
        # tmp_packet = np.concatenate([tmp_codeword[i:i+block_len] for tmp_codeword in codeword])
        tmp_packet = codeword[:, packet_idx[i:i+block_len]].flatten()
        tmp_udp_packet = struct.pack("I",idx) + numpy_array_to_udp_packet(tmp_packet)
        udp_packet.append(tmp_udp_packet)
    return udp_packet

def encoder_udp(array_dict, data_idx, block_len, packet_idx):
    codeword, codeword_idx, bit_array_len = codeword_generate(array_dict, data_idx)
    udp_packet = packet_diffusion(codeword, block_len, packet_idx)
    return udp_packet, codeword_idx, bit_array_len



def packet_aggregation(udp_packet, packet_idx, block_len, data_idx, freeze_idx, codeword_idx, bit_array_len):
    sort_idx = [struct.unpack("I", tmp_packet[:4])[0] for tmp_packet in udp_packet]
    packet_data_del = np.array([udp_packet_to_numpy_array(tmp_packet[4:]) for _, tmp_packet in sorted(zip(sort_idx, udp_packet))])
    packet_data = np.ones((int(1024/block_len), len(packet_data_del[0])))*0.5
    for i, tmp_idx in enumerate(sorted(sort_idx)):
        packet_data[tmp_idx] = packet_data_del[i]
    
    restore_codeword = []
    inverse_packet_idx = np.argsort(packet_idx)
    for i in range(0, packet_data.shape[1],block_len):
        tmp_codeword = packet_data[:,i:i+block_len].flatten()
        restore_codeword.append(tmp_codeword[inverse_packet_idx])

    decode_partial = partial(decoding, freeze_idx=freeze_idx, data_idx=data_idx)
    with ProcessPoolExecutor() as executor:
        decoding_data = np.array(list(executor.map(decode_partial, restore_codeword)),dtype=np.int8)
    del executor, decode_partial

    restore_array = []
    for i, array_len in enumerate(bit_array_len):
        tmp_array = np.concatenate(decoding_data[codeword_idx[i]:codeword_idx[i+1]])[:array_len]
        restore_array.append(bit_to_numpy(tmp_array))
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
