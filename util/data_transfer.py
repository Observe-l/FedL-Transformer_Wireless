import socket
import select
import struct
import numpy as np
import os
import tqdm
import time

UDP_IP = ''
TCP_IP = ''
buf = 65536

'''
start flag means begin to transfer first data, No timeout.
UDP frame:
1. Node ID, [0:4]
2. Weight ID, [4:8]
3. Shape info, [8:12+shape_size]
4. Array, [12+shape_size:16+array_length]

'''
def udp_server(sock, timeout = 0.5, start_flag=False):
    if start_flag:
        packet, cli_addr = sock.recvfrom(buf)
    else:
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            packet, cli_addr = sock.recvfrom(buf)
        else:
            packet = 'complete'
            info = 'complete'
            cli_addr = [None,None]
            return packet, info
    # Unpack the server and weight infomation
    info_node_id = struct.unpack('!I',packet[:4])[0]
    info_weight_id = struct.unpack('!I',packet[4:8])[0]
    info={"node":info_node_id,"weight":info_weight_id}

    # Unpack the shape size
    shape_size = struct.unpack('!I', packet[:4])[0]
    # Unpack the shape data
    shape_data = packet[4:4 + shape_size]
    shape = struct.unpack('!' + 'I' * (shape_size // 4), shape_data)

    # Unpack the array length
    array_length = struct.unpack('!I', packet[4 + shape_size: 8 + shape_size])[0]
    array_data = packet[8 + shape_size: 8 + shape_size + array_length]
    # Unpack the array data
    array = np.frombuffer(array_data, dtype=np.float32).reshape(shape)
    return array, info

def udp_sender(array, address, port, info):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Send the serveri information and weight information to check packet loss
    info_node_id = info['node']
    info_weight_id = info['weight']
    info_data = struct.pack('!I',info_node_id)+struct.pack('!I',info_weight_id)

    # Serialize the array shape and data
    shape = array.shape
    shape_data = struct.pack('!' + 'I' * len(shape), *shape)
    shape_size = len(shape_data)
    array_data = array.tobytes()
    array_length = len(array_data)

    # Pack all information into one byte stream
    packet = info_data + struct.pack('!I', shape_size) + shape_data + struct.pack('!I', array_length) + array_data

    # Send the packet
    sock.sendto(packet, (address, port))

    sock.close()



def tcp_server(conn: socket.socket) -> np.ndarray:
    # Get the shape array shape information
    shape_size = struct.unpack('!I', conn.recv(4))[0]
    shape_data = conn.recv(shape_size)
    shape = struct.unpack('!' + 'I' * (shape_size // 4), shape_data)
    # Get the lenth of the array
    array_len = struct.unpack('!I',conn.recv(4))[0]
    array_data = b''
    # Receive all the byte data
    while len(array_data) < array_len:
        packet = conn.recv(min(array_len - len(array_data),4096))
        if not packet:
            return None
        array_data += packet
    # Transfer byte data to numpy array
    array = np.frombuffer(array_data, dtype=np.float32).reshape(shape)
    return array

def tcp_sender(conn: socket.socket, array:np.ndarray) -> None:
    # Get the shape of the array
    shape = array.shape
    shape_data = struct.pack('!' + 'I' * len(shape), *shape)
    shape_size = len(shape_data)
    # Send the shape information
    conn.sendall(struct.pack('!I', shape_size))
    conn.sendall(shape_data)

    # Convert numpy array to bytes
    array_data = array.tobytes()
    # Length of the array
    array_length = len(array_data)
    # Send the array data
    conn.sendall(struct.pack('!I',array_length))
    conn.sendall(array_data)