import socket
import select
import struct
import numpy as np
import os
import tqdm
import time

UDP_IP = ''
TCP_IP = ''
buf = 1024

'''
start flag means begin to transfer first data, No timeout
'''
def udp_server(port, timeout = 5, start_flag=False):
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.bind((UDP_IP, port))
    if start_flag:
        rec, cli_addr = sk.recvfrom(buf)
    else:
        ready = select.select([sk], [], [], timeout)
        if ready[0]:
            rec, cli_addr = sk.recvfrom(buf)
        else:
            rec = b'complete'
            cli_addr = [None,None]
    return rec,cli_addr[0]

def udp_send(msg, ip, port):
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.sendto(msg,(ip,port))



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