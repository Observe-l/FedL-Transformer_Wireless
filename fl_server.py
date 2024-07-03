# Required Libraries

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
import random
import socket

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from util.get_dataset import get_tr_test_data
from util.data_transfer import tcp_server, tcp_sender
from util.data_transfer import udp_server, udp_sender

SERVER_IP = '192.168.1.110'
CLIENT1_IP = '192.168.1.111'
CLIENT2_IP = '192.168.1.199'
CLIENT3_IP = '192.168.1.107'
CLIENT4_IP = '192.168.1.192'
CLIENT_IP = [CLIENT1_IP, CLIENT2_IP, CLIENT3_IP, CLIENT4_IP]

SERVER_PORT = 19998
CLIENT_PORT = 19999

CLIENT_NUM = 4

# Transformer Definition (Dependencies)

# Multi-head attention with Q, K, V
class multiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads):
        super(multiHeadAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.WqL = []
        self.WkL = []
        self.WvL = []

        for i in range(self.num_heads):
            Wq_init = tf.random_normal_initializer()
            Wq = tf.Variable(initial_value=Wq_init(shape=(int(input_shape[-1]), self.key_dim), dtype="float32"), trainable=True)
            self.WqL.append(Wq)

            Wk_init = tf.random_normal_initializer()
            Wk = tf.Variable(initial_value=Wk_init(shape=(int(input_shape[-1]), self.key_dim), dtype="float32"), trainable=True)
            self.WkL.append(Wk)

            Wv_init = tf.random_normal_initializer()
            Wv = tf.Variable(initial_value=Wv_init(shape=(int(input_shape[-1]), int(input_shape[-1])), dtype="float32"), trainable=True)
            self.WvL.append(Wv)

        Wlt_init = init = tf.random_normal_initializer()
        self.Wlt = tf.Variable(initial_value=Wlt_init(shape=((self.num_heads * int(input_shape[-1])), int(input_shape[-1])), dtype="float32"), trainable=True)

    def call(self, inputs):

        # inputs : batch_size x time_steps x dim
        x = inputs

        # transform for generating Q,K,V : (batch_size * time_steps) x dim
        x_tran = tf.reshape(x, [-1])
        x_tran = tf.reshape(x_tran, [-1, int(inputs.shape.as_list()[-1])])

        a_xL = []

        # Generate Query, Key and Value corresponding to each attention head
        for i in range(self.num_heads):

            # Query : batch_size x time_steps x dq
            xq = tf.matmul(x_tran, self.WqL[i])
            xq = tf.reshape(xq, [-1, int(inputs.shape.as_list()[-2]), int(xq.shape.as_list()[-1])])

            # Key : batch_size x time_steps x dk
            xk = tf.matmul(x_tran, self.WkL[i])
            xk = tf.reshape(xk, [-1, int(inputs.shape.as_list()[-2]), int(xk.shape.as_list()[-1])])

            # Value : batch_size x time_steps x dv
            xv = tf.matmul(x_tran, self.WvL[i])
            xv = tf.reshape(xv, [-1, int(inputs.shape.as_list()[-2]), int(xv.shape.as_list()[-1])])

            # Transposing each key in a batch (xk_t : batch_size x dk x time_steps)
            xk_t = tf.transpose(xk, perm=[0, 2, 1])

            # Computing scaled dot product self attention of each time step in each training sample (s_a : batch_size x time_steps x time_steps)
            s_a = tf.math.multiply(tf.keras.layers.Dot(axes=(1, 2))([xk_t, xq]), (1/self.key_dim))

            # Applying Softmax Layer to the self attention weights for proper scaling (sft_s_a : batch_size x time_steps x time_steps)
            sft_s_a = tf.keras.layers.Softmax(axis=2)(s_a)

            # Computing attention augmented values for each time step and each training sample (a_x : batch_size x time_steps x dim)
            a_xL.append(tf.keras.layers.Dot(axes=(1, 2))([xv, sft_s_a]))

        # Concatenate and applying linear transform for making dimensions compatible
        a_x = tf.concat(a_xL, -1)

        # Transform to shape a_x_tran : ((batch_size x time_steps) x (dim x num_heads))
        a_x_tran = tf.reshape(a_x, [-1])
        a_x_tran = tf.reshape(a_x_tran, [-1, (self.num_heads*int(inputs.shape.as_list()[-1]))])

        # Get the dimensions compatible after applying linear transform
        a_x_tran = tf.matmul(a_x_tran, self.Wlt)
        a_x_tran = tf.reshape(a_x_tran, [-1, int(inputs.shape.as_list()[-2]), int(inputs.shape.as_list()[-1])])

        return a_x_tran


# Transformer Block implemented as a Layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = multiHeadAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionEmbeddingLayer(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.position_embedding_layer = layers.Embedding(
            input_dim=(sequence_length), output_dim=output_dim
        )
        self.sequence_length = sequence_length

    def call(self, inputs):
        position_indices = tf.range(self.sequence_length)  #tf.range(1, self.sequence_length + 1, 1)
        embedded_words = inputs
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
    

# Creating the model
# Initializing the transformer model
def get_transformer_model(num_features, num_attn_heads, hidden_layer_dim, num_transformer_blocks, time_dim):
  transformer_blocks = []

  for i in range(num_transformer_blocks):
      transformer_blocks.append(TransformerBlock(num_features, num_attn_heads, hidden_layer_dim))

  # Model
  inputs = layers.Input(shape=(time_dim, num_features,))
  x = inputs

  # Trainable Embedding
  embedding_layer = PositionEmbeddingLayer(50, num_features)
  x = embedding_layer(x)

  for i in range(num_transformer_blocks):
      x = transformer_blocks[i](x)

  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(32, activation="relu")(x)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(1)(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  optim = keras.optimizers.SGD(learning_rate=0.0001)
  model.compile(optimizer=optim, loss='mse', metrics=['mse'])

  return model

def aggregate_weights(client_weights):
    """Aggregate the weights from multiple clients by averaging them.
    
    Args:
        client_weights (list): A list of lists containing the weights from each client.
    
    Returns:
        list: A list containing the averaged weights.
    """
    # Stack the weights along a new dimension
    stacked_weights = [np.stack([client_weights[j][i] for j in range(len(client_weights))], axis=0, dtype=np.float32) for i in range(len(client_weights[0]))]
    
    # Calculate the average along the new dimension
    averaged_weights = [np.average(weight, axis=0) for weight in stacked_weights]
    
    return averaged_weights

# Can use following input arguments
#num_attn_heads = 3
#hidden_layer_dim = 32  # Hidden layer size in feed forward network inside transformer
#num_transformer_blocks = 3
#num_features : Number of features at each time step (in the case of RUL prediction its number of sensors). Last dimension
#               of the dataset
#              Extract Using: num_features = np.asarray(X_tr).astype(np.float32).shape[-1]

#time_dim : History window size (Secondlast dimension of the dataset)
#              Extract Using: time_dim = np.asarray(X_tr).astype(np.float32).shape[-2]

if __name__ == "__main__":
    #Server Side
    #Datapaths (Put datapaths here)
    tr_dp_1 = './dataset/train_FD001.txt'
    te_dp_1 = './dataset/test_FD001.txt'
    gt_dp_1 = './dataset/RUL_FD001.txt'
    # FL parameters (Set These)
    C = 5
    num_total_clients = 2
    # num_clients_per_round = C * num_total_clients
    num_clients_per_round = 2
    num_comm_rounds = 10

    ###Define model
    # Can use following input arguments
    #num_attn_heads = 3
    #hidden_layer_dim = 32  # Hidden layer size in feed forward network inside transformer
    #num_transformer_blocks = 3
    #num_features : Number of features at each time step (in the case of RUL prediction its number of sensors). Last dimension
    #               of the dataset
    #              Extract Using: num_features = np.asarray(X_tr).astype(np.float32).shape[-1]

    #time_dim : History window size (Secondlast dimension of the dataset)
    #              Extract Using: time_dim = np.asarray(X_tr).astype(np.float32).shape[-2]



    '''
    UDP version
    Create UDP socket
    '''
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('',SERVER_PORT))

    X_tr, Y_tr, X_test_1, Y_test_1, cov_adj_mat_1 = get_tr_test_data(tr_dp_1, te_dp_1, gt_dp_1)
    global_model = get_transformer_model(num_features=np.asarray(X_tr).astype(np.float32).shape[-1], 
                                        num_attn_heads=3, 
                                        hidden_layer_dim=32, 
                                        num_transformer_blocks=3, 
                                        time_dim=np.asarray(X_tr).astype(np.float32).shape[-2])
    server_info = {"node":999}
    weight_len = len(global_model.weights)
    ###Communication Rounds Loop
    for i in range(num_comm_rounds):

        # Get Weights from all Clients, Create a dictionary and a list
        client_weights = {client_id:{} for client_id in range(CLIENT_NUM)}
        # Receive weight from client
        start_flag = True
        while True:
            tmp_weight, client_info = udp_server(server_socket,start_flag=start_flag)
            # Finish the loop when timeout
            if client_info == 'complete':
                break
            start_flag = False
            # Add weights into the dictionary
            client_weights[client_info['node']][client_info['weight']] = tmp_weight
            # Check whether complete transfer
            if all(len(sub_dict) == weight_len for sub_dict in client_weights.values()):
                break
        # Assign 0 to those lost weight
        for tmp_client_id, tmp_client_weight in client_weights.items():
            tmp_weight_id = tmp_client_weight.keys()
            # Find the lost packet
            lost_weight = set(range(weight_len)) - set(tmp_weight_id)
            for tmp_idx in lost_weight:
                tmp_shape = global_model.weights[tmp_idx].numpy().shape
                tmp_weight = np.zeros(tmp_shape, dtype=np.float32)
                client_weights[tmp_client_id][tmp_idx] = tmp_weight
        # Sort the dictionary and add to the list
        # sort_weight_dict = {tmp_client_id:dict(sorted(tmp_client_weight.items())) for tmp_client_id, tmp_client_weight in client_weights.items()}
        # Sort the dictionary and add to the list
        weight_list = [list(dict(sorted(tmp_client_weight.items())) for  tmp_client_weight in client_weights.values())]
        
        print(f"Aggregate weight for round {i}")
        aggregate_weight = aggregate_weights(weight_list)

        for weight_id in range(weight_len):
            global_model.weights[weight_id].assign(aggregate_weight[weight_id])

        # Send Updated Model Weights to Client
        #---
        # Put code here, serialize aggregate_weight and send via TCP/UDP
        #---
        for client_ip in CLIENT_IP:
            for weight_id in range(weight_len):
                server_info['weight'] = weight_id
                # Send weight to client via UDP
                udp_sender(global_model.weights[weight_id].numpy(),client_ip,CLIENT_PORT,server_info)