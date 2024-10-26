from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import math
import random
import socket

import os
import optparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from util.get_dataset import get_tr_test_data

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
    stacked_weights = [np.stack([client_weights[j][i] for j in range(len(client_weights))], axis=0).astype(np.float32) for i in range(len(client_weights[0]))]
    
    # Calculate the average along the new dimension
    averaged_weights = [np.average(weight, axis=0) for weight in stacked_weights]
    
    return averaged_weights

keras.utils.get_custom_objects().update({"PositionEmbeddingLayer": PositionEmbeddingLayer})
keras.utils.get_custom_objects().update({"TransformerBlock":TransformerBlock})


def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("-l","--lose",default=0.05,type=float,help="Node ID")
    optParse.add_option("-b","--batch",default=1024,type=int,help="batch size")
    optParse.add_option("-e","--epoch",default=1,type=int,help="epoch")
    options, args = optParse.parse_args()
    return options

if __name__ == "__main__":
    #Client client
    #Datapaths (Put datapaths here)
    options = get_options()

    node_id = [0,1,2,3]
    # FL parameters (Set These)
    B = options.batch
    E = options.epoch
    if options.lose == 0.03:
        drop_weight = 14
    elif options.lose == 0.04:
        drop_weight = 80
    elif options.lose == 0.05:
        drop_weight = 317
    elif options.lose == 0.06:
        drop_weight = 876
    else:
        drop_weight = 0
    weight_size = 17027
    num_comm_rounds = 1000
    local_model = []
    x_tr_list = []
    y_tr_list = []
    for tmp_id in node_id:
        tr_dp_1 = f'./dataset/node{tmp_id}/train_FD001.txt'
        te_dp_1 = f'./dataset/node{tmp_id}/test_FD001.txt'
        gt_dp_1 = f'./dataset/node{tmp_id}/RUL_FD001.txt'

        ##Put code to load local model data
        X_tr, Y_tr, X_test_1, Y_test_1, cov_adj_mat_1 = get_tr_test_data(tr_dp_1, te_dp_1, gt_dp_1)
        x_tr_list.append(X_tr)
        y_tr_list.append(Y_tr)
        tmp_model = get_transformer_model(num_features = np.asarray(X_tr).astype(np.float32).shape[-1], 
                                            num_attn_heads=3, 
                                            hidden_layer_dim=32, 
                                            num_transformer_blocks=3, 
                                            time_dim=np.asarray(X_tr).astype(np.float32).shape[-2])
        local_model.append(tmp_model)


    tr_dp_1 = './dataset/train_FD001.txt'
    te_dp_1 = './dataset/test_FD001.txt'
    gt_dp_1 = './dataset/RUL_FD001.txt'
    X_tr, Y_tr, X_test_1, Y_test_1, cov_adj_mat_1 = get_tr_test_data(tr_dp_1, te_dp_1, gt_dp_1)
    global_model = get_transformer_model(num_features = np.asarray(X_tr).astype(np.float32).shape[-1], 
                                        num_attn_heads=3, 
                                        hidden_layer_dim=32, 
                                        num_transformer_blocks=3, 
                                        time_dim=np.asarray(X_tr).astype(np.float32).shape[-2])

    ###Communication Rounds Loop
    training_history = {client_id:[] for client_id in range(len(local_model))}
    for tmp_round in range(num_comm_rounds):
        # Train One Communication Round
        weight_dict = {client_id:{} for client_id in range(len(local_model))}
        for i in range(len(local_model)):
            history = local_model[i].fit(x_tr_list[i], y_tr_list[i], batch_size=B, epochs=E)
            training_history[i].append(history)
            weight_len = len(local_model[i].weights)
            # get the weight and save them in a list
            for k in range(weight_len):
                tmp_weight = local_model[i].weights[k].numpy()
                weight_dict[i][k] = tmp_weight
            
        # Offline transfer
        for tmp_key in weight_dict.keys():
            selected_flat_indices = np.random.choice(weight_size, 317, replace=False)
            current_position = 0
            for array_id, array in weight_dict[tmp_key].items():
                array_size = array.size
                # Check if the random indices fall within this array's range
                relevant_indices = (selected_flat_indices >= current_position) & (selected_flat_indices < current_position + array_size)
                # Get the corresponding indices relative to this array
                relative_indices = selected_flat_indices[relevant_indices] - current_position
                
                if relative_indices.size > 0:
                    if array.ndim == 1:
                        weight_dict[tmp_key][array_id][relative_indices] = 0
                    else:
                        # Convert flat indices to 2D indices for this array
                        row_indices, col_indices = np.unravel_index(relative_indices, array.shape)
                        # Assign 0 to the selected elements
                        weight_dict[tmp_key][array_id][row_indices, col_indices] = 0
                # Update current_position to move to the next array's range
                current_position += array_size

        '''
        FedAvg algorithm
        '''
        # Sort the dictionary and add to the list
        weight_list = [list(dict(sorted(tmp_client_weight.items())).values()) for  tmp_client_weight in weight_dict.values()]
        print(f"Aggregate weight for round {tmp_round}")
        aggregate_weight = aggregate_weights(weight_list)
        for weight_id in range(len(aggregate_weight)):
            global_model.weights[weight_id].assign(aggregate_weight[weight_id])
        

        # Offline transfer back to client
        for i in range(len(local_model)):
            tmp_weight_list = deepcopy(aggregate_weight)
            selected_flat_indices = np.random.choice(weight_size, 317, replace=False)
            current_position = 0
            for array_id, array in enumerate(tmp_weight_list):
                array_size = array.size
                # Check if the random indices fall within this array's range
                relevant_indices = (selected_flat_indices >= current_position) & (selected_flat_indices < current_position + array_size)
                # Get the corresponding indices relative to this array
                relative_indices = selected_flat_indices[relevant_indices] - current_position

                if relative_indices.size > 0:
                    if array.ndim == 1:
                        tmp_weight_list[array_id][relative_indices] = 0
                    else:
                        # Convert flat indices to 2D indices for this array
                        row_indices, col_indices = np.unravel_index(relative_indices, array.shape)
                        # Assign 0 to the selected elements
                        tmp_weight_list[array_id][row_indices, col_indices] = 0
                # Update current_position to move to the next array's range
                current_position += array_size
            
            # Assign weight to local model
            for weight_id in range(len(tmp_weight_list)):
                local_model[i].weights[weight_id].assign(tmp_weight_list[weight_id])
        
    with open(f"result/fl_loss_{options.lose}_E_{options.epoch}_B_{options.batch}.pkl", "wb") as f:
        pickle.dump(training_history, f)