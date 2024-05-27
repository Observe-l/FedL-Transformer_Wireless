import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




def get_tr_test_data(tr_path, test_path, gt_path):

    def gen_sequence(id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means
        we need to drop those which are below the window-length. """
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]

    # function to generate labels
    def gen_labels(id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]

    w1 = 30
    sequence_length =  50
    # read training data
    train_df = pd.read_csv(tr_path, sep=" ", header=None)
    train_df.drop(train_df.columns[[5, 9, 10, 14, 20, 22, 23, 26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's2', 's3',
                        's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14',
                        's15', 's17', 's20', 's21']

    # read test data
    test_df = pd.read_csv(test_path, sep=" ", header=None)
    test_df.drop(test_df.columns[[5, 9, 10, 14, 20, 22, 23, 26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's2', 's3',
                        's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14',
                        's15', 's17', 's20', 's21']

    # read ground truth data
    truth_df = pd.read_csv(gt_path, sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    # final Training data
    train_df = train_df.sort_values(['id','cycle'])

    # Data Labeling for training data - generate column RUL
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
    train_df.drop('max', axis=1, inplace=True)

    # generate label1 column for training data
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )

    # Data Labeling for test data - generate column RUL

    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)

    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df['RUL'] = test_df['RUL'].clip(upper=125)
    test_df.drop('max', axis=1, inplace=True)

    # generate label1 column for test data
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )


    # MinMax normalization for training data

    scaler = MinMaxScaler()
    train_df['cycle_norm'] = train_df['cycle']

    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    train_df_sc = scaler.fit_transform(train_df[sequence_cols].to_numpy())
    train_df_sc = pd.DataFrame(train_df_sc, columns=sequence_cols)

    train_df = pd.concat([train_df[['id','cycle']], train_df_sc, train_df[['RUL', 'label1']]],axis=1)

    # MinMax normalization for test data

    #scaler = MinMaxScaler()
    test_df['cycle_norm'] = test_df['cycle']

    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    #test_df_sc = scaler.fit_transform(test_df[sequence_cols].to_numpy())
    test_df_sc = scaler.transform(test_df[sequence_cols].to_numpy())

    test_df_sc = pd.DataFrame(test_df_sc, columns=sequence_cols)

    test_df = pd.concat([test_df[['id','cycle']], test_df_sc, test_df[['RUL', 'label1']]],axis=1)

    # pick the feature columns
    sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)

    # generator for the training sequences
    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols))
            for id in train_df['id'].unique())

    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

    deseq_array = np.transpose(np.reshape(seq_array,(-1, 18)))
    cov_adj_mat = np.cov(deseq_array)

    # generate labels (generated from "RUL" col as it's RUL regression)
    label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL'])
                for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)

    X_tr = seq_array
    Y_tr = label_array

    # Test Data
    seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:]
                        for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Last cycle test data - seq_array for test data
    X_test = seq_array_test_last

    y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

    # Last cycle test data - label_array for test data
    Y_test = label_array_test_last

    return X_tr, Y_tr, X_test, Y_test, cov_adj_mat