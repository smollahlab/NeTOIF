import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR) 


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class BiLSTM:
    
    def __init__(self, hidden_size, seq_len, out_size, holders, dropout=False):
        # initialize parameters
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.out_size = out_size
        self.dropout = dropout
        self.dropout_prob = holders['dropout_prob']
        
        initializer = tf.keras.initializers.glorot_normal
        # build the bi-directional lstm 
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer= initializer())
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=initializer())
        
    def call(self, input_seq):
        if self.dropout:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell, 1-self.dropout_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell, 1-self.dropout_prob)
            rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                              cell_bw=bw_cell,
                                                              inputs=input_seq,
                                                              dtype=tf.float32)
        else:   
            rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                              cell_bw=self.bw_cell,
                                                              inputs=input_seq,
                                                              dtype=tf.float32)
        outputs = tf.add(states[0].h, states[1].h)
        w_i = glorot([self.hidden_size, self.out_size])
        outputs = tf.matmul(outputs, w_i)
#         outputs = tf.layers.conv1d(outputs, self.out_size, 1, use_bias=False)

        return outputs
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)    