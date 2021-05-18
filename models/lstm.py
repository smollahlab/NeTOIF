import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) 

class BiLSTM:
    
    def __init__(self, hidden_size, seq_len, holders, dropout=False):
        # initialize parameters
        self.hidden_size = hidden_size
        self.seq_len = seq_len
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

        return outputs
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)    