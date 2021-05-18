"""Spectral Graph Convolutional filter cell."""
import numpy as np
import tensorflow as tf

def _dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class GraphConvLayer:
    def __init__(self, time_steps, gcn_layers, input_dim, hidden_dim, output_dim, name, 
                 num_features_nonzeros, dropout_prob, act=tf.nn.relu, dropout=False):
#                  name, act=tf.nn.relu, bias=False, dropout=):
        self.time_steps = time_steps
        self.gcn_layers = gcn_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act=act
        self.dropout = dropout
        self.num_features_nonzeros = num_features_nonzeros
        
        self.w_list=[]
        with tf.variable_scope(name):
            
            for i in range(self.gcn_layers):
                if i==0:
                    w_i = glorot([input_dim, hidden_dim], name='w'+str(i))
                else:
                    w_i = glorot([hidden_dim, output_dim], name='w'+str(i))
                self.w_list.append(w_i)
                
            if self.dropout:
                self.dropout_prob = dropout_prob
            else:
                self.dropout_prob = 0.
                    
    def call(self, adjs, feats, sparse=False):  
        
        embeds_list=[]
        for i in range(self.time_steps):
            adj_norm = adjs[i]
            x = feats[i]
        
            if sparse:
                x = sparse_dropout(x, 1-self.dropout_prob, self.num_features_nonzeros[i])
            else:
                x = tf.nn.dropout(x, 1-self.dropout_prob)
            hw = _dot(x=x, y=self.w_list[0], sparse=sparse)  
            ahw = _dot(x=adj_norm, y=hw, sparse=True)
            last_l = self.act(ahw)
                
            for j in range(1, self.gcn_layers):
                hw = _dot(x=last_l, y=self.w_list[j]) 
                ahw = _dot(x=adj_norm, y=hw, sparse=True)
                last_l = self.act(ahw)
            embeds_list.append(last_l)      
              
        return embeds_list 
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)        