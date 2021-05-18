import tensorflow as tf

tf.flags.DEFINE_string("f", "", "")
# define the paths of the datasets
# rppa 
tf.flags.DEFINE_string("rppa", "datasets/rppa", "")
# breast cancer GE 
tf.flags.DEFINE_string("GE", "datasets/GE", "")

# general parameters 
tf.flags.DEFINE_float('learning_rate', 0.005, 'Learning rate')
tf.flags.DEFINE_float('dropout_prob', 0.3, 'Dropout rate')
tf.flags.DEFINE_integer('gcn_layers', 1, 'num of gcn layers')
tf.flags.DEFINE_float('train_ratio', 0.5, 'time point to predict')
tf.flags.DEFINE_float('sparse_rate', 0.0, 'time point to predict')
tf.flags.DEFINE_integer('num_run', 20, 'num of runs')
# tf.flags.DEFINE_integer('time_steps', 8, 'time point to predict')
# tf.flags.DEFINE_integer('hidden_dim', 1, 'hidden embed size of gcn')
# tf.flags.DEFINE_integer('hidden_size', 1, 'LSTM hidden size')
tf.flags.DEFINE_integer('time_steps', 5, 'time point to predict')
tf.flags.DEFINE_integer('hidden_dim', 6, 'hidden embed size of gcn')
tf.flags.DEFINE_integer('hidden_size', 6, 'LSTM hidden size')
tf.flags.DEFINE_integer('window_size', 4, 'LSTM prediction window size')
tf.flags.DEFINE_integer('batch_size', 400, 'LSTM training batch size')
tf.flags.DEFINE_integer('epochs', 500, 'Number of epochs to train')



## imputation task

# import tensorflow as tf
# 
# # define the paths of the datasets
# # rppa 
# tf.flags.DEFINE_string("rppa", "datasets/rppa", "")
# # breast cancer GE 
# tf.flags.DEFINE_string("GE", "datasets/GE", "")
# 
# # general parameters 
# tf.flags.DEFINE_float('learning_rate', 0.005, 'Learning rate')
# tf.flags.DEFINE_float('dropout_prob', 0.3, 'Dropout rate')
# tf.flags.DEFINE_integer('gcn_layers', 1, 'num of gcn layers')
# tf.flags.DEFINE_float('train_ratio', 0.1, 'time point to predict')
# tf.flags.DEFINE_integer('time_steps', 8, 'time point to predict')
# tf.flags.DEFINE_integer('hidden_dim', 1, 'hidden embed size of gcn')
# tf.flags.DEFINE_integer('hidden_size', 1, 'LSTM hidden size')
# # tf.flags.DEFINE_integer('time_steps', 5, 'time point to predict')
# # tf.flags.DEFINE_integer('hidden_dim', 6, 'hidden embed size of gcn')
# # tf.flags.DEFINE_integer('hidden_size', 6, 'LSTM hidden size')
# tf.flags.DEFINE_integer('window_size', 2, 'LSTM prediction window size')
# tf.flags.DEFINE_integer('batch_size', 200, 'LSTM training batch size')
# tf.flags.DEFINE_integer('epochs', 500, 'Number of epochs to train')