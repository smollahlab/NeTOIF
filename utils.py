import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import random
import os
import csv
import pandas as pd


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# def load_data_predict(dataset, time_steps, missing_ratios=None):
#      
#     if not missing_ratios:
#         assert len(missing_ratios) == time_steps
#      
#     eval_path = os.path.join(dataset,'predict_eval_{}.npz'.format(str(time_steps)))
#     try:
#         print('loading data......')
#         adj_list, feat_list, val_idx, test_idx = np.load(eval_path, encoding='bytes', 
#                                                allow_pickle=True)['data']
#     except IOError:
#         feat_file = os.path.join(dataset, 'MDD_RPPA_Level3_preprocessed_2020-9.xlsx')
#         feat_df = pd.read_excel(feat_file, sheet_name='MDD_RPPA_Level3_annotated').set_index('Protein')
#         feat_df.columns = [c.split('_')[0]+'_'+c.split('_')[1] for c in feat_df.columns]
#         feat_df = feat_df.loc[:,~feat_df.columns.str.startswith('Ctrl')]
#         feat_df = feat_df.apply(pd.to_numeric, errors='ignore')
#         feat_df = feat_df.groupby(feat_df.columns, axis=1, sort=False).mean()
#          
#         feats = feat_df.values
#         feats = feats.reshape([feats.shape[0],6,5]).transpose([2, 0, 1])
#         feat_list = []
#         for i in range(time_steps):
#             if not missing_ratios:
#                 count = []
#                 num_removed = int(feats[0].shape[0] * feats[0].shape[1] * missing_ratios[i])
#                 while len(count) < num_removed:
#                     x_id = random.randint(0, feats[0].shape[0]-1)
#                     y_id = random.randint(0, feats[0].shape[1]-1)
#                     temp_id = str(x_id) + "_" + str(y_id)
#                     if temp_id not in count:
#                         feats[i][x_id][y_id] = 0
#                         count.append(temp_id)
#                 feat_list.append(feats[i])
#             else:    
#                 feat_list.append(feats[i])
#          
#         protein_names = feat_df.index.tolist()
#          
#         network_file = os.path.join(dataset, 'string_interactions.tsv')
#         net_df = pd.read_csv(network_file,sep='\t')
#          
#         adj = np.zeros([len(protein_names),len(protein_names)])
#         links = []
#         no_links = []
#         for index, row in net_df.iterrows():
#             try:
#                 v1_idx = protein_names.index(row['node1'])
#                 v2_idx = protein_names.index(row['node2'])
#                 score = float(row['combined_score'])
#                  
#                 adj[v1_idx,v2_idx] = score
#                 adj[v2_idx,v1_idx] = score
#                  
#                 links.append((row['node1'], row['node2']))
#             except:
#                 no_links.append((row['node1'], row['node2']))
#          
#         adj_list = []
#         for i in range(time_steps):
#             adj_list.append(sp.coo_matrix(adj))
#          
#         temp_indexes = range(len(protein_names))
#         val_idx = np.array(random.sample(temp_indexes, int(len(protein_names)*0.2)))
#         test_idx = np.setdiff1d(np.array(temp_indexes), val_idx)
#          
#         np.savez(eval_path, data=[adj_list, feat_list, val_idx, test_idx])
#      
#     return adj_list, feat_list, val_idx, test_idx
 
# load_data('datasets/rppa', 2)

def load_data_prediction_GE(dataset, time_steps, val_ratio=0.2, sparse_flag = False, sparse_rate = 0.1):
    
    eval_path = os.path.join(dataset,'prediction_eval_{}_{}.npz'.format(str(time_steps),str(val_ratio)))
    if sparse_flag:
        eval_path = os.path.join(dataset,'prediction_eval_{}_{}_{}.npz'.format(str(time_steps),str(val_ratio),str(sparse_rate)))
    
    try:
        print('loading data.....')
        adj_list, feat_list, protein_names, val_idx, test_idx = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
    except IOError:
        feat_file = os.path.join(dataset, '12859_2008_2579_MOESM2_ESM_processed.xlsx')
        feat_df = pd.read_excel(feat_file, sheet_name='ZR75.1_QUA_P4_cluster_named').set_index('Search_key')
        
        print(feat_df.columns)
        feats = feat_df.values
        feats = feats.reshape([feats.shape[0],1,8]).transpose([2, 0, 1])
        
        protein_names = feat_df.index.tolist()
            
        network_file = os.path.join(dataset, 'string_interactions.tsv')
        net_df = pd.read_csv(network_file,sep='\t')
        
        adj = np.zeros([len(protein_names),len(protein_names)])
        links = []
        no_links = []
        for index, row in net_df.iterrows():
            try:
                v1_idx = protein_names.index(row['node1'])
                v2_idx = protein_names.index(row['node2'])
                score = float(row['combined_score'])
                
                adj[v1_idx,v2_idx] = score
                adj[v2_idx,v1_idx] = score
                
                links.append((row['node1'], row['node2']))
            except:
                no_links.append((row['node1'], row['node2']))
        
        adj_list = []
        for i in range(time_steps):
            adj_list.append(sp.coo_matrix(adj))
            
        
#         feat_list = []
#         for i in range(time_steps):
#             feat_list.append(feats[i])
            
        edges = []
        for i in range(len(protein_names)):
            for j in range(1):
                edges.append((i,j))   
            
        feat_list = []
        for i in range(time_steps):
            temp_feat = feats[i]
            if sparse_flag and i<(time_steps-1):
                # randomly remove features (sparse_rate) and replace with zero
                remove_idx = np.array(random.sample(range(len(edges)), int(len(edges)*sparse_rate)))
                for i in remove_idx:
                    x, y = edges[i]
                    temp_feat[x][y] = 0
            feat_list.append(temp_feat) 
        
        # generate the train/val/test masks for the last time step
        temp_indexes = range(len(protein_names))
        val_idx = np.array(random.sample(temp_indexes, int(len(protein_names)*val_ratio)))
        test_idx = np.setdiff1d(np.array(temp_indexes), val_idx)
    
        np.savez(eval_path, data=[adj_list, feat_list, protein_names, val_idx, test_idx])
        
    return adj_list, feat_list, val_idx, test_idx, protein_names

def load_data_impute_GE(dataset, time_steps, train_ratio=0.7, num_run=20, sparse_flag = False, sparse_rate = 0.1):
    
    eval_path = os.path.join(dataset,'impute_eval_{}_{}_{}.npz'.format(str(time_steps),str(num_run),str(train_ratio)))
    if sparse_flag:
        eval_path = os.path.join(dataset,'impute_eval_{}_{}_{}_{}.npz'.format(str(time_steps),str(num_run),
                                                                              str(train_ratio), str(sparse_rate)))
    try:
        print('loading data.....')
        adj_list, feat_list, protein_names, train_masks, val_masks, test_masks = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
    except IOError:
        feat_file = os.path.join(dataset, '12859_2008_2579_MOESM2_ESM_processed.xlsx')
        feat_df = pd.read_excel(feat_file, sheet_name='ZR75.1_QUA_P4_cluster_named').set_index('Search_key')
        
        print(feat_df.columns)
        feats = feat_df.values
        feats = feats.reshape([feats.shape[0],1,8]).transpose([2, 0, 1])
        
        protein_names = feat_df.index.tolist()
            
        network_file = os.path.join(dataset, 'string_interactions.tsv')
        net_df = pd.read_csv(network_file,sep='\t')
        
        adj = np.zeros([len(protein_names),len(protein_names)])
        links = []
        no_links = []
        for index, row in net_df.iterrows():
            try:
                v1_idx = protein_names.index(row['node1'])
                v2_idx = protein_names.index(row['node2'])
                score = float(row['combined_score'])
                
                adj[v1_idx,v2_idx] = score
                adj[v2_idx,v1_idx] = score
                
                links.append((row['node1'], row['node2']))
            except:
                no_links.append((row['node1'], row['node2']))
        
        adj_list = []
        for i in range(time_steps):
            adj_list.append(sp.coo_matrix(adj))
        
    #     print('num_links:', len(links), 'num_no_links:', len(no_links), adj)
        
        # generate the train/val/test masks for the last time step
        edges = []
        for i in range(len(protein_names)):
            for j in range(1):
                edges.append((i,j))
                
        feat_list = []
        for i in range(time_steps):
            temp_feat = feats[i]
            if sparse_flag and i<(time_steps-1):
                # randomly remove features (sparse_rate) and replace with zero
                remove_idx = np.array(random.sample(range(len(edges)), int(len(edges)*sparse_rate)))
                for i in remove_idx:
                    x, y = edges[i]
                    temp_feat[x][y] = 0
            feat_list.append(temp_feat)        
        
        train_masks = []
        val_masks = []      
        test_masks = []
        for k in range(num_run):
            train_idx = np.array(random.sample(range(len(edges)), int(len(edges)*train_ratio)))
            temp_idx = np.setdiff1d(np.array(range(len(edges))), train_idx)
            val_idx = np.array(random.sample(list(temp_idx), int(len(temp_idx)*0.2)))
            test_idx = np.setdiff1d(temp_idx, val_idx)
            train_mask = np.zeros([len(protein_names), 1])
            for i in train_idx:
                x, y = edges[i]
                train_mask[x][y] = 1
            val_mask = np.zeros([len(protein_names), 1])
            for i in val_idx:
                x, y = edges[i]
                val_mask[x][y] = 1
            test_mask = np.zeros([len(protein_names), 1])
            for i in test_idx:
                x, y = edges[i]
                test_mask[x][y] = 1
            
            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)
        
        np.savez(eval_path, data=[adj_list, feat_list, protein_names, train_masks, val_masks, test_masks])
        
    return adj_list, feat_list, train_masks, val_masks, test_masks, protein_names
    
       
# load_data_impute_GE('datasets/GE', 4)       

def load_data_impute(dataset, time_steps, train_ratio=0.7, num_run=20, sparse_flag = False, sparse_rate = 0.1):
    
    eval_path = os.path.join(dataset,'impute_eval_{}_{}_{}.npz'.format(str(time_steps), str(num_run),str(train_ratio)))
    if sparse_flag:
        eval_path = os.path.join(dataset,'impute_eval_{}_{}_{}_{}.npz'.format(str(time_steps),str(num_run),
                                                                              str(train_ratio), str(sparse_rate)))
    try:
        print('loading data......')
        adj_list, feat_list, protein_names, ligand_names, train_masks, val_masks, test_masks = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
#         adj_list, feat_list, train_idx, val_idx, test_idx = \
#             np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
    except IOError:
        feat_file = os.path.join(dataset, 'MDD_RPPA_Level3_preprocessed_2020-9.xlsx')
        feat_df = pd.read_excel(feat_file, sheet_name='MDD_RPPA_Level3_annotated').set_index('Protein')
        feat_df.columns = [c.split('_')[0]+'_'+c.split('_')[1] for c in feat_df.columns]
        feat_df = feat_df.loc[:,~feat_df.columns.str.startswith('Ctrl')]
        feat_df = feat_df.apply(pd.to_numeric, errors='ignore')
        feat_df = feat_df.groupby(feat_df.columns, axis=1, sort=False).mean()
        
        feats = feat_df.values
        feats = feats.reshape([feats.shape[0],6,5]).transpose([2, 0, 1])
#         feat_list = []
#         for i in range(time_steps):
#             feat_list.append(feats[i])
    #         feat_list.append(sp.coo_matrix(feats[i]))
        
    #     print(feats.shape)
        
        protein_names = feat_df.index.tolist()
        ligand_names = pd.Index([v.split('_')[0] for v in feat_df.columns.to_numpy().reshape([6,5]).transpose([1,0])[0]]).tolist()
        
        network_file = os.path.join(dataset, 'string_interactions_new.tsv')
        net_df = pd.read_csv(network_file,sep='\t')
        
        adj = np.zeros([len(protein_names),len(protein_names)])
        links = []
        no_links = []
        for index, row in net_df.iterrows():
            try:
                v1_idx = protein_names.index(row['node1'])
                v2_idx = protein_names.index(row['node2'])
                score = float(row['combined_score'])
                
                adj[v1_idx,v2_idx] = score
                adj[v2_idx,v1_idx] = score
                
                links.append((row['node1'], row['node2']))
            except:
                no_links.append((row['node1'], row['node2']))
        
        adj_list = []
        for i in range(time_steps):
            adj_list.append(sp.coo_matrix(adj))
        
    #     print('num_links:', len(links), 'num_no_links:', len(no_links), adj)
        
        # generate the train/val/test masks for the last time step
        edges = []
        for i in range(len(protein_names)):
            for j in range(6):
                edges.append((i,j))
                
        feat_list = []
        for i in range(time_steps):
            temp_feat = feats[i]
            if sparse_flag and i<(time_steps-1):
                # randomly remove features (sparse_rate) and replace with zero
                remove_idx = np.array(random.sample(range(len(edges)), int(len(edges)*sparse_rate)))
                for i in remove_idx:
                    x, y = edges[i]
                    temp_feat[x][y] = 0
            feat_list.append(temp_feat)                
        
        
        train_masks = []
        val_masks = []      
        test_masks = []  
        for k in range(num_run):
            train_idx = np.array(random.sample(range(len(edges)), int(len(edges)*train_ratio)))
            temp_idx = np.setdiff1d(np.array(range(len(edges))), train_idx)
            val_idx = np.array(random.sample(list(temp_idx), int(len(temp_idx)*0.2)))
            test_idx = np.setdiff1d(temp_idx, val_idx)
            train_mask = np.zeros([len(protein_names), 6])
            for i in train_idx:
                x, y = edges[i]
                train_mask[x][y] = 1
            val_mask = np.zeros([len(protein_names), 6])
            for i in val_idx:
                x, y = edges[i]
                val_mask[x][y] = 1
            test_mask = np.zeros([len(protein_names), 6])
            for i in test_idx:
                x, y = edges[i]
                test_mask[x][y] = 1
            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)
            
        np.savez(eval_path, data=[adj_list, feat_list, protein_names, ligand_names, 
                                  train_masks, val_masks, test_masks])
#         # split the train/val/test sets
#         train_idx = np.array(random.sample(range(len(protein_names)), int(len(protein_names)*train_ratio))) 
#         temp_indexes = np.setdiff1d(np.array(range(len(protein_names))), train_idx)
#         val_idx = np.array(random.sample(list(temp_indexes), int(len(temp_indexes)*0.2)))
#         test_idx = np.setdiff1d(temp_indexes, val_idx)
#         
#         np.savez(eval_path, data=[adj_list, feat_list, train_idx, val_idx, test_idx])
    return adj_list, feat_list, train_masks, val_masks, test_masks, protein_names, ligand_names
#     return adj_list, feat_list, train_idx, val_idx, test_idx

# load_data_impute('datasets/rppa', 2)

def build_train_samples_imputation(embeds_list, feats, time_steps):
    points = []
    for i in range(time_steps-1): 
        points.append(embeds_list[i])
    point_seq = tf.concat([tf.expand_dims(p, 1) for p in points], 1)
    
    return point_seq, feats[-1]
    
def build_train_samples_prediction(embeds_list, feats, time_steps, 
                                   window_size, val_idx, test_idx,
                                   base_mat=None):
    if base_mat == None:
        base_mat = tf.zeros_like(embeds_list[0]) # compare with no padding
    
    ps_x_trains = []
    ps_y_trains = []
    ps_x_vals = []
    ps_y_vals = []
    ps_x_tests = []
    ps_y_tests = []
#     for i in range(1, time_steps):
    for i in range(window_size, time_steps):
        points = []
        for j in range(window_size):
            s = i - window_size + j
            if s < 0:
                points.append(base_mat)
            else:
                points.append(embeds_list[j])
                
        point_seq = tf.concat([tf.expand_dims(p, 1) for p in points], 1)
        
        if i < (time_steps-1):
            ps_x_trains.append(point_seq)
            ps_y_trains.append(feats[i])
        
        else:
            ps_x_vals.append(tf.gather(point_seq, val_idx))
            ps_y_vals.append(tf.gather(feats[i], val_idx))
            ps_x_tests.append(tf.gather(point_seq, test_idx))
            ps_y_tests.append(tf.gather(feats[i], test_idx))
        
    ps_x_train = tf.concat(ps_x_trains, 0)   
    ps_y_train = tf.concat(ps_y_trains, 0)   
    ps_x_val = tf.concat(ps_x_vals, 1)   
    ps_y_val = tf.concat(ps_y_vals, 1)   
    ps_x_test = tf.concat(ps_x_tests, 1)   
    ps_y_test = tf.concat(ps_y_tests, 1)   
#         ps_x_trains.append(tf.gather(point_seq, train_idx[i]))
#         ps_y_trains.append(tf.gather(feats[i], train_idx[i]))
#         
#         if len(test_idx[i]) != 0 & len(val_idx[i]) != 0:
#             ps_x_vals.append(tf.gather(point_seq, val_idx[i]))
#             ps_y_vals.append(tf.gather(feats, val_idx[i]))
#             ps_x_tests.append(tf.gather(point_seq, test_idx[i]))
#             ps_y_tests.append(tf.gather(feats, test_idx[i]))
#         
#     ps_x_train = tf.concat(ps_x_trains, 1)   
#     ps_y_train = tf.concat(ps_y_trains, 1)   
#     ps_x_val = tf.concat(ps_x_vals, 1)   
#     ps_y_val = tf.concat(ps_y_vals, 1)   
#     ps_x_test = tf.concat(ps_x_tests, 1)   
#     ps_y_test = tf.concat(ps_y_tests, 1)   
        
    return ps_x_train, ps_y_train, ps_x_val, ps_y_val, ps_x_test, ps_y_test

def process_GEdata(dataset):
    
    network_file = os.path.join(dataset, 'string_interactions.tsv')
    net_df = pd.read_csv(network_file,sep='\t')
    unique_proteins = []
    for index, row in net_df.iterrows():
        p1 = row['node1']
        p2 = row['node2']
        if p1 not in unique_proteins:
            unique_proteins.append(p1)
        if p2 not in unique_proteins:
            unique_proteins.append(p2)
            
    print(len(unique_proteins))
    
    feat_file = os.path.join(dataset, '12859_2008_2579_MOESM2_ESM.xlsx')
    feat_df = pd.read_excel(feat_file, sheet_name='ZR75.1_QUA_P4_cluster_named').set_index('Search_key')
    
#     for s in unique_proteins:
#         if s not in feat_df.index.tolist(): 
#             print(s)
    
    feat_df = feat_df[feat_df.index.isin(unique_proteins)]
    
    processed_file = os.path.join(dataset, '12859_2008_2579_MOESM2_ESM_processed.xlsx')
    writer = pd.ExcelWriter(processed_file)
    feat_df.to_excel(writer, 'ZR75.1_QUA_P4_cluster_named')
    writer.save()
    print(feat_df.index)
    
# process_GEdata('datasets/GE')    