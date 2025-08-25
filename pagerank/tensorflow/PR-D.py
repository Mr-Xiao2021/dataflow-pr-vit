tf_config = {
    "cluster": {
        #"ps": ["11.11.11.10:2222"],  # 参数服务器的列表
        "worker": ["11.11.11.15:4446","11.11.11.10:4446"],
                   #,"11.11.11.12:2222","11.11.11.13:2222"],# 工作节点的列表
    },
    "task": {
        "type": "worker",  # 任务类型（参数服务器）
        "index": 0     # 在参数服务器列表中的索引
    }
}

import tensorflow as tf
import numpy as np
import sys
import time
import os
import json
os.environ['TF_CONFIG'] = json.dumps(tf_config)
strategy = tf.distribute.MultiWorkerMirroredStrategy()

def print_output(final_rank):
    rank_idx = np.argsort(final_rank[:, 0], axis=0)
    rank_idx_asc = rank_idx[::-1]
    print("1. Printing top 20 node ids with their ranks")
    print("S No. \t Node Id \t Rank")
    for i in range(20):
        print(i + 1, "\t", rank_idx_asc[i], "\t", final_rank[rank_idx_asc[i]][0])

@tf.function
def compute_pagerank(v, M, e, beta, min_err):
    iterations = tf.constant(0, dtype=tf.int32)
    total_traversed_edges = tf.constant(0, dtype=tf.int64)
    
    num_edges = tf.cast(tf.size(M.values), tf.int64)  # Number of edges in the graph

    for _ in tf.range(1000):
        pagerank = tf.sparse.sparse_dense_matmul(M, v) * beta + e
        diff_in_rank = tf.reduce_sum(tf.abs(pagerank - v))
        v.assign(pagerank)
        iterations += 1
        total_traversed_edges += num_edges
        
        if diff_in_rank < min_err:
            break
    
    return v, total_traversed_edges

def pagerank(adj_matrix, num_nodes, teleport_prob=0.15, min_err=1.0e-3):
    beta = 1 - teleport_prob
    e = np.ones((num_nodes, 1), dtype=np.float32) * (teleport_prob / num_nodes)
    
    v = tf.Variable(tf.ones([num_nodes, 1], dtype=tf.float32) / num_nodes, trainable=False)
    M = adj_matrix / tf.sparse.reduce_sum(adj_matrix, axis=0)

    with strategy.scope():
        result, traversed_edges = strategy.run(compute_pagerank, args=(v, M, e, beta, min_err))
        result = tf.concat(strategy.experimental_local_results(result), axis=0)  # Concatenate results from multiple replicas
        traversed_edges = strategy.reduce(tf.distribute.ReduceOp.SUM, traversed_edges, axis=None)  # Sum traversed edges

    return result, traversed_edges

def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            data.append((node1, node2))

    max_nodeid = max(max(data, key=lambda x: x[0])[0], max(data, key=lambda x: x[1])[1]) + 1
    adj_matrix = tf.sparse.SparseTensor(indices=[[node2, node1] for node1, node2 in data], values=[1.0] * len(data), dense_shape=[max_nodeid, max_nodeid])

    return adj_matrix, max_nodeid

def main(filepath):
    # 获取所有可见的物理 GPU
    physical_gpus = strategy.num_replicas_in_sync
    print(f"物理 GPU 数量: {physical_gpus}")

    start_time = time.time()
    adj_matrix, num_nodes = load_data(filepath)

    calculation_time = time.time()
    pagerank_vector, traversed_edges = pagerank(adj_matrix, num_nodes)
    end_time = time.time()

    execution_time = end_time - calculation_time
    all_time = end_time - start_time
    GTEPS = tf.cast(traversed_edges, tf.float32) * 2 / execution_time / 1e9  # Cast traversed_edges to float32
    print_output(pagerank_vector.numpy())
    print(traversed_edges)
    print("Pagerank calculation took", execution_time, "seconds")
    print("Pagerank all took", all_time, "seconds")
    print("GTEPS:", GTEPS.numpy())  # Convert GTEPS to numpy before printing

# python PR-D.py graph500-scale18-ef16_adj.edges
# python PR-D.py graph500-scale20-ef16_adj.edges
# python PR-D.py graph500-scale22-ef16_adj.edges
if __name__ == '__main__':
    root_path = '/mnt/7T/tianyu/pagerank/'
    filepath = root_path + sys.argv[1]

    main(filepath)