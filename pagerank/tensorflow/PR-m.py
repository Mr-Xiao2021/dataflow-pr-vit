# python PR-m.py ./graph500-scale18-ef16_adj.edges
import tensorflow as tf
import numpy as np
import sys
import time
import os
import json
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2", "/gpu:3"])

def print_output(final_rank):
    rank_idx = np.argsort(final_rank[:, 0])[::-1]
    print("\nPrinting top 20 node ids with their ranks")
    print(f"Rank | Node ID | Score")
    print("-" * 30)
    for i in range(20):
        print(f"{i+1:4d} | {rank_idx[i]:7d} | {final_rank[rank_idx[i]][0]:.6f}")

@tf.function
def compute_pagerank(v, M, e, beta, min_err):
    iterations = tf.constant(0, dtype=tf.int32)
    total_traversed_edges = tf.constant(0, dtype=tf.int64)
    
    num_edges = tf.cast(tf.size(M.values), tf.int64)  # Number of edges in the graph

    for step in tf.range(1000):
        pagerank = tf.sparse.sparse_dense_matmul(M, v) * beta + e
        diff_in_rank = tf.reduce_sum(tf.abs(pagerank - v))
        v.assign(pagerank)
        iterations += 1
        total_traversed_edges += num_edges
        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            tf.print("Step", step, ", err =", diff_in_rank)

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
    edge_count = 0
    with open(filepath, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            edge_count += 1
            data.append((node1, node2))

    max_nodeid = max(max(data, key=lambda x: x[0])[0], max(data, key=lambda x: x[1])[1]) + 1
    adj_matrix = tf.sparse.SparseTensor(indices=[[node2, node1] for node1, node2 in data], values=[1.0] * len(data), dense_shape=[max_nodeid, max_nodeid])
    print(f'===> nodes = {max_nodeid - 1}, edges = {edge_count}')
    return adj_matrix, max_nodeid, edge_count

def main(filepath):
    # 获取所有可见的物理 GPU
    physical_gpus = strategy.num_replicas_in_sync
    print(f"GPUs: {physical_gpus}")

    start_time = time.time()
    adj_matrix, num_nodes, num_edges = load_data(filepath)

    calculation_time = time.time()
    pagerank_vector, traversed_edges = pagerank(adj_matrix, num_nodes)
    end_time = time.time()

    execution_time = end_time - calculation_time
    all_time = end_time - start_time
    traversed_edges = tf.cast(traversed_edges, tf.float32)
    GTEPS = traversed_edges / execution_time / 1e9  # Cast traversed_edges to float32

    print("================================ TensorFlow METHOD =======================================")
    print(f"\t\tGraph: nodes: {num_nodes - 1}, edges: {num_edges}")
    # print("Pagerank calculation took", execution_time, "seconds for traversed_edges_nums: ",traversed_edges.numpy().item())
    # print(f"\t\tExecution costs: {execution_time * 1000:.3f} ms, GTEPS: {GTEPS.numpy():.6f}")
    print(f"\t\tGTEPS: {GTEPS.numpy():.6f}")
    print(f"\t\tTotal Cost: {all_time:.3f} seconds")
    print("================================ TensorFlow METHOD =======================================")
    print_output(pagerank_vector.numpy())

    
# export ORION_CUDART_VERSION=11.8
# python PR-m.py g18.edges
# python PR-m.py g19.edges
# python PR-m.py g20.edges
if __name__ == '__main__':
    root_path = '/gemini/code/graph/'
    filepath = root_path + sys.argv[1]

    main(filepath)