import tensorflow as tf
import numpy as np
import sys
import time


tf.compat.v1.disable_eager_execution()

def print_output(final_rank):
    rank_idx = np.argsort(final_rank[:, 0])[::-1]
    print("\nPrinting top 20 node ids with their ranks")
    print(f"Rank | Node ID | Score")
    print("-" * 30)
    for i in range(20):
        print(f"{i+1:4d} | {rank_idx[i]:7d} | {final_rank[rank_idx[i]][0]:.6f}")
    
def pagerank(adj_matrix, num_nodes, teleport_prob=0.15, min_err=1.0e-3):
    beta = 1 - teleport_prob
    e = np.ones((num_nodes, 1)) * teleport_prob / num_nodes

    v = tf.compat.v1.placeholder(tf.float32, shape=[num_nodes, 1])
    M = adj_matrix / tf.sparse.reduce_sum(adj_matrix, axis=0)

    pagerank = tf.add(tf.sparse.sparse_dense_matmul(M, v) * beta, e)
    diff_in_rank = tf.reduce_sum(tf.abs(pagerank - v))

    init = tf.compat.v1.global_variables_initializer()
    traversed_edges_counter = 0
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        R = np.ones((num_nodes, 1)) / num_nodes
        while True:
            new_pagerank = sess.run(pagerank, feed_dict={v: R})
            err_norm = sess.run(diff_in_rank, feed_dict={pagerank: new_pagerank, v: R})
            R = new_pagerank
            traversed_edges_counter += adj_matrix.indices.shape[0]
            if err_norm < min_err:
                break
    return R,traversed_edges_counter

def load_data(filepath):
    data = []
    edge_count = 0
    with open(filepath, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            data.append((node1, node2))
            edge_count += 1 
    
    max_nodeid = max(max(data, key=lambda x: x[0])[0], max(data, key=lambda x: x[1])[1]) + 1
    adj_matrix = tf.sparse.SparseTensor(indices=[[node2, node1] for node1, node2 in data], values=[1.0]*len(data), dense_shape=[max_nodeid, max_nodeid])
    return adj_matrix, max_nodeid, edge_count

# export ORION_CUDART_VERSION=11.8
# python PR.py Rmat-18.edges
# python PR.py Rmat-19.edges
# python PR.py Rmat-20.edges
if __name__ == '__main__':
    root_path = '/gemini/code/graph/'
    filepath = root_path + sys.argv[1]
    
    start_time = time.time()    
    adj_matrix, num_nodes, num_edges = load_data(filepath)
    # print(f"shape of matrix:{adj_matrix.shape}, num_nodes={num_nodes}")
    calculation_time = time.time()
    pagerank_vector,traversed_edges = pagerank(adj_matrix, num_nodes)
    end_time = time.time()

    execution_time = end_time - calculation_time
    all_time = end_time - start_time

    GTEPS = traversed_edges / execution_time / 1e9


    print("================================ TensorFlow METHOD =======================================")
    print(f"\t\tGraph: nodes: {num_nodes - 1}, edges: {num_edges}")
    print("Pagerank calculation took", execution_time, "seconds for", traversed_edges,"edges")
    print(f"\t\tGTEPS: {GTEPS:.6f}")
    print(f"\t\tTotal Cost: {all_time:.3f} seconds")
    print("================================ TensorFlow METHOD =======================================")
    print_output(pagerank_vector)