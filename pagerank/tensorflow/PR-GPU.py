import tensorflow as tf
import numpy as np
import sys
import time


def print_output(final_rank):
    rank_idx = np.argsort(final_rank[:, 0])[::-1]
    print("\nPrinting top 20 node ids with their ranks")
    print(f"Rank | Node ID | Score")
    print("-" * 30)
    for i in range(20):
        print(f"{i+1:4d} | {rank_idx[i]:7d} | {final_rank[rank_idx[i]][0]:.6f}")


@tf.function  # 编译成图，加速执行
def pagerank_step(M, v, e, beta):
    return beta * tf.sparse.sparse_dense_matmul(M, v) + e


def pagerank(adj_matrix, num_nodes, teleport_prob=0.15, min_err=1.0e-3):
    beta = 1 - teleport_prob
    e = tf.ones((num_nodes, 1), dtype=tf.float32) * teleport_prob / num_nodes

    # 初始 rank 向量
    R = tf.ones((num_nodes, 1), dtype=tf.float32) / num_nodes
    traversed_edges_counter = 0

    for _ in range(1000):  # 最大迭代次数，避免死循环
        new_R = pagerank_step(adj_matrix, R, e, beta)
        err = tf.reduce_sum(tf.abs(new_R - R))
        traversed_edges_counter += adj_matrix.indices.shape[0]
        R = new_R
        if err < min_err:
            break
    return R.numpy(), traversed_edges_counter


def load_data(filepath):
    data = []
    edge_count = 0
    with open(filepath, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            data.append((node1, node2))
            edge_count += 1

    max_nodeid = max(max(data, key=lambda x: x[0])[0],
                     max(data, key=lambda x: x[1])[1]) + 1

    indices = [[node2, node1] for node1, node2 in data]
    values = [1.0] * len(data)
    adj_matrix = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[max_nodeid, max_nodeid]
    )

    # 列归一化（重要，否则 PageRank 发散）
    col_sum = tf.sparse.reduce_sum(adj_matrix, axis=0)
    adj_matrix = tf.sparse.reorder(
        tf.sparse.SparseTensor(
            indices=indices,
            values=values / tf.gather(col_sum, [i[1] for i in indices]),
            dense_shape=[max_nodeid, max_nodeid]
        )
    )
    return adj_matrix, max_nodeid, edge_count


if __name__ == '__main__':
    root_path = '/gemini/code/graph/'
    filepath = root_path + sys.argv[1]

    start_time = time.time()
    adj_matrix, num_nodes, num_edges = load_data(filepath)
    calculation_time = time.time()
    pagerank_vector, traversed_edges = pagerank(adj_matrix, num_nodes)
    end_time = time.time()

    execution_time = end_time - calculation_time
    all_time = end_time - start_time

    GTEPS = traversed_edges / execution_time / 1e9

    print("================================ TensorFlow METHOD =======================================")
    print(f"\t\tGraph: nodes: {num_nodes - 1}, edges: {num_edges}")
    print("Pagerank calculation took", execution_time, "seconds for", traversed_edges, "edges")
    print(f"\t\tGTEPS: {GTEPS:.6f}")
    print(f"\t\tTotal Cost: {all_time:.3f} seconds")
    print("================================ TensorFlow METHOD =======================================")
    # print_output(pagerank_vector)
