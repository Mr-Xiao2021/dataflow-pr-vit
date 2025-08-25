import os
import time
import sys
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

# ----------------------------
# 初始化 Horovod / 选择 GPU
# ----------------------------
hvd.init()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

DTYPE = tf.float32

def log0(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs, flush=True)

# ----------------------------
# 读取边并构建本地分块稀疏矩阵
# 邻接按“目标节点(行)”分块：每个进程存 [row_start, row_end) 的行块
# M 是列随机（column-stochastic），values = 1/outdeg[src]
# ----------------------------
def load_partitioned_sparse_block(edge_path):
    # 各进程都读一遍文件，计算：
    # - 全局最大节点 id -> num_nodes
    # - 全局列归一化因子 outdeg[src]
    data = []
    max_node_id = 0
    with open(edge_path, 'r') as f:
        for ln in f:
            a, b = map(int, ln.split())
            # 边：(src=a, dst=b) —— 计算 M*v 时使用 row=dst, col=src
            data.append((a, b))
            if a > max_node_id: max_node_id = a
            if b > max_node_id: max_node_id = b
    num_nodes = max_node_id + 1

    # 统计 out-degree（列和）
    outdeg = np.zeros(num_nodes, dtype=np.int64)
    for src, _ in data:
        outdeg[src] += 1
    outdeg = np.maximum(outdeg, 1)  # 防止除0（dangling）

    # 根据 rank 拆分行块
    world = hvd.size()
    rank  = hvd.rank()
    rows_per_rank = (num_nodes + world - 1) // world
    row_start = rank * rows_per_rank
    row_end   = min(num_nodes, (rank + 1) * rows_per_rank)
    rows_local = max(0, row_end - row_start)

    # 过滤出落在本地行块的边（按目标节点分配）
    # SparseTensor 需要本地行坐标从 0 开始
    local_indices = []
    local_values  = []
    local_edges = 0
    for src, dst in data:
        if row_start <= dst < row_end:
            local_indices.append([dst - row_start, src])            # [row_local, col_global]
            local_values.append(1.0 / float(outdeg[src]))           # 列归一化后的值
            local_edges += 1

    if rows_local == 0:
        # 处理极端小图或 rank 过多的情况
        st = tf.sparse.SparseTensor(indices=[[0, 0]], values=[0.0], dense_shape=[1, num_nodes])
        st = tf.sparse.reorder(st)
        return st, num_nodes, 0, row_start, row_end

    local_indices = np.array(local_indices, dtype=np.int64) if local_indices else np.empty((0,2), np.int64)
    local_values  = np.array(local_values,  dtype=np.float32) if local_values  else np.empty((0,), np.float32)

    local_block = tf.sparse.SparseTensor(
        indices=local_indices,
        values=tf.constant(local_values, dtype=DTYPE),
        dense_shape=[rows_local, num_nodes]
    )
    local_block = tf.sparse.reorder(local_block)
    return local_block, num_nodes, local_edges, row_start, row_end

# ----------------------------
# PageRank 主循环（模型并行）
# - 每步：
#   1) y_local = beta * (M_local @ v_full) + e_local
#   2) v_new = allgather(y_local, axis=0)  → 得到完整向量
#   3) diff_local = sum(|y_local - v_local_prev|), diff = allreduce_sum(diff_local)
# ----------------------------
def pagerank_model_parallel(M_local, num_nodes, row_start, row_end,
                            teleport_prob=0.15, min_err=1e-3, max_steps=1000):
    beta = tf.constant(1.0 - teleport_prob, dtype=DTYPE)

    # 初值 v_full（全局向量，所有进程都保有一份）
    v = tf.fill([num_nodes, 1], tf.constant(1.0/num_nodes, dtype=DTYPE))

    rows_local = int(row_end - row_start)
    if rows_local == 0:
        # 本 rank 没有行，仍需参与通信
        # 建一个形状 [0,1] 的张量（Horovod 允许 allgather 空分片）
        v_local = tf.reshape(v[0:0], [0, 1])

    # 常量 teleport 向量的本地切片：每个条目 = teleport_prob/num_nodes
    e_local = tf.fill([rows_local, 1], tf.constant(teleport_prob/num_nodes, dtype=DTYPE))

    traversed_edges_local_total = 0

    @tf.function
    def step_fn(v_prev):
        # 本地 SpMV：只算自己负责的行块
        y_local = beta * tf.sparse.sparse_dense_matmul(M_local, v_prev) + e_local
        return y_local

    # 记录 v 的上一轮本地切片，便于算 diff
    def slice_local(x):
        return x[row_start:row_end, :] if rows_local > 0 else tf.reshape(x[0:0], [0,1])

    v_local_prev = slice_local(v)

    for it in range(max_steps):
        t0 = tf.timestamp()

        y_local = step_fn(v)  # [rows_local, 1]

        # 拼成完整向量（按行块顺序 allgather）
        v_new = hvd.allgather(y_local)  # [num_nodes, 1]

        # 计算收敛误差（全局 L1 diff）
        diff_local = tf.reduce_sum(tf.abs(y_local - v_local_prev)) if rows_local > 0 else tf.constant(0.0, DTYPE)
        diff = hvd.allreduce(diff_local, op=hvd.Sum)

        # 更新
        v = v_new
        v_local_prev = slice_local(v)

        # 边遍历计数（每轮遍历一次全图的边）
        # 每个 rank 只统计自己的本地边，再做 allreduce 求和（可放回 loop 外累计）
        traversed_edges_local_total += int(tf.size(M_local.values))

        # 仅 rank0 打印
        if hvd.rank() == 0:
            tf.print("Iter", it, "diff =", diff)

        if diff < tf.constant(min_err, DTYPE):
            break

    # 统计全局 traversed_edges = 迭代轮数 * 全图边数
    # 先汇总本地边数：
    local_edges = int(tf.size(M_local.values))
    global_edges = hvd.allreduce(tf.constant(local_edges, tf.int64), op=hvd.Sum).numpy()
    # 估计迭代次数：traversed_edges_local_total / 本地边数
    steps_done = traversed_edges_local_total // max(local_edges, 1)
    traversed_edges_global = steps_done * global_edges

    return v.numpy(), traversed_edges_global, steps_done

# ----------------------------
# 工具：打印 Top-K
# ----------------------------
def print_topk(v, k=20):
    if hvd.rank() != 0:
        return
    v = v.reshape(-1)
    idx = np.argsort(-v)[:k]
    print("\nPrinting top {} node ids with their ranks".format(k))
    print("Rank | Node ID | Score")
    print("-"*30)
    for i, nid in enumerate(idx, 1):
        print(f"{i:4d} | {nid:7d} | {v[nid]:.6f}")

# ----------------------------
# main
# ----------------------------
def main(edge_file):
    if hvd.rank() == 0:
        print(f"World size: {hvd.size()}, this is rank {hvd.rank()} (local_rank={hvd.local_rank()})")

    t0 = time.time()
    M_local, num_nodes, local_edges, row_start, row_end = load_partitioned_sparse_block(edge_file)
    t1 = time.time()

    # 汇报图规模信息（仅 rank0 打印）
    global_edges = hvd.allreduce(tf.constant(local_edges, tf.int64), op=hvd.Sum).numpy()
    if hvd.rank() == 0:
        print(f"===> num_nodes = {num_nodes}, edges = {global_edges} (rows_local[{row_start},{row_end}))")

    t2 = time.time()
    v, traversed_edges, steps = pagerank_model_parallel(
        M_local, num_nodes, row_start, row_end,
        teleport_prob=0.15, min_err=1e-3, max_steps=1000
    )
    t3 = time.time()

    exec_time = t3 - t2
    total_time = t3 - t0
    GTEPS = traversed_edges / exec_time / 1e9 if exec_time > 0 else 0.0

    if hvd.rank() == 0:
        print("================================ TensorFlow + NCCL (Horovod) =================================")
        print(f"\tGraph: nodes: {num_nodes}, edges: {global_edges}")
        print(f"\tSteps: {steps}")
        print(f"\tExecution time: {exec_time:.3f} s")
        print(f"\tTraversed edges: {traversed_edges}")
        print(f"\tGTEPS: {GTEPS:.6f}")
        print(f"\tTotal Cost: {total_time:.3f} s")
        print("======================================================================================================")
        print_topk(v, k=20)

if __name__ == "__main__":
    root_path = '/gemini/code/graph/'
    filepath = root_path + sys.argv[1]
    main(filepath)


"""
# 安装Horovod
# 不用。export ORION_CUDART_VERSION=11.8
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod[gpu,tensorflow]

# 普通用户
mpirun -np 2 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  -x HOROVOD_GPU_ALLREDUCE=NCCL -x HOROVOD_GPU_ALLGATHER=NCCL -x HOROVOD_GPU_BROADCAST=NCCL \
  python PR-m-Horovod.py g18.edges

# root用户
mpirun --allow-run-as-root -np 8 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  -x HOROVOD_GPU_ALLREDUCE=NCCL -x HOROVOD_GPU_ALLGATHER=NCCL -x HOROVOD_GPU_BROADCAST=NCCL \
  python PR-m-Horovod.py Rmat-18.edges

"""