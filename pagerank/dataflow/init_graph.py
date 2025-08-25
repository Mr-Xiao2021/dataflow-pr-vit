# import sys
# import numpy as np
# import pandas as pd
# import time

# start = time.time()
# origin = sys.argv[1]
# processed = sys.argv[2]
# df = pd.read_csv(origin, sep=' ', header=None, comment='#')
# coo = df.to_numpy()
# nodes = np.unique(coo)

# reorder = {}
# for i in range(nodes.shape[0]):
#     reorder[nodes[i]] = i
    
# for i in range(coo.shape[0]):
#     coo[i][0] = reorder[coo[i][0]]
#     coo[i][1] = reorder[coo[i][1]]


# coo = coo[coo[:,1].argsort()]

# coo.T.astype(np.int32).tofile(processed)
# end = time.time()
# print(f"===> nodes = {len(reorder)}, edges = {coo.shape[0]}")
# print(f"Graph node: {len(reorder)}, edges: {coo.shape[0]}")
# print(f"Init Graph: {end - start:.4f} seconds")



import sys
import numpy as np
import pandas as pd
import time


origin = sys.argv[1]
processed = sys.argv[2]
start = time.time()

# 高效读取：pandas 直接读为 int32 数组
df = pd.read_csv(origin, sep=' ', header=None, comment='#', dtype=np.int64)
coo = df.to_numpy()

# 去重并排序节点（unique 已经会排序）
nodes, inverse = np.unique(coo, return_inverse=True)

# 直接用 inverse 得到映射后的边表
coo = inverse.reshape(coo.shape)

# 按目标节点排序 (第二列)
coo = coo[np.argsort(coo[:, 1])]

# 存储为 int32 二进制
coo.T.astype(np.int32).tofile(processed)

end = time.time()
print(f"===> nodes = {len(nodes)}, edges = {coo.shape[0]}")
print(f"Graph node: {len(nodes)}, edges: {coo.shape[0]}")
print(f"Init Graph: {end - start:.4f} seconds")
