单卡：
```bash
python PR-GPU.py <GraphName>
```

分布式：
```bash
# 安装Horovod
# 不用。export ORION_CUDART_VERSION=11.8
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod[gpu,tensorflow]

# root用户
mpirun --allow-run-as-root -np <card_num> \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  -x HOROVOD_GPU_ALLREDUCE=NCCL -x HOROVOD_GPU_ALLGATHER=NCCL -x HOROVOD_GPU_BROADCAST=NCCL \
  python PR-m-Horovod.py <GraphName>

```