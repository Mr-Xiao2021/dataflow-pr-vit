#!/bin/bash
# 默认参数
card_num=1
graph_name="Rmat-18"
data_path="/gemini/code/graph"

# 更新软件包列表
# apt-get update

# 安装 bc 计算器工具
# apt-get install -y bc

# 参数解析
while getopts ":c:g:" opt; do
  case $opt in
    c)
      card_num=$OPTARG
      ;;
    g)
      graph_name=$OPTARG
      ;;
    *)
      echo "Usage: bash run.sh -g <graph_name> -c <card_num>"
      exit 1
      ;;
  esac
done

echo "================= CONFIG ================="
echo "Graph name : ${graph_name}"
echo "Card num   : ${card_num}"
echo "Data path  : ${data_path}"
echo "=========================================="

START_TIME=$(date +%s.%3N)

# 编译逻辑
echo "Compiling..."
if [ "$card_num" -eq 1 ]; then
    nvcc -o pagerank -lgomp src/main.cc src/mc/mc-cuda.cu
else
    nvcc -o pagerank -lgomp src/main.cc src/mc/multi-gpu.cu
fi
echo "Compile Finished"

# 预处理图
echo "Running init_graph.py to preprocess graph..."
python_output=$(python init_graph.py "${data_path}/${graph_name}.edges" "${data_path}/${graph_name}.bin")
echo "$python_output"

# 提取节点数和边数
nodes=$(echo "$python_output" | grep -oP '(?<=nodes\s=\s)\d+')
edges=$(echo "$python_output" | grep -oP '(?<=edges\s=\s)\d+')

if [ -z "$nodes" ] || [ -z "$edges" ]; then
    echo "[Error]: Failed to parse nodes or edges from init_graph.py output."
    exit 1
fi



# 执行主程序
echo "Start calculating..."
./pagerank -f "${data_path}/${graph_name}.bin" -n "$nodes" -e "$edges" -c "$card_num"

END_TIME=$(date +%s.%3N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo -e "\t\tTotal Cost: $ELAPSED_TIME seconds"
echo "================================ DATAFLOW METHOD ======================================="

# python rank_show.py
