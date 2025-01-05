#!/bin/bash

# 获取脚本所在目录的上级目录（项目根目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 设置环境变量
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 创建日志目录
mkdir -p $PROJECT_ROOT/logs

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 在后台运行程序，将输出重定向到日志文件
nohup python -m scripts.transform_data \
    --config $PROJECT_ROOT/config/data_transform.yaml \
    > $PROJECT_ROOT/logs/transform_${TIMESTAMP}.log 2>&1 &

# 获取进程ID
PID=$!

# 将进程ID写入文件
echo $PID > $PROJECT_ROOT/logs/transform.pid

echo "Started data transformation process with PID: $PID"
echo "Check logs at: $PROJECT_ROOT/logs/transform_${TIMESTAMP}.log"
