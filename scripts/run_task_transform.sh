#!/bin/bash

# 设置环境变量
export PYTHONPATH=/Users/gumpcehng/CascadeProjects/DataQualityAssessment:$PYTHONPATH

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 在后台运行程序，将输出重定向到日志文件
nohup python -m scripts.transform_data \
    --config config/data_transform.yaml \
    > logs/transform_${TIMESTAMP}.log 2>&1 &

# 获取进程ID
PID=$!

# 将进程ID写入文件
echo $PID > logs/transform.pid

echo "Started data transformation process with PID: $PID"
echo "Check logs at: logs/transform_${TIMESTAMP}.log"
