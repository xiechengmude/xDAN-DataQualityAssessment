#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 检查是否提供了任务ID
if [ -z "$1" ]; then
    echo "请提供任务ID"
    echo "用法: $0 TASK_ID"
    exit 1
fi

TASK_ID=$1
TASK_DIR="${PROJECT_ROOT}/outputs/${TASK_ID}"
PID_FILE="${TASK_DIR}/task.pid"
LOG_FILE="${TASK_DIR}/task.log"
CONFIG_FILE="${TASK_DIR}/task_config.yaml"

# 检查任务目录是否存在
if [ ! -d "$TASK_DIR" ]; then
    echo "错误: 任务目录不存在: $TASK_DIR"
    exit 1
fi

# 读取进程ID
if [ ! -f "$PID_FILE" ]; then
    echo "错误: 找不到进程ID文件: $PID_FILE"
    exit 1
fi

PID=$(cat "$PID_FILE")

# 检查进程是否在运行
if ps -p $PID > /dev/null; then
    STATUS="运行中"
    # 获取CPU和内存使用情况
    if command -v ps >/dev/null 2>&1; then
        CPU_MEM=$(ps -p $PID -o %cpu,%mem | tail -n 1)
        CPU=$(echo $CPU_MEM | awk '{print $1}')
        MEM=$(echo $CPU_MEM | awk '{print $2}')
        STATUS="${STATUS} (CPU: ${CPU}%, MEM: ${MEM}%)"
    fi
else
    STATUS="已完成"
fi

# 打印任务信息
echo "任务状态:"
echo "----------------------------------------"
echo "任务ID: ${TASK_ID}"
echo "状态: ${STATUS}"
echo "进程ID: ${PID}"
echo "日志文件: ${LOG_FILE}"
echo "配置文件: ${CONFIG_FILE}"
echo "----------------------------------------"

# 如果任务已完成，显示最后10行日志
if [ "$STATUS" = "已完成" ]; then
    echo ""
    echo "最后10行日志:"
    echo "----------------------------------------"
    tail -n 10 "$LOG_FILE"
fi
