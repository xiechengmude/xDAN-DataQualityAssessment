#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 获取当前时间作为任务ID
TASK_ID=$(date +%Y%m%d_%H%M%S)
TASK_DIR="${PROJECT_ROOT}/outputs/${TASK_ID}"
LOG_FILE="${TASK_DIR}/task.log"
CONFIG_FILE=${1:-"${PROJECT_ROOT}/config/default_config.yaml"}

# 创建输出目录
mkdir -p "$TASK_DIR"

# 打印任务信息
echo "启动数据质量评估任务..."
echo "配置文件: ${CONFIG_FILE}"
echo ""

# 运行Python脚本并将输出重定向到日志文件
cd "${PROJECT_ROOT}"
nohup python "${SCRIPT_DIR}/run_assessment.py" --config "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &

# 获取进程ID
PID=$!

# 保存进程ID和配置信息到文件
echo $PID > "${TASK_DIR}/task.pid"
cp "${CONFIG_FILE}" "${TASK_DIR}/task_config.yaml"

echo "任务已在后台启动"
echo "任务ID: ${TASK_ID}"
echo "进程ID: ${PID}"
echo "日志文件: ${LOG_FILE}"
echo "配置文件: ${CONFIG_FILE}"
echo ""
echo "使用以下命令查看日志:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "使用以下命令检查任务状态:"
echo "${SCRIPT_DIR}/check_task.sh ${TASK_ID}"
