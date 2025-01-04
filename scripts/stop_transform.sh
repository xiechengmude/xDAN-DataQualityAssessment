#!/bin/bash

# 检查PID文件是否存在
if [ -f logs/transform.pid ]; then
    PID=$(cat logs/transform.pid)
    
    # 检查进程是否仍在运行
    if ps -p $PID > /dev/null; then
        echo "Stopping data transformation process (PID: $PID)..."
        kill $PID
        rm logs/transform.pid
        echo "Process stopped successfully."
    else
        echo "Process is not running."
        rm logs/transform.pid
    fi
else
    echo "No running process found."
fi
