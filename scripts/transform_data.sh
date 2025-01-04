#!/bin/bash

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 默认配置文件路径
CONFIG_FILE="config/data_transform.yaml"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--config config_file.yaml]"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting data transformation..."
echo "Using config file: $CONFIG_FILE"

# 运行转换脚本
python scripts/transform_data.py --config "$CONFIG_FILE"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "Data transformation completed successfully!"
else
    echo "Error occurred during transformation"
    exit 1
fi
