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
        --output)
            OUTPUT_ARG="--output $2"
            shift 2
            ;;
        --push-to-hub)
            PUSH_ARG="--push-to-hub"
            shift
            ;;
        --no-push-to-hub)
            PUSH_ARG="--no-push-to-hub"
            shift
            ;;
        *)
            echo "Usage: $0 [--config config_file.yaml] [--output output_file.json] [--push-to-hub | --no-push-to-hub]"
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

# 构建命令
CMD="python scripts/transform_data.py --config $CONFIG_FILE"
if [ ! -z "$OUTPUT_ARG" ]; then
    CMD="$CMD $OUTPUT_ARG"
    echo "Output path overridden by command line argument"
fi
if [ ! -z "$PUSH_ARG" ]; then
    CMD="$CMD $PUSH_ARG"
    if [ "$PUSH_ARG" == "--push-to-hub" ]; then
        echo "Will push results to HuggingFace Hub (forced by command line)"
    else
        echo "Will NOT push results to HuggingFace Hub (disabled by command line)"
    fi
fi

# 运行转换脚本
eval $CMD

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "Data transformation completed successfully!"
else
    echo "Error occurred during transformation"
    exit 1
fi
