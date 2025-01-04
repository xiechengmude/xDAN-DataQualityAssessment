from datasets import load_dataset

# 加载数据集
dataset = load_dataset("xDAN2099/xDAN-Agentic-Chat-v1-part2_20250105_041914")

# 打印第一个样本
print("\nFirst sample:")
print(dataset['train'][0])

# 打印数据集信息
print("\nDataset info:")
print(dataset)
