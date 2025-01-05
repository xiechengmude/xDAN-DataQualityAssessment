from datasets import load_dataset
import json
from pathlib import Path
import statistics

# 加载最新的数据集
dataset = load_dataset("xDAN2099/xDAN-Agentic-Chat-v1-part2_20250105_110553")

# 统计token使用情况
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0
token_stats = []

for item in dataset['train']:
    if 'token_info' in item:
        token_info = item['token_info']
        total_input_tokens += token_info['input_tokens']
        total_output_tokens += token_info['output_tokens']
        total_cost += token_info['estimated_cost']
        token_stats.append({
            'input_tokens': token_info['input_tokens'],
            'output_tokens': token_info['output_tokens'],
            'total_tokens': token_info['total_tokens'],
            'cost': token_info['estimated_cost']
        })

# 计算平均值
avg_input_tokens = statistics.mean([s['input_tokens'] for s in token_stats])
avg_output_tokens = statistics.mean([s['output_tokens'] for s in token_stats])
avg_total_tokens = statistics.mean([s['total_tokens'] for s in token_stats])
avg_cost = statistics.mean([s['cost'] for s in token_stats])

print("\n=== Token 使用统计 ===")
print(f"样本数量: {len(token_stats)}")
print(f"\n总计:")
print(f"- 输入 Tokens: {total_input_tokens:,}")
print(f"- 输出 Tokens: {total_output_tokens:,}")
print(f"- 总 Tokens: {total_input_tokens + total_output_tokens:,}")
print(f"- 总成本: ¥{total_cost:.4f}")

print(f"\n平均每条:")
print(f"- 输入 Tokens: {avg_input_tokens:.1f}")
print(f"- 输出 Tokens: {avg_output_tokens:.1f}")
print(f"- 总 Tokens: {avg_total_tokens:.1f}")
print(f"- 成本: ¥{avg_cost:.4f}")

# 输出一个示例的完整信息
print("\n=== 示例数据 ===")
example = dataset['train'][0]
print(f"问题: {example['instruction'][:200]}...")
print(f"\n回答: {example['refined_output'][:200]}...")
print("\nToken 信息:")
print(json.dumps(example['token_info'], indent=2, ensure_ascii=False))
