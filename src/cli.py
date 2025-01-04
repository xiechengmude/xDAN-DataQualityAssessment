import asyncio
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
import time
import logging
from typing import Optional
import os

from .data_loader import DataLoader
from .processor import DataProcessor

app = typer.Typer()
console = Console()

def get_default_config_path() -> Path:
    """获取默认配置文件路径"""
    # 优先使用环境变量中的配置路径
    if 'DQA_CONFIG_PATH' in os.environ:
        return Path(os.environ['DQA_CONFIG_PATH'])
    
    # 否则使用相对于项目根目录的默认路径
    return Path(__file__).parent.parent / 'config' / 'default_config.yaml'

def setup_logging(config_path: str):
    """设置日志"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_config = config['logging']
    log_path = Path(log_config['save_path'])
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / log_config['file_name']),
            logging.StreamHandler()
        ]
    )

@app.command()
def process_dataset(
    config_path: Path = typer.Option(
        None,
        help="配置文件路径，如果不指定则使用默认路径"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="输出目录"
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        help="数据集名称，例如：tatsu-lab/alpaca"
    ),
    dataset_split: Optional[str] = typer.Option(
        None,
        help="数据集分片，例如：train"
    )
):
    """处理数据集"""
    try:
        # 设置日志
        if config_path is None:
            config_path = get_default_config_path()
        setup_logging(config_path)
        
        # 初始化组件
        data_loader = DataLoader(config_path)
        processor = DataProcessor(config_path)
        
        # 如果指定了数据集名称，更新配置
        if dataset_name:
            data_loader.config['datasets'][0]['name'] = dataset_name
            if dataset_split:
                data_loader.config['datasets'][0]['split'] = dataset_split
        
        # 开始处理
        start_time = time.time()
        console.print("[bold green]Starting dataset processing...[/bold green]")
        
        # 加载数据
        items = data_loader.load_and_convert()
        
        # 批量处理
        batch_size = processor.config['concurrency']['batch_size']
        batches = [items[i:i + batch_size] 
                  for i in range(0, len(items), batch_size)]
        
        all_results = []
        all_failed = []
        
        for batch in batches:
            result = asyncio.run(processor.process_batch(batch))
            all_results.extend(result.successful)
            all_failed.extend(result.failed)
        
        # 过滤结果
        filtered_results = processor.filter_results(all_results)
        
        # 保存结果
        if output_dir is None:
            output_dir = Path(processor.config['output']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_loader.save_processed_data(
            filtered_results,
            output_dir / "successful.jsonl"
        )
        data_loader.save_processed_data(
            all_failed,
            output_dir / "failed.jsonl"
        )
        
        # 计算统计信息
        total_time = time.time() - start_time
        total_items = len(items)
        successful_items = len(filtered_results)
        failed_items = len(all_failed)
        filtered_items = len(all_results) - len(filtered_results)
        
        # 创建结果表格
        table = Table(title="Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Items", str(total_items))
        table.add_row("Successfully Processed", str(successful_items))
        table.add_row("Failed Items", str(failed_items))
        table.add_row("Filtered Items", str(filtered_items))
        table.add_row("Success Rate", f"{(successful_items/total_items)*100:.2f}%")
        table.add_row("Processing Time", f"{total_time:.2f} seconds")
        table.add_row("Items per Second", f"{total_items/total_time:.2f}")
        
        console.print(table)
        
        # 输出文件位置
        console.print(f"\n[bold green]Results saved to:[/bold green]")
        console.print(f"Successful items: {output_dir}/successful.jsonl")
        console.print(f"Failed items: {output_dir}/failed.jsonl")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
