#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from datasets import load_dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processor import DataProcessor
from src.models import AlpacaItem, BatchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """Load configuration file"""
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_task_name() -> str:
    """Generate task name in format: task_quality_YYYYMMDD_HHMMSS"""
    now = datetime.now()
    return f"task_quality_{now.strftime('%Y%m%d_%H%M%S')}"

def load_hf_dataset(dataset_config: dict) -> List[AlpacaItem]:
    """Load dataset from HuggingFace
    
    Args:
        dataset_config: Dataset configuration containing:
            - name: dataset name
            - config: optional config name
            - split: optional dataset split
            - num_samples: optional sample count
            - field_mapping: field mapping configuration
    
    Returns:
        List[AlpacaItem]: List of loaded data items
    """
    logger.info(f"Loading dataset: {dataset_config['name']}")
    
    # Load dataset
    dataset_args = {
        "path": dataset_config["name"],
        "split": dataset_config.get("split", "train")
    }
    if dataset_config.get("config"):
        dataset_args["name"] = dataset_config["config"]
    
    dataset = load_dataset(**dataset_args)
    
    # Sample if specified
    if dataset_config.get("num_samples", -1) > 0:
        dataset = dataset.shuffle(seed=42).select(range(dataset_config["num_samples"]))
    
    # Convert to AlpacaItems
    items = []
    field_mapping = dataset_config["field_mapping"]
    
    for entry in dataset:
        item = AlpacaItem(
            instruction=entry[field_mapping["instruction"]] if field_mapping.get("instruction") else "",
            input=entry[field_mapping["input"]] if field_mapping.get("input") else "",
            output=entry[field_mapping["output"]] if field_mapping.get("output") else "",
            sources=dataset_config["name"]  # Changed from list to string
        )
        items.append(item)
    
    return items

def load_assessment_data(config: dict) -> List[AlpacaItem]:
    """Load assessment data from configured datasets
    
    Args:
        config: Configuration containing datasets field
    
    Returns:
        List[AlpacaItem]: List of loaded data items
    """
    all_items = []
    
    for dataset_config in config["datasets"]:
        try:
            items = load_hf_dataset(dataset_config)
            all_items.extend(items)
            logger.info(f"Loaded {len(items)} items from {dataset_config['name']}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config['name']}: {str(e)}")
    
    return all_items

def save_results(results: List[AlpacaItem], task_name: str, output_dir: Path, config: dict):
    """Save assessment results
    
    Args:
        results: List of assessment results
        task_name: Task name
        output_dir: Output directory
        config: Configuration information
    """
    timestamp = datetime.now()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build result data
    result_data = {
        "task_info": {
            "task_name": task_name,
            "timestamp": timestamp.isoformat(),
            "model_name": config['openai']['model_name'],
            "total_samples": len(results)
        },
        "results": []
    }
    
    # Process each result item
    for item in results:
        item_dict = item.model_dump()
        
        # Add model info and timestamp to metadata
        if 'metadata' not in item_dict:
            item_dict['metadata'] = {}
        item_dict['metadata'].update({
            'model_name': config['openai']['model_name'],
            'timestamp': timestamp.isoformat()
        })
        
        result_data['results'].append(item_dict)
    
    # Save as JSON file
    output_file = output_dir / f"{task_name}_results.json"
    logger.info(f"Saving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # Push to Hugging Face Hub if configured
    output_config = config.get('output', {})
    if output_config.get('push_to_hub'):
        try:
            from datasets import Dataset
            
            # Convert results to a format suitable for Dataset
            dataset_dict = {
                "task_name": [],
                "timestamp": [],
                "model_name": [],
                "instruction": [],
                "input": [],
                "output": [],
                "processed_output": [],
                "category": [],
                "quality_metrics": [],
                "score": [],
                "metadata": [],
                "sources": []  # Added sources field
            }
            
            # Populate the dataset dictionary
            for item in results:
                item_dict = item.model_dump()
                dataset_dict["task_name"].append(task_name)
                dataset_dict["timestamp"].append(timestamp.isoformat())
                dataset_dict["model_name"].append(config['openai']['model_name'])
                dataset_dict["instruction"].append(item_dict["instruction"])
                dataset_dict["input"].append(item_dict["input"])
                dataset_dict["output"].append(item_dict["output"])
                dataset_dict["processed_output"].append(item_dict["processed_output"])
                dataset_dict["category"].append(item_dict["category"])
                dataset_dict["quality_metrics"].append(json.dumps(item_dict["quality_metrics"]))
                dataset_dict["score"].append(item_dict["score"])
                dataset_dict["metadata"].append(json.dumps(item_dict["metadata"]))
                dataset_dict["sources"].append(item_dict.get("sources", ""))  # Get sources with default empty string
            
            # Create and push dataset
            hub_config = output_config.get('hub_config', {})
            dataset = Dataset.from_dict(dataset_dict)
            
            # Push to hub with repository name as task_name
            repo_id = f"{hub_config['repository_id']}/{task_name}"
            dataset.push_to_hub(
                repo_id,
                token=hub_config['token'],
                private=hub_config.get('private', True)
            )
            logger.info(f"Successfully pushed results to Hugging Face Hub: {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push results to Hugging Face Hub: {str(e)}")

async def run_assessment(config_path: str = "config/default_config.yaml"):
    """Run the complete data quality assessment
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(Path(config_path))
    
    # Generate task name
    task_name = get_task_name()
    logger.info(f"Starting assessment task: {task_name}")
    
    # Load assessment data
    items = load_assessment_data(config)
    logger.info(f"Loaded {len(items)} items in total")
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Process items using process_batch instead of process_items
    batch_result = await processor.process_batch(items)
    logger.info(f"Processed {len(batch_result.successful)} items successfully, {len(batch_result.failed)} items failed")
    
    # Save results
    output_dir = Path(config['output']['base_dir'])
    save_results(batch_result.successful, task_name, output_dir, config)
    logger.info("Assessment completed successfully")

async def main():
    """Main entry point"""
    try:
        await run_assessment()
    except Exception as e:
        logger.error(f"Error during assessment: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
