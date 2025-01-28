from datasets import load_dataset

def main():
    print("Downloading dataset...")
    dataset = load_dataset("xDAN2099/xDAN-Agentic-Chat-v1-alpaca-sharegpt-sample")
    print("Dataset downloaded and cached successfully!")
    print(f"Dataset info: {dataset}")

if __name__ == "__main__":
    main()
