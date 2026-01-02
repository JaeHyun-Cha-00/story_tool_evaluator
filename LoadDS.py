from datasets import load_dataset
import pandas as pd

ds = load_dataset("Disya/eq-bench-creative-writing-v3")

def get_dataset():
    return ds['train']

# Extract a "model" column
def get_models():
    dataset = get_dataset()
    models = [item['model'] for item in dataset]
    return models

# Extract a "response" column
def get_responses():
    dataset = get_dataset()
    responses = [item['response'] for item in dataset]
    return responses

# Create a CSV file for "model" and "response" column
def to_csv(file_path: str):
    dataset = get_dataset()
    data = {
        'model': [item['model'] for item in dataset],
        'response': [item['response'] for item in dataset]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

if __name__ == "__main__":
    df = to_csv("dataset/creative_writing_dataset.csv")
    print(df.head())