import os
import pandas as pd
import json

def load_json_to_df(filepath, label):
    with open(filepath, 'r') as file:
        data = json.load(file)

    texts = []
    for entry in data:
        if isinstance(entry, dict):
            text = entry.get('complaint_text')
            if text:
                texts.append(text)
        elif isinstance(entry, str):
            texts.append(entry)

    return pd.DataFrame({'text': texts, 'label': label})

def main():
    data_dir = 'data'
    merged_df = pd.DataFrame()

    label_map = {
        'electricity': ['electricity.json', 'electricity_complaints.json'],
        'water': ['water.json', 'water_complaints.json'],
        'garbage': ['garbage.json', 'garbage_complaints.json'],
        'road': ['road.json', 'road_complaints.json']
    }

    for label, files in label_map.items():
        for filename in files:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                print(f"Loading {filename} as {label}")
                df = load_json_to_df(path, label)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            else:
                print(f"File {filename} not found. Skipping.")

    print("Total complaints:", len(merged_df))
    merged_df.to_csv(os.path.join(data_dir, 'merged_complaints.csv'), index=False)
    print("Merged dataset saved to data/merged_complaints.csv")

if __name__ == "__main__":
    main()
