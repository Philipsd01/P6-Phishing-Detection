import glob
import os
import csv
import pandas as pd
from io import StringIO

def clean_csv(file_path):
    # Open file and remove commented lines
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    filtered = [line for line in lines if not line.lstrip().startswith('//')]
    csv_data = StringIO(''.join(filtered))
    
    # Now read the cleaned CSV data
    df = pd.read_csv(csv_data, engine='python', on_bad_lines='skip')
    
    # Rename columns if necessary
    rename_map = {
        "SubjectLine": "subject",
        "EmailBody": "body",
        "Email Text": "body",
        "Message": "body",
        "text": "body",
        "Category": "label",
        "Email Type": "label",
        "spam": "label"
    }
    df = df.rename(columns=rename_map)
    
    # Ensure these columns exist by adding missing ones
    for col in ["sender", "subject", "receiver", "date", "label", "body"]:
        if col not in df.columns:
            df[col] = None

    # Drop rows with missing label or body
    df.dropna(subset=["label", "body"], inplace=True)
    
    # Drop unneeded columns
    columns_to_drop = ["Unnamed: 0", "sender", "receiver", "date", "urls"]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Clean subject and body text
    if "subject" in df.columns:
        df["subject"] = (df["subject"]
                         .astype(str)
                         .str.lower()
                         .str.strip()
                         .str.replace('\n', ' ', regex=True))
    if "body" in df.columns:
        df["body"] = (df["body"]
                      .astype(str)
                      .str.lower()
                      .str.replace('\n', ' ', regex=True)
                      .str.strip())
    
    # Standardize labels to numeric
    label_map = {
        "ham": 0,
        "legitimate": 0,
        "0.0": 0,
        "Safe Email": 0,
        "phishing": 1,
        "spam": 1,
        "1.0": 1,
        "Phishing Email": 1
    }
    if "label" in df.columns:
        df["label"] = df["label"].map(label_map).fillna(df["label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    
    # Keep only the desired columns in order
    column_order = ["subject", "body", "label"]
    df = df[[col for col in column_order if col in df.columns]]
    
    return df

# Process all CSV files in the validation folder and combine them
csv_files = glob.glob('data/validation/*.csv')
combined_dfs = []

for file_path in csv_files:
    print(f"Processing {file_path}...")
    try:
        cleaned_df = clean_csv(file_path)
        combined_dfs.append(cleaned_df)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if combined_dfs:
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    output_combined_path = os.path.join('data', 'validation', 'cleaned', 'combined_cleaned_sample.csv')
    combined_df.to_csv(output_combined_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved combined cleaned dataset to {output_combined_path}")
else:
    print("No files were cleaned successfully.")

print("Done!")

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Clean NaNs if needed
    df.dropna(subset=["subject", "body", "label"], inplace=True)

    # Combine subject and body
    df['text'] = df['subject'] + " " + df['body']
    
    # Ensure label is integer
    df['label'] = df['label'].astype(int)

    return df[['text', 'label']]
