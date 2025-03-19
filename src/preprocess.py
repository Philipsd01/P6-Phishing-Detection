import glob
import os
import csv
import pandas as pd

def clean_csv(file_path):
    
    # Reads a CSV from file_path and applies basic cleaning steps.
    # Returns the cleaned DataFrame.
    
    df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    
    # Rename of columns (adjust to match every dataset)
    rename_map = {
        "MessageID": "id",
        "SubjectLine": "subject",
        "Sender": "sender",
        "Receiver": "receiver",
        "ReceivedDate": "date",
        "EmailBody": "body",
        "Email Text": "body",
        "Category": "label",
        "Email Type": "label"
        # etc...
    }
    df = df.rename(columns=rename_map)
    

# Ensure these columns exist, creating them if they're missing
    for col in ["sender", "subject", "receiver", "date", "label", "body"]:
        if col not in df.columns:
            df[col] = None  # or some default string

    # "fillna" won't fail, because the columns definitely exist
    df["sender"] = df["sender"].fillna("(unknown sender)")
    df["subject"] = df["subject"].fillna("(no subject)")
    df["receiver"] = df["receiver"].fillna("(unknown receiver)")
    df["date"] = df["date"].fillna("(unknown date)")

    # Drop rows with missing label or body
    df.dropna(subset=["label", "body"], inplace=True)
    # Drop columns that are not needed
    columns_to_drop = ["Unnamed: 0", "urls", "sender", "receiver", "date"] # Add more columns to drop if needed
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    
    # Lowercase and strip subject/body text
    if "subject" in df.columns:
        df["subject"] = (
            df["subject"]
            .astype(str)                            # Ensure it’s string
            .str.lower()                            # Convert to lowercase
            .str.strip()                            # Remove leading/trailing whitespace
            .str.replace('\n', ' ', regex=True)     # Replace newlines with a space
        )
    if "body" in df.columns:
        df["body"] = (
            df["body"]
            .astype(str)
            .str.lower()
            .str.replace('\n', ' ', regex=True)
            .str.strip()
        )
    
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
        # etc...
    }

    if "label" in df.columns:
        df["label"] = df["label"].map(label_map).fillna(df["label"])
        # fillna() to keep unknown labels as-is or re-map them
    df["label"] = pd.to_numeric(df["label"], errors="coerce")  # Convert "1.0" -> 1.0 float
    df["label"] = df["label"].astype("Int64")                  # or int if you're sure there are no NaNs


    # Ensure consistent column order
    column_order = ["subject", "body", "label"]
    df = df[[col for col in column_order if col in df.columns]]


    # Return the cleaned DataFrame
    return df

csv_files = glob.glob('data/unprocessed_data/*.csv')

for file_path in csv_files:
    print(f"Processing {file_path}...")
    try:
        cleaned_df = clean_csv(file_path)

        # Build a new filename and places it in different directory
        base_name = os.path.basename(file_path)        # Extracts the filename from the path
        name_no_ext = os.path.splitext(base_name)[0]   # Removes the file extension
        cleaned_name = f"{name_no_ext}_cleaned.csv"    # Adds a new file extension to the filename
        output_path = os.path.join('data/processed_data/', cleaned_name) # Combines the new filename with a directory

        cleaned_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL) # Save the cleaned DataFrame to a new CSV file putting all fields in quotes
        print(f"Saved cleaned file to {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Done!")