import pandas as pd

def load_and_prepare_data(csv_path):
    """
    Loads a CSV file and returns a DataFrame with 'text' and 'label' columns.
    Combines 'subject' and 'body' into a single 'text' column.
    """
    df = pd.read_csv(csv_path)
    
    # Clean NaNs if needed
    df.dropna(subset=["subject", "body", "label"], inplace=True)

    # Combine subject and body
    df['text'] = df['subject'] + " " + df['body']
    
    # Ensure label is integer
    df['label'] = df['label'].astype(int)

    return df[['text', 'label']]
