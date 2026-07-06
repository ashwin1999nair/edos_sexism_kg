"""
data_loader.py
--------------
Loads and preprocesses the EDOS dataset for Tasks A and B.
Provides class distributions, label mappings, and PyTorch Dataset objects
ready for use with HuggingFace Trainer.
"""

import os
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

# LABEL MAPPINGS
TASK_A_LABELS={"not sexist": 0,"sexist": 1}
TASK_B_LABELS = {"1. threats, plans to harm and incitement": 0,"2. derogation": 1,"3. animosity": 2,"4. prejudiced discussions": 3}

TASK_A_ID2LABEL={v: k for k, v in TASK_A_LABELS.items()}
TASK_B_ID2LABEL={v: k for k, v in TASK_B_LABELS.items()}

# DATA LOADING
def load_edos_data(data_dir: str, task: str):
    if task not in ("A", "B"):
        raise ValueError(f"Task must be 'A' or 'B', got '{task}'")
    train_path=os.path.join(data_dir, "edos_labelled_aggregated.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset not found at {train_path}. "
            "Please download from https://github.com/rewire-online/edos "
            "and place edos_labelled_aggregated.csv in your data/ folder.")
    df = pd.read_csv(train_path)
    # Validate expected columns exist
    required_cols= ["text", "label_sexist", "label_category", "split"]
    missing_cols= [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}") 
    train_df= df[df["split"] == "train"].copy()
    dev_df= df[df["split"] == "dev"].copy()
    test_df= df[df["split"] == "test"].copy()

    for split_df in [train_df, dev_df, test_df]:
        split_df.reset_index(drop=True, inplace=True)
        assign_labels(split_df, task)

    return train_df, dev_df, test_df

def assign_labels(df: pd.DataFrame, task: str):
    if task == "A":
        df["label"]=df["label_sexist"]
        df["label_id"]=df["label"].map(TASK_A_LABELS)
    else:
        mask=df["label_sexist"] == "sexist"
        df.drop(df[~mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["label"]=df["label_category"]
        df["label_id"]=df["label"].map(TASK_B_LABELS)

    unmapped = df["label_id"].isna().sum()
    if unmapped>0:
        unmapped_vals = df.loc[df["label_id"].isna(), "label"].unique()
        raise ValueError(f"{unmapped} rows have unmapped labels: {unmapped_vals}")
    

# CLASS DISTRIBUTION
def print_class_distribution(df: pd.DataFrame, split_name: str, task: str):
    counts=Counter(df["label"])
    total=len(df)
    print(f"Task {task} : {split_name} split ({total} posts)")
    for label, count in sorted(counts.items()):
        pct=count / total * 100
        print(f"{label:<45} {count:>5} ({pct:.1f}%)")

# PYTORCH DATASET
class EDOSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts=df["text"].tolist() # list of raw text strings
        self.labels=df["label_id"].tolist() # list of integers (0, 1, etc.)
        self.tokenizer=tokenizer # RoBERTa tokenizer
        self.max_length=max_length # max tokens per post (128)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding=self.tokenizer(self.texts[idx],truncation=True,padding="max_length",max_length=self.max_length,return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(),"attention_mask": encoding["attention_mask"].squeeze(),
            "labels": self.labels[idx]}
    
def get_datasets(data_dir: str, task: str, max_length: int = 128):
    train_df, dev_df, test_df=load_edos_data(data_dir, task)

    for name, df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        print_class_distribution(df, name, task)

    tokenizer=RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset=EDOSDataset(train_df, tokenizer, max_length)
    dev_dataset=EDOSDataset(dev_df,   tokenizer, max_length)
    test_dataset=EDOSDataset(test_df,  tokenizer, max_length)

    num_labels=2 if task=="A" else 4
    id2label=TASK_A_ID2LABEL if task=="A" else TASK_B_ID2LABEL
    print(f"\nDatasets ready — Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")
    print(f"num_labels: {num_labels}")

    return train_dataset, dev_dataset, test_dataset, num_labels, id2label
