import sys

from datetime import datetime
from typing import Optional

import datasets
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import datasets
from datasets import load_dataset, Dataset, DatasetDict

def df_to_dataset(df, train_ix=160, test_ix=190):
    ds_dict = {"train": Dataset.from_dict(df.iloc[:train_ix].to_dict()),
           "test": Dataset.from_dict(df.iloc[train_ix:test_ix].to_dict()),
           "validation": Dataset.from_dict(df.iloc[test_ix:].to_dict())}
    ds = DatasetDict(ds_dict)
    return ds

def load_df():
    map_fase = {0: "otros", 1: "bases", 2: "convocatoria", 
            3: "lista de aspirantes", 4: "lista definitiva", 5: "tribunales",
            6: "correccion de erorres", 7: "modificacion"}
    file_name = "boib_train.csv"
    df = pd.read_csv(file_name)
    #df["label"] = df["Fase"].replace(map_fase)
    df_out = df.rename(columns={"Resolucion": "text"}).drop(columns=["Fase"]).copy()
    return df_out

