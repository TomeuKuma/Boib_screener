import sys
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate

model_name = "dccuchile/distilbert-base-spanish-uncased"
map_fase = {0: "otros", 1: "bases", 2: "convocatoria", 
            3: "lista de aspirantes", 4: "lista definitiva", 5: "tribunales",
            6: "correccion de erorres", 7: "modificacion"}


def df_to_dataset(df, train_ix=160, test_ix=190):
    ds_dict = {"train": Dataset.from_pandas(df.iloc[:train_ix]),
           "test": Dataset.from_pandas(df.iloc[train_ix:test_ix]),
           "validation": Dataset.from_pandas(df.iloc[test_ix:])}
    ds = DatasetDict(ds_dict)
    return ds


def load_df():  
    file_name = "boib_train.csv"
    df = pd.read_csv(file_name)
    df["label"] = df["Fase"].copy()#.replace(map_fase)
    df_out = df.rename(columns={"Resolucion": "text"}).drop(columns=["Fase"]).copy()
    return df_out


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/distilbert-base-spanish-uncased")
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)



def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = df_to_dataset(load_df())
    data["train"]
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    id2label = map_fase
    label2id = {v:k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    training_args = TrainingArguments(
        output_dir="ditilbert_es_fase",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()