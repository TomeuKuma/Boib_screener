import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline
from transformers import DataCollatorWithPadding
import pandas as pd
import torch
from datasets import Dataset, DatasetDict

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

#model_name = "dccuchile/distilbert-base-spanish-uncased"
map_fase = {0: "otros", 1: "bases", 2: "convocatoria", 
            3: "lista de aspirantes", 4: "lista definitiva", 5: "tribunales",
            6: "correccion de erorres", 7: "modificacion"}
data = df_to_dataset(load_df())
text = "Aprobación de la convocatoria y de las bases que han de regir el proceso de provisión de puestos de trabajo de Jefes/as de Área del Patronat Municipal de l´Habitatge i Rehabilitació Integral de Barris con personal laboral fijo de este mismo organismo público, mediante el procedimiento de concurso de méritos"
model_name = "ditilbert_es_fase/checkpoint-500"
classifier = pipeline("sentiment-analysis", model=model_name)
out = classifier(text)

test_set = data["test"]["text"]
out = classifier(test_set)

#data = pd.DataFrame([{**o, **{"text":t, "label_true": l}} for o, t,l  in zip(out, test_set, data["test"]["label"])])
#print(data)

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])