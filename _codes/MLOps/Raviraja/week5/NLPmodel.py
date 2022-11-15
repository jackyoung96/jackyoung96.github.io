# !python -m pip install pytorch_lightning torch transformers datasets sklearn torchmetrics wandb matplotlib
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
import torchmetrics
from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix, accuracy_score

from datasets import load_dataset
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import onnxruntime as ort
from scipy.special import softmax

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # Transformer (BERT) model

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        # processing the data
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])

        preds = torch.argmax(logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("train/accuracy", train_acc.detach().cpu(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = self.val_accuracy_metric(preds, batch["label"])
        self.log("val/loss", loss.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val/acc", val_acc.detach().cpu(), prog_bar=True, on_epoch=True)
        return {"labels": batch["label"], "logits": logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        labels = labels.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        data = confusion_matrix(labels, preds)
        df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(7, 4))
        plot = sns.heatmap(
            df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        )  # font size
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

class ColaPredictor:
    def __init__(self, model_path):
        # creating the onnxruntime session
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        # Preparing inputs
        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        # Run the model (None = get all the outputs)
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Normalising the outputs
        scores = softmax(ort_outs[0])[0].tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions