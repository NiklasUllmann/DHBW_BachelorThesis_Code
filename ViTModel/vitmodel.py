from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from vit_pytorch.vit import ViT
from vit_pytorch.recorder import Recorder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from x_transformers import Encoder
from sklearn.metrics import f1_score, accuracy_score



class ViTModel():
    def __init__(self, load=False, path=None):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if(not load):
            self.model = ViT(
                image_size=320,
                patch_size=16,
                num_classes=10,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.5,
                emb_dropout=0.1,
            ).to(self.device)
            print("init ViT")
        else:
            self.model = torch.load(path, map_location=self.device)
            print("load ViT")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)

    def train(self, train, val, epoch):
        train_loss_record = []
        train_acc_record = []
        val_loss_record = []
        val_acc_record = []
        for epoch in range(epoch):
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in tqdm(train):
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train)
                epoch_loss += loss / len(train)

            train_acc_record.append(
                epoch_accuracy.detach().cpu().numpy().flat[0])
            train_loss_record.append(epoch_loss.detach().cpu().numpy().flat[0])

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in val:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    val_output = self.model(data)
                    val_loss = self.criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(val)
                    epoch_val_loss += val_loss / len(val)

                val_acc_record.append(
                    epoch_val_accuracy.detach().cpu().numpy().flat[0])
                val_loss_record.append(
                    epoch_val_loss.detach().cpu().numpy().flat[0])

            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )

        result = {"train_loss_record": train_loss_record, "train_acc_record": train_acc_record,
                  "val_loss_record": val_loss_record, "val_acc_record": val_acc_record}
        print(result)
        return result

    def conv_matrix(self, val, class_count):
        confusion_matrix = np.zeros((class_count, class_count))
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(val):
                inputs = inputs.to(self.device)
                classes = classes.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix

    def save_model(self, path):
        torch.save(self.model, path)

    def predict_and_attents(self, img):

        img = img.to(self.device)

        self.model = Recorder(self.model)
        preds, attns = self.model(img)
        self.model = self.model.eject()

        return preds, attns

    def eval_metric(self, batch):

        label_array = []
        preds_array = []

        for data, label in batch:
            data = data.to(self.device)
            label = label.to(self.device)

            output = self.model(data)

            label_array = np.append(label_array, label.detach().cpu().numpy())
            preds_array = np.append(
                preds_array, output.argmax(dim=1).detach().cpu().numpy())

        if (len(label_array) == len(preds_array)):
            print("ViT F1 Score: "+str(f1_score(label_array, preds_array, average="macro")))
            print("ViT Accuracy: "+str(accuracy_score(label_array, preds_array)))
