import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from vit_pytorch import ViT
import matplotlib.pyplot as plt
from CNN.cnn import CNN
import numpy as np
from lime import lime_image
from torchvision.transforms import transforms
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime


class CNNModel():
    def __init__(self, load=False, path=None, pretrained=False):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if(not load):
            self.model = CNN().to(self.device)
            print("init CNN")
        if(load and pretrained):
            self.pretrained = True
            self.model = torchvision.models.resnet152(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 10)
            self.model = self.model.to(self.device)
            print("pretrained CNN")

        else:
            self.model = torch.load(path, map_location=self.device)
            print("load CNN")

        # self.model.register_full_backward_hook()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)

    def train_and_val(self, train, val, epoch):
        train_loss_record = []
        train_acc_record = []
        val_loss_record = []
        val_acc_record = []
        self.model.train()
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
            print("Current Time =", datetime.now().strftime("%H:%M:%S"))

        result = {"train_loss_record": train_loss_record, "train_acc_record": train_acc_record,
                  "val_loss_record": val_loss_record, "val_acc_record": val_acc_record}
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

    def lime_and_explain(self, pil_img, class_num):

        explainer = lime_image.LimeImageExplainer(random_state=42)
        explanation = explainer.explain_instance(np.array(pil_img),
                                                 self.batch_predict,
                                                 top_labels=10,
                                                 hide_color=1,
                                                 num_samples=1000,
                                                 num_features=100000)

        temp, mask = explanation.get_image_and_mask(
            class_num, positive_only=True, num_features=5, hide_rest=True)

        return temp, mask

    def batch_predict(self, imgs):
        self.model.eval()
        preprocess_transform = self.get_preprocess_transform()

        batch = torch.stack(tuple(preprocess_transform(i)
                            for i in imgs), dim=0)

        batch = batch.to(self.device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def get_preprocess_transform(self):
        transf = transforms.Compose([
            transforms.ToTensor(),
        ])

        return transf

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
            return [f1_score(label_array, preds_array, average="macro"), accuracy_score(label_array, preds_array)]

    def predict(self, img):
        batch = img.to(self.device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
