from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from vit_pytorch import ViT


class ViTModel():
    def __init__(self):
        print("init")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ViT(
        image_size=320,
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
# optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
# scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    def train(self, train, val, epoch):
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

            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
