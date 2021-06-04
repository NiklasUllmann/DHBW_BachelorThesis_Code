import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class CNN(pl.LightningModule):


    def __init__(self):
        super().__init__()

    # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(320 * 320 * 3, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

    # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):

        # extracting input and output from the batch
        x, labels = batch

        # doing a forward pass
        pred = self.forward(x)

        # calculating the loss
        loss = F.nll_loss(pred, labels)

        # logs
        logs = {"train_loss": loss}

        output = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for logging purposes
            "log": logs,
        }
        self.log('my_loss', loss, prog_bar=True)

        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('my_loss', loss, prog_bar=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': log}
