import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# pytorch-lightning
from pytorch_lightning import LightningModule


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes, transposed_batch[4]

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class RegTrainer(LightningModule):
    def __init__(self, args):
        super(RegTrainer, self).__init__()
        self.args = args
        self.model = vgg19()
        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   )
        self.loss = Bay_Loss(args.use_background)

    def forward(self, inputs):
        return self.model(inputs)

    def prepare_data(self):
        self.downsample_ratio = self.args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(self.args.data_dir, x),
                                  self.args.crop_size,
                                  self.args.downsample_ratio,
                                  self.args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(self.args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=self.args.num_workers,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = MultiStepLR(self.optimizer, milestones=[600])
        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        inputs, points, targets, st_sizes, cood = batch
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p for p in points]
        targets = [t for t in targets]
        outputs = self(inputs)
        N = inputs.size(0)
        prob_list = self.post_prob(cood[0], points, st_sizes)
        log['train/loss'] = loss = self.loss(prob_list, targets, outputs)
        pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
        res = pre_count - gd_count
        mse = np.mean(res * res)
        mae = np.mean(abs(res))
        log['train/mse'] = mse
        log['train/mae'] = mae

        return {'loss': loss,
                'progress_bar': {'train/mse': loss, 'train/mae': mae},
                'log': log
                }

    def validation_step(self, batch, batch_nb):
        log = {}
        epoch_res = []
        inputs, count, name = batch
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        outputs = self(inputs)
        res = count[0].item() - torch.sum(outputs).item()
        epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        log['val_mse'] = torch.tensor(mse)
        log['val_mae'] = torch.tensor(mae)
        return log

    def validation_epoch_end(self, outputs):
        mean_val_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        mean_val_mae = torch.stack([x['val_mae'] for x in outputs]).mean()

        return {'progress_bar': {'mean_val_mse': mean_val_mse,
                                 'mean_val_mae': mean_val_mae},
                'log': {
                        'val/mean_val_mse': mean_val_mse,
                        'val/mean_val_mae': mean_val_mae,
                        }
               }
