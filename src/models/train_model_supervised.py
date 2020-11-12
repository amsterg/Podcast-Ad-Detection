import argparse
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import librosa
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
from src.data.data_utils import AdPodTorchDataset, plot_confusion_matrix  # noqa

np.random.seed(42)


class SUPERVISED_ADD(nn.Module):
    """
    supervised ad/non-ad classification model

    Attributes
    ----------
    ...

    Methods
    -------
    ...

    """

    def __init__(self,
                 input_shape=(251, 40),
                 load_model=False,
                 epoch=0,
                 device=torch.device('cpu'),
                 loss_=None, mode='train'):
        super(SUPERVISED_ADD, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        self.model_save_string = os.path.join(
            self.__class__.__name__ + '_Epoch_{}.pt')

        self.device = device

        self.lstm = nn.LSTM(self.config['MEL_CHANNELS'],
                            self.config['HIDDEN_SIZE'],
                            self.config['NUM_LAYERS'],
                            batch_first=True,
                            bidirectional=True)
        self.linear1 = nn.Linear(2*self.config['HIDDEN_SIZE'],
                                 self.config['EMBEDDING_SIZE'])
        self.linear2 = nn.Linear(self.config['EMBEDDING_SIZE'], int(
            self.config['EMBEDDING_SIZE']/3))
        self.linear3 = nn.Linear(int(self.config['EMBEDDING_SIZE']/3), 2)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch
        self.loss_ = loss_
        self.opt = None
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        model_log_dir = os.path.join(
            self.config['MODEL_SAVE_DIR'], '{}'.format(self.__class__.__name__))
        run_log_dir = os.path.join(
            self.config['RUNS_DIR'], '{}'.format(self.__class__.__name__))

        if not load_model:
            model_save_dir = os.path.join(os.path.join(
                model_log_dir, "run_{}".format(
                                          len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                          )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

            os.makedirs(model_save_dir, exist_ok=True)

            # self.writer = SummaryWriter(log_dir=os.path.join(
            #                             run_log_dir, "run_{}".format(
            #                                 len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                model_log_dir, "run_{}".format(
                                          len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0))
                                          )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
        self.writer = SummaryWriter(log_dir=os.path.join(
            run_log_dir, "run_{}".format(
                len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))

    def forward(self, frames):

        o, (h, _) = self.lstm(frames)  # lstm out,hidden,
        x = torch.mean(o, dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x

    def loss_fn(self, loss_, outs, labels):

        loss_ = loss_(outs, labels)
        return loss_

    def direct_classification_loss(self, embeds, labels):
        labels = labels.reshape(-1, 1).squeeze()
        return self.ce_loss(embeds, labels)

    def train_loop(self,
                   opt,
                   lr_scheduler,
                   loss_,
                   batch_size=1,
                   gaze_pred=None,
                   cpt=0):

        train_iterator = torch.utils.data.DataLoader(self.dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        self.val_iterator = torch.utils.data.DataLoader(self.dataset_val,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)

        if self.load_model:
            self.load_model_cpt(cpt=cpt)

        for epoch in range(self.epoch, 20000):
            for i, (data, labels) in enumerate(train_iterator):
                print(data.shape)
                data = data.view(
                    -1,
                    self.config['MEL_CHANNELS'],
                    self.config['SLIDING_WIN_SIZE_SUPERVISED'],
                ).transpose(1, 2)
                opt.zero_grad()

                out = self.forward(data)
                loss = self.loss_fn(loss_, out, labels)
                self.loss = loss

                self.loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)

                opt.step()
                self.writer.add_scalar('Loss', self.loss.data.item(), epoch)
                self.writer.add_scalar('Accuracy', self.accuracy(), epoch)

            self.writer.add_scalar('ValLoss', self.val_loss(), epoch)
            self.writer.add_scalar('Accuracy', self.accuracy(), epoch)

            if epoch % 2 == 0:

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    def accuracy(self):
        acc = 0
        ix = 0
        for i, (data, labels) in enumerate(self.val_iterator):
            data = data.view(
                -1,
                self.config['MEL_CHANNELS'],
                self.config['SLIDING_WIN_SIZE_SUPERVISED'],
            ).transpose(1, 2)
            outs = self.forward(data)
            outs = torch.argmax(outs, 1)
            ix += outs.shape[0]
            acc += (outs == labels).sum().item()
            if i == 1:
                break
        return acc/ix

    def val_loss(self):
        with torch.no_grad():
            val_loss = []

            for ix, (datum, labels) in enumerate(self.val_iterator):
                datum = datum.view(
                    -1,
                    self.config['MEL_CHANNELS'],
                    self.config['SLIDING_WIN_SIZE_SUPERVISED'],
                ).transpose(1, 2)
                outs = self.forward(datum)

                loss = self.loss_fn(self.loss_, outs, labels)

                val_loss.append(loss)

                if ix == self.config['VAL_LOSS_COUNT']:
                    break

        return torch.mean(torch.stack(val_loss)).data.item()

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda')):
        self.epoch = int(cpt)

        model_pickle = torch.load(self.model_save_string.format(self.epoch),
                                  map_location=device)
        print(model_pickle.keys())
        self.load_state_dict(model_pickle['model_state_dict'])
        if opt:
            self.opt.load_state_dict(model_pickle['optimizer_state_dict'])
        self.global_step = model_pickle['epoch']
        self.loss = model_pickle['loss']
        print("Loaded Model at epoch {},with loss {}".format(
            self.epoch, self.loss))

    def infer(self, fname, cpt=None):
        aud = preprocess_aud(fname)
        embeds = self.embed(aud, group=True)
        return embeds

    def dataset_metrics(self):
        acc = 0
        ix = 0
        preds_dataset = []
        labels_dataset = []

        for i, (data, labels) in tqdm(enumerate(self.val_iterator)):
            data = data.view(
                -1,
                self.config['MEL_CHANNELS'],
                self.config['SLIDING_WIN_SIZE_SUPERVISED'],
            ).transpose(1, 2)
            outs = self.forward(data)
            outs = torch.argmax(outs, 1)
            ix += outs.shape[0]
            acc += (outs == labels).sum().item()
            preds_dataset.append(outs)
            labels_dataset.append(labels)

            if i == 10:
                break
        preds_dataset = torch.cat(preds_dataset, dim=0).cpu().numpy()
        labels_dataset = torch.cat(labels_dataset, dim=0).cpu().numpy()
        cm = confusion_matrix(labels_dataset, preds_dataset)
        plot_confusion_matrix(preds_dataset, labels_dataset,
                              label_names=['ads', 'non-ads'])
        acc = acc/ix
        print("Accuracy: ", acc)
        print("CM: ", cm)
        print("F1: ", f1_score(labels_dataset, preds_dataset))
        print("Accuracy: ", accuracy_score(labels_dataset, preds_dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="cpu or cuda",
                        default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument("--dataset_train",
                        help="path to train_dataset",
                        required=False, default='')
    parser.add_argument("--dataset_val",
                        help="path to val_dataset",
                        required=False, default='')
    parser.add_argument("--mode",
                        help="train or eval",
                        required=True,
                        choices=['train', 'eval'])
    parser.add_argument(
        "--filedir",
        help="dir with fnames to run similiarity eval,atleast 2, separted by a comma",
        type=str)
    parser.add_argument("--load_model",
                        help="to load previously saved model checkpoint",
                        default=False)
    parser.add_argument(
        "--cpt",
        help="# of the save model cpt to load, only valid if valid_cpt is true"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    model = SUPERVISED_ADD(device=device,
                           loss_=loss_,
                           load_model=args.load_model, mode=args.mode).to(device=device)
    model.dataset_train = AdPodTorchDataset(
        model.config['PODS_TRAIN_DIR'], device
    )
    model.dataset_val = AdPodTorchDataset(
        model.config['PODS_VAL_DIR'], device
    )
    # model.dataset_val = model.dataset_train
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.opt = optimizer
    cpt = args.cpt
    if args.load_model:
        model.load_model_cpt(cpt=cpt, device=device)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None
    if args.mode == 'train':
        model.train_loop(optimizer,
                         lr_scheduler,
                         loss_,
                         batch_size=model.config['BATCH_SIZE'],
                         cpt=cpt)
    elif args.mode == 'eval':
        model.val_iterator = torch.utils.data.DataLoader(model.dataset_val,
                                                         batch_size=4 *
                                                         model.config['BATCH_SIZE'],
                                                         shuffle=True,
                                                         drop_last=True)
        model.dataset_metrics()
