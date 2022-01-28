import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import os

class BasicDataset(Dataset):

    def __init__(self, dataDir):

        super().__init__()
        self.dataDir = dataDir
        assert os.path.isdir(dataDir), dataDir + "does not exist"
        self.subDirs = glob.glob(dataDir+'/*/')
        
        self.filesPerSubDir = []
        for subDir in self.subDirs:
            files = glob.glob(subDir + "*")
            self.filesPerSubDir.append(len(files))

    def __getitem__(self, index):
        dirIdx = 0
        files = 0
        for i, curDir in enumerate(self.filesPerSubDir):
            files += self.filesPerSubDir[i]
            if files > index:
                break
            dirIdx += 1
        
        curDir = self.subDirs[dirIdx]
        curIdx = index - sum([self.filesPerSubDir[i] for i in range(dirIdx)])
        files = glob.glob(curDir + "*")
        curFile = files[curIdx]

        ckpt = torch.load(curFile)
        x = ckpt['x']
        action_tuple = ckpt['action_tuple']
        real_target = ckpt['real_target']
        x = x.squeeze(0)
        real_target = real_target.unsqueeze(0)
        return x, torch.LongTensor(action_tuple), real_target

    def __len__(self):
        total = sum(self.filesPerSubDir)
        return total


class BasicModel(nn.Module):
    '''
    Basic model that predicts target value for 3^12 possible actions
    given the previous observation and the action.
    '''
    def __init__(self):
        super(BasicModel, self).__init__()
        input_size = 72
        output_size = pow(3, 12)

        self.layers = nn.Sequential(
            nn.Linear(input_size, pow(2, 7)),
            nn.ReLU(),
            nn.BatchNorm1d(pow(2, 7)),
            nn.Linear(pow(2, 7), pow(2, 9)),
            nn.ReLU(),
            nn.BatchNorm1d(pow(2, 9)),
            nn.Linear(pow(2, 9), output_size)
        )
    def forward(self, x):
        x = self.layers(x)
        shape = tuple([-1] + [3 for i in range(12)])
        x = x.reshape(shape)
        return x

def trainModel(dataDir,
               resume_from=None,
               outDir="",
               numEpochs= 1,
               learning_rate = 1e-3,
               batch_size=4,
               num_workers = 6,
               save_every = 1):

    model = BasicModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=learning_rate, params = model.parameters())
    prevEpoch = -1
    trainLosses = []
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        prevEpoch = checkpoint['epoch']

        print("Resuming from checkpoint at, ", resume_from, ", epoch, ", prevEpoch)

        trainLosses = checkpoint['trainLosses']
        model.load_state_dict(checkpoint['state_dict'], strict = False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint

    ds = BasicDataset(dataDir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loss_fn = torch.nn.SmoothL1Loss()

    for epoch in range(prevEpoch + 1, prevEpoch + 1 + numEpochs):
        pbar = tqdm(dl)
        for xs, action_tuples, real_targets in pbar:
            model.train()
            predictions = model(xs.to(device))

            #index using action tuple
            predicted_targets = torch.stack([predictions[i][action_tuples[i].chunk(12, dim =0)] for i in range(batch_size)])
            loss = loss_fn(predicted_targets, real_targets.to(device))
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLosses.append(loss_value)
            pbar.set_postfix({'Epoch' : epoch,
                              'Loss_cur' : loss_value,
                               'Loss_mean': sum(trainLosses[-100:])/(len(trainLosses[-100:])+1e-4)})

            if np.isinf(loss_value):
                print("Inf values in target ", torch.max(torch.isinf(real_targets)))
                print("Inf values in predicted ", torch.max(torch.isinf(predicted_targets)))
            if np.isnan(loss_value):
                print("Nan values in target ", torch.max(torch.isnan(real_targets)))
                print("Nan values in predicted ", torch.max(torch.isnan(predicted_targets)))

        pbar.close()

        if (epoch - prevEpoch) % save_every == 0:
            #save checkpoint for this epoch
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses' : trainLosses,
            }
            torch.save(checkpoint, outDir + "checkpoint" + str(epoch) + '.pt')
