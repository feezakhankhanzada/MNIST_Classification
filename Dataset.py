import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch

import config

train = pd.read_csv('/home/feeza/CNN/digit-recognizer/train.csv')
test = pd.read_csv('/home/feeza/CNN/digit-recognizer/test.csv')

X = train.iloc[: , 1 : ].values
y = train.iloc[: , 0].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)

X_train = torch.from_numpy(X_train).type(torch.LongTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

X_test = torch.from_numpy(X_test).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

X_train = X_train.view(-1 , 1 , 28 , 28).float()
X_test = X_test.view(-1 , 1 , 28 , 28).float()

train_dataset = torch.utils.data.TensorDataset(X_train , y_train)
test_dataset = torch.utils.data.TensorDataset(X_test , y_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset , batch_size = config.BATCH_SIZE , shuffle = False)
test_dataloader = torch.utils.data.DataLoader(test_dataset , batch_size = config.BATCH_SIZE , shuffle = False)


