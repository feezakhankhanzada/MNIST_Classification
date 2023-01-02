import torch
import torch.nn as nn
from torch.autograd import Variable
from Model import CNN
import config
from Dataset import test_dataloader

model = CNN()

def fit(train_loader):

    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(config.EPOCHS):
        correct = 0
        for batch_idx , (X_batch , y_batch) in enumerate(train_loader):
            vX_batch = Variable(X_batch).float()
            vy_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(vX_batch)
            loss = error(output , vy_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data , 1)[1]
            correct += (predicted == vy_batch).sum()

            if batch_idx % 50 == 0:
                print("Epoch: " , epoch , "[" , batch_idx*len(X_batch) , "/" , \
                len(train_loader.dataset) , "(" , round(batch_idx/len(train_loader.dataset) * 100 , 2) ,\
                ")" , "%\tLoss: " , loss.data , "\tAccuracy: " , round(float(correct*100) / float(config.BATCH_SIZE * (batch_idx + 1)) , 2))

def evaluate():
    correct = 0
    for test_imgs , test_labels in test_dataloader:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output , 1)[1]
        print(correct , predicted , test_labels)
        if (predicted == test_labels).sum():
            correct += 1

    print("Test Accuracy: " , round(float(correct) / len(test_dataloader) * config.BATCH_SIZE , 2) , "%")