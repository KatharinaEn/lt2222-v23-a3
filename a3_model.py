import os
import sys
import argparse
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import torch
from torch import nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import re 
import os 
import os.path
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle
    
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        after_input_layer = self.input_layer(data)
        output = self.softmax(after_input_layer)
        
        return output

class MyDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            self.x_train, self.y_train, self.x_test, self.y_test = data
            self.train_set = [self.x_train, self.y_train]
            self.test_set = [self.x_test, self.y_test]
            label_encoder = LabelEncoder()
            self.y_train = label_encoder.fit_transform(self.y_train)
            self.y_test = label_encoder.transform(self.y_test)
        self.samples = []

        for i in range(self.x_train.shape[0]):
            x = torch.tensor(self.x_train.getrow(i).toarray()[0])
            y = torch.tensor(self.y_train[i])
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        return self.samples[idx]


def train(model, epochs=4, batch_size=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            model_input = torch.stack([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])
            output = model(model_input)
            loss = loss_function(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model



def test(model, test_set):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_set:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
        accuracy = (np.array(y_pred) == np.array(y_true)).sum().item() / len(y_true)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    cnfm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix: ")
    print(cnfm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--epochs", dest="epochs", type=int, default="4", help="The number of epochs used to train the model.")
    parser.add_argument("--linearity", dest="nonlinearity", type=str, default=None, help="The name of the non linear activation function.")
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    dataset = MyDataset(args.featurefile)
    train_set = TensorDataset(torch.Tensor(dataset.train_set[0]), torch.LongTensor(dataset.y_train))
    test_set = TensorDataset(torch.Tensor(dataset.test_set[0]), torch.LongTensor(dataset.y_test))
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4)
    input_size = train_set[0][0].shape[0]
    output_size = dataset.y_train.shape[1]
    model = Model(input_size, output_size)
    model = train(model, dataset, epochs=args.epochs)

    
    print('Done!')

