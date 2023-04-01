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
        for i in range(len(self.x_train)):
            x = torch.tensor(self.x_train[i])
            y = torch.tensor(self.y_train[i])
            self.samples.append((x, y))

    # def __len__(self):
    #     return len(self.samples)

    # def __getitem__(self, idx):
    #     return self.samples[idx]


def train(model, samples, epochs=4, batch_size=4):
    train_dataset = [sample for sample in train_dataset if sample[1].numel() > 0]
    samples = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(samples):
            model_input = torch.Tensor([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])
            optimizer.zero_grad()
            output = model(model_input)
            loss = loss_function(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')
            loss.backward()
            optimizer.step()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances")
    args = parser.parse_args()
    data = args.featurefile
    print("Reading {}...".format(args.featurefile))
    print(data)
    # file_path = os.path.abspath(data)
    # datafile = pd.read_pickle(file_path)
