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
import csv
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import scipy.sparse as sp
import pickle


file_path = './data/enron_sample'


def loaddata(directory):
    mailtext = [] # X # mails
    authors_list = [] # y # authors
    for author in os.listdir(directory):
        author_path = os.path.join(directory, author)
        if os.path.isdir(author_path):
            for file_name in os.listdir(author_path):
                file_path = os.path.join(author_path, file_name)
                with open(file_path, encoding='latin-1') as f:
                    for line in f.readlines():   
                        if line.startswith('Message-ID') or line.startswith('Date') or line.startswith('Sent:') or line.startswith('To:') or line.startswith('Subject:') or line.startswith('\n') or line.startswith('-----Original Message----- '):
                            continue
                        elif line.endswith('\n') or line.endswith('*'):
                            header_pattern = r"(?s)(.*?\ne:\s*[^\n]+)"
                            line = re.sub(header_pattern, "", line)
                            mailtext.append(line)
                            authors_list.append(author) 

    return mailtext, authors_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir)) # = ./data/enron_sample

    mailtext, authors_list =loaddata(args.inputdir)
 

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize)) 

    matrix = vectorizer.fit_transform(mailtext)
    print(matrix.toarray())


    X = matrix
    y = authors_list
    

    x_train, x_test, y_train, y_test = train_test_split(matrix, authors_list, test_size=args.testsize/100, shuffle=True, max_features=args.dims)

    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)


    print("Writing to {}...".format(args.outputfile))

    data = [x_train, y_train, x_test, y_test]
    with open(args.outputfile, 'wb') as file:
        pickle.dump(data, file)
        print("Saved the data")

    print("Done!")
    
