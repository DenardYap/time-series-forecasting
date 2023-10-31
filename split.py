import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/electric_clean.txt", sep=';')
# print("Length of dataset", len(df))
# print(df.head())

zero_count = (df == 0.0).sum()
print(zero_count)