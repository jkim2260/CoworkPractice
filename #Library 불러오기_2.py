#Library 불러오기

import torch
import torch.nn as nn
import torch.functional as F
import torch.autograd as Variable   
from torch.utils.data import Dataset, DataLoader
from torch import optim

# Python Library 불러오기
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as mlt
import seaborn as sns

#sklearn 라이브러리 불러오기

from sklearn.preprocessing import MinMaxScaler #정규화 과정
from sklearn.model_selection import train_test_split #데이터 분리

#이미지 시각화
import matplotlib.pyplot as plt


#진행사항 파악
from tqdm.auto import tqdm

#경고 끄기
import warnings
warnings.filterwarnings('ignore')


#Check GPU and fix the random seed
#쿠다 연동가능한지 중간중간 코드에 넣는 방법 파악.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#재현성을 결정하는 단계]\
#넘파이, 토치, cuda 랜덤시드 고정(똑같은 값을 비교를 위해 가중치 값이 랜덤으로 변화하는 것을 방지하는 목적) 
#향후 모든텐서를 gpu에 넣어줌, 모델을 gpu에 넣어줌
np.random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


#전처리된 데이터 읽기
df= pd.read_csv('C:/Users/gk571/OneDrive/바탕 화면/FL/df_preprocessed.csv', encoding='utf-8')
df.set_index('datetime', inplace=True) #데이터 프레임의 인덱스를 datetime으로 설정
df.describe()

df.head(24)

#슬라이딩 윈도우 함수 정의
def sliding_winodw(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

df.plot()

#Get the numpy array
train_data = df.iloc[:,:].values

train_data = train_data.astype('float')

#train_data = pd.DataFrame(train_data)

#feature engineering
#전처리 작업
sc = MinMaxScaler()
train_data = sc.fit_transform(train_data)
seq_length = 24

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x, y = sliding_winodw(train_data, seq_length)
#train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=2)
#train_len = int(len(y)*0.7)
#valid_len = len(y)-train_len

x_tensor =torch.tensor(x_train, dtype=torch.float)
x_tensor =x_tensor.to(device)
y_tensor =torch.tensor(y_train, dtype=torch.float)
y_tensor =y_tensor.to(device)

x_tensor_1 = torch.tensor(x_test, dtype=torch.float)
x_tensor_1 =x_tensor_1.to(device)
y_tensor_1 =torch.tensor(y_test, dtype=torch.float)
y_tensor_1 =y_tensor_1.to(device)

#train_x = X_tensor[:train_len]
#train_y = y_tensor[:train_len]
#valid_x = X_tensor[train_len:]
#valid_y = y_tensor[train_len:]

class LSTM(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, num_layers):
         super(LSTM, self).__init__()
         self.num_features = num_features #Feature의 수
         self.num_layers = num_layers #LSTM Layer의 수
         self.input_size = input_size #LSTM의 input size(입력크기로 훈련 데이터의 feature의 수를 사용)
         self.hidden_size = hidden_size #은닉층의 뉴런개수
         self.seq_length = seq_length #시퀀스 길이
         
         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
         self.fc = nn.Linear(hidden_size, num_features)
 
    def forward(self, x):
         h_0 = torch.zeros(self.num_layers, x_tensor.size(0), self.hidden_size).cuda()
         c_0 = torch.zeros(self.num_layers, x_tensor.size(0), self.hidden_size).cuda()
         # Propagate input through LSTM
         _, (h_out, _) = self.lstm(x_tensor, (h_0, c_0))
         h_out = h_out.view(-1, self.hidden_size)
         out = self.fc(h_out)
         return out
    
#Define hyperparameters
num_epochs = 2000
learning_rate = 0.001
    
input_size = 22
hidden_size = 2
#num_layers = 1
num_features = 1
    
lstm=LSTM(num_features, input_size, hidden_size, num_layers)
lstm.cuda()
lstm = lstm.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


#Traing the model
for epoch in range(num_epochs):
    outputs = lstm(x_tensor)
    #obtain the loss function
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        

        
                       
