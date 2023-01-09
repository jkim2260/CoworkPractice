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


#데이터 불러오기 
df = pd.read_csv('C:/Users/gk571/OneDrive/바탕 화면/New folder/RTU_3_2_Aug.csv', encoding='utf-8',skiprows=[0,1,2])
df = df.drop(df.index[0]) #첫번째 행 삭제(b.IoT data) 
df.head()

#날짜 칼럼을 인덱스로 사용
df['datetime'] = pd.to_datetime(df['Time Point'])
df['hour'] = df['datetime'].dt.hour #시간정보
df['dow'] = df['datetime'].dt.weekday #요일정보(월요일=0, 일요일=6) #dow:day of the week
df['date'] = df['datetime'].dt.date.astype('str') #날짜정보(YYYY-MM-DD(문자), 다시 문자열로 변환)
df['day'] = df['datetime'].dt.day #일정보
df['month'] = df['datetime'].dt.month #월정보

#특별 공휴일 정보(2022, USA 기준)
#feature: 'holiday' saturday, sunday, holiday
special_days = ['2022-01-01', '2022-01-03', '2022-01-17', '2020-02-21', '2022-05-30', '2022-06-20', '2022-07-04', '2022-09-05', '2022-10-10',  '2022-11-11', '2022-11-24', '2022-12-26']
#New Year's Day, Martin Luther King, Jr., Washington's Birth day, Memorial Day, Juneteeth Independence Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thangsgiving Day, Christmas Day
df['holiday'] = df['dow'].isin([5,6]).astype(int) #isin([5,6])_토요일, 일요일 데이터 프레임안에 괄호안의 데이터의 유무(True,False_Bool형태로 나타냄) astype(int) #데이터 타입 int 정수, float 실수
df.loc[df.date.isin(special_days), 'holiday'] = 1 #df.loc[행, 열] 행, 열에 해당하는 데이터를 가져옴 #df.loc[이름]_문자, df.iloc[행인덱스, 열인덱_숫자


#건물 운영시간(임의로 선정)
def building_hour(row):
    if (row['hour'] >= 7 and row['hour'] <= 21) & (row['holiday'] == 0):
        return 1
    else:
        return 0
 
 #함수를 데이터프레임에 적용하여 'building_hour'열을 만듬
df['building_hour'] =df.apply(building_hour, 1)   

df.tail(20)

#데이터 정보 및 데이터타입 확인
#df.info()

#누락된 데이터 값 확인
df.isnull().sum().sum()
df = df.dropna()

df = df[df.columns.drop(list(df.filter(regex='Time Point')))] #열삭제

#plt.figure(figsize=(16,9))
#sns.lineplot(x='hour', y='RTU_03_Outdoor_Air_Temperature_Active', data=df)
#plt.xlabel('Date')
#plt.ylabel('RTU_03_Outdoor_Air_Temperature_Active[°F]')

#날짜 데이터를 주기성이 띄는 데이터로 변환 #시간(시간만 일단 고려해도될듯), 요일, 일, 월(시간변수를 사인, 코사인으로 변환)
#시간 관련데이터 cyclical encoding하여 변수추가 후 기존데이터 삭제

sinhour = np.sin(2*np.pi*df['hour']/24)
coshour = np.cos(2*np.pi*df['hour']/24)

sindow = np.sin(2*np.pi*df['dow']/7)
cosdow = np.cos(2*np.pi*df['dow']/7)

sindate = np.sin(2*np.pi*df['day']/31)
cosdate = np.cos(2*np.pi*df['day']/31)

#sinmonth = np.sin(2*np.pi*df['month']/12) #현재 데이터는 8월만 있음
#cosmonth = np.cos(2*np.pi*df['month']/12) 

df = df.assign(sinhour=sinhour.values, coshour=coshour.values, sindow=sindow.values, cosdow=cosdow.values, sindate=sindate.values, cosdate=cosdate.values)

df.head()

df.info()


#칼럼 순서 변경
col1=df.columns[-14:].to_list()
col2=df.columns[:-8].to_list()
new_col = col1+col2
df=df[new_col]

df. head()

##save the preprocessed data
df.to_csv('./df_preprocessed.csv', index=False)


        
                       