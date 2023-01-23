#CA_Onebuilindg_preprocess.py


#진행사항 파악
from tqdm.auto import tqdm


# Python Library 불러오기
import os
import numpy as np
import pandas as pd

#데이터 불러오기 
df = pd.read_csv('C:/Practice/ISE_537_CA_OneBuilding_Part_1_raw.csv')
df.head()

#날짜 칼럼을 인덱스로 사용
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date.astype('str') #날짜정보(YYYY-MM-DD(문자), 다시 문자열로 변환)
df['month'] = df['DateTime'].dt.month #월정보(1~12)
df['dow'] = df['DateTime'].dt.weekday #요일정보(월요일=0, 일요일=6) #dow:day of the week
df['day'] = df['DateTime'].dt.day #일정보(1~31)
df['hour'] = df['DateTime'].dt.hour #시간정보(0~23)

#특별 공휴일 정보(2017, USA 기준)
#feature: 'holiday' saturday, sunday, holiday
special_days = ['2017-01-02', '2017-01-16', '2017-05-29', '2017-06-19', '2017-07-04', '2017-09-04', '2017-11-10', '2017-11-23', '2017-12-25']
#New Year's Day, Martin Luther King, Jr., Washington's Birth day, Memorial Day, Juneteeth Independence Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thangsgiving Day, Christmas Day
df['holiday'] = df['dow'].isin([5,6]).astype(int) #isin([5,6])_토요일, 일요일 데이터 프레임안에 괄호안의 데이터의 유무(True,False_Bool형태로 나타냄) astype(int) #데이터 타입 int 정수, float 실수
df.loc[df.date.isin(special_days), 'holiday'] = 1 #df.loc[행, 열] 행, 열에 해당하는 데이터를 가져옴 #df.loc[이름]_문자, df.iloc[행인덱스, 열인덱_숫자

#누락된 데이터 값 확인
df.isnull().sum().sum()
df = df.dropna()

#날짜 데이터를 주기성이 띄는 데이터로 변환 #시간(시간만 일단 고려해도될듯), 요일, 일, 월(시간변수를 사인, 코사인으로 변환)
#시간 관련데이터 cyclical encoding하여 변수추가 후 기존데이터 삭제

sinmonth = np.sin(2*np.pi*df['month']/12) #현재 데이터는 8월만 있음
cosmonth = np.cos(2*np.pi*df['month']/12) 

sindow = np.sin(2*np.pi*df['dow']/7)
cosdow = np.cos(2*np.pi*df['dow']/7)

sindate = np.sin(2*np.pi*df['day']/31)
cosdate = np.cos(2*np.pi*df['day']/31)

sinhour = np.sin(2*np.pi*df['hour']/24)
coshour = np.cos(2*np.pi*df['hour']/24)


df = df.assign(sinmonth=sinmonth.values, cosmonth=cosmonth.values, sindow=sindow.values, cosdow=cosdow.values, sindate=sindate.values, cosdate=cosdate.values, sinhour=sinhour.values, coshour=coshour.values)

df.head()

#칼럼 순서 변경
col1=df.columns[-14:].to_list()
col2=df.columns[:-14].to_list()
new_col = col1+col2
df=df[new_col]


##save the preprocessed data
df.to_csv('C:/Practice/ISE_537_CA_OneBuilding_Part_1.csv', index=False)


#date time 못옮김(직접함)
#Sliding window
#one hot encoding 추가적인 작업 필요
#데이터 샘플링안함(1시간단위로)
#pandas resample 함수 참조