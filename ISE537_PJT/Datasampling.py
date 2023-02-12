import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
from tqdm import tqdm
import time
import io

def resampling_byIdenti(data, State):
    data.set_index('DateTime', inplace=True)
    data.rename(columns = {'ï..Identifier':'Identifier'}, inplace=True)
    
    groups = data.groupby("Identifier")
    temp_result = dict(list(groups))
    temp_result_key = list(temp_result.keys())
    temp_result_value = list(temp_result.values())
    
    temp = []
    for i in range(len(temp_result_value)):
        Indentifier_temp = temp_result_key[i]
        temp1 = temp_result_value[i].drop([temp_result_value[0].columns[0], temp_result_value[0].columns[1]], axis=1)
        temp1 = temp1.resample(rule='1H').mean()
        temp1 = temp1.dropna()

        temp1.insert(2, 'Identifier', Indentifier_temp)
        temp1.insert(3, 'State', State)
        temp.append(temp1)
    
    print(len(temp[0]), len(temp[1]), len(temp[2]))
    
    result = pd.DataFrame(temp[0])
    for i in range(1, len(temp)):
        temp_pd = pd.DataFrame(temp[i])
        result = pd.concat([result, temp_pd], axis = 0, ignore_index=False)
    
    return result

def data_change(data):
    #날짜 칼럼을 인덱스로 사용
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['date'] = data['DateTime'].dt.date.astype('str') #날짜정보(YYYY-MM-DD(문자), 다시 문자열로 변환)
    data['month'] = data['DateTime'].dt.month #월정보(1~12)
    data['dow'] = data['DateTime'].dt.weekday #요일정보(월요일=0, 일요일=6) #dow:day of the week
    #data['woy']= data['DateTime'].dt.weekofyear #1년 48주
    data['day'] = data['DateTime'].dt.day #일정보(1~31)
    data['hour'] = data['DateTime'].dt.hour #시간정보(0~23)

    #특별 공휴일 정보(2017, USA 기준)
    #feature: 'holiday' saturday, sunday, holiday
    special_days = ['2017-01-02', '2017-01-16', '2017-05-29', '2017-06-19', '2017-07-04', '2017-09-04', '2017-11-10', '2017-11-23', '2017-12-25']
    #New Year's Day, Martin Luther King, Jr., Washington's Birth day, Memorial Day, Juneteeth Independence Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thangsgiving Day, Christmas Day
    data['holiday'] = data['dow'].isin([5,6]).astype(int) #isin([5,6])_토요일, 일요일 데이터 프레임안에 괄호안의 데이터의 유무(True,False_Bool형태로 나타냄) astype(int) #데이터 타입 int 정수, float 실수
    data.loc[data.date.isin(special_days), 'holiday'] = 1 #data.loc[행, 열] 행, 열에 해당하는 데이터를 가져옴 #data.loc[이름]_문자, data.iloc[행인덱스, 열인덱_숫자
    sinmonth = np.sin(2*np.pi*data['month']/12)
    cosmonth = np.cos(2*np.pi*data['month']/12) 

    sindow = np.sin(2*np.pi*data['dow']/7)
    cosdow = np.cos(2*np.pi*data['dow']/7)

    sindate = np.sin(2*np.pi*data['day']/31)
    cosdate = np.cos(2*np.pi*data['day']/31)

    sinhour = np.sin(2*np.pi*data['hour']/24)
    coshour = np.cos(2*np.pi*data['hour']/24)

    data = data.assign(sinmonth=sinmonth.values, cosmonth=cosmonth.values, sindow=sindow.values, cosdow=cosdow.values, sindate=sindate.values, cosdate=cosdate.values, sinhour=sinhour.values, coshour=coshour.values)
    data.head()

    #칼럼 순서 변경
    col1=data.columns[-17:].to_list()
    col2=data.columns[:-17].to_list()
    new_col = col1+col2
    data=data[new_col]
    
    return data

def interporation(df):
    df['T_ctrl'] = df['T_ctrl'].interpolate(method='linear')
    df['T_stp_cool'] = df['T_stp_cool'].interpolate(method='linear')
    df['T_stp_heat'] = df['T_stp_heat'].interpolate(method='linear')
    df['Humidity'] = df['Humidity'].interpolate(method='linear')
    df['auxHeat1'] = df['auxHeat1'].interpolate(method='linear')
    df['auxHeat2'] = df['auxHeat2'].interpolate(method='linear')
    df['auxHeat3'] = df['auxHeat3'].interpolate(method='linear')
    df['compCool1'] = df['compCool1'].interpolate(method='linear')
    df['compCool2'] = df['compCool2'].interpolate(method='linear')
    df['compHeat1'] = df['compHeat1'].interpolate(method='linear')
    df['compHeat2'] = df['compHeat2'].interpolate(method='linear')
    df['fan'] = df['fan'].interpolate(method='linear')
    df['Thermostat_Temperature'] = df['Thermostat_Temperature'].interpolate(method='linear')
    df['Thermostat_Motion'] = df['Thermostat_Motion'].interpolate(method='linear')    
    df['T_out'] = df['T_out'].interpolate(method='linear')
    df['RH_out'] = df['RH_out'].interpolate(method='linear')
    return df

def data_drop(data):
    data = data.drop(['HvacMode'], axis = 1)
    data = data.drop(['Event'], axis = 1)
    data = data.drop(['Schedule'], axis = 1)
    data = data.drop(['Remote_Sensor_1_Temperature'], axis = 1)
    data = data.drop(['Remote_Sensor_1_Motion'], axis = 1)
    data = data.drop(['Remote_Sensor_2_Temperature'], axis = 1)
    data = data.drop(['Remote_Sensor_2_Motion'], axis = 1)
    data = data.drop(['Remote_Sensor_3_Temperature'], axis = 1)
    data = data.drop(['Remote_Sensor_3_Motion'], axis = 1)
    data = data.drop(['Remote_Sensor_4_Temperature'], axis = 1)
    data = data.drop(['Remote_Sensor_4_Motion'], axis = 1)
    data = data.drop(['Remote_Sensor_5_Temperature'], axis = 1)
    data = data.drop(['Remote_Sensor_5_Motion'], axis = 1)
    return data

def resampling(data):
    #hourly resampling by State and Identifier 
    df_1 = data.groupby('State').get_group("CA")
    df_1_new = resampling_byIdenti(df_1, 'CA')

    df_2 = data.groupby('State').get_group("IL")
    df_2_new = resampling_byIdenti(df_2, 'IL')

    df_3 = data.groupby('State').get_group("NY")
    df_3_new = resampling_byIdenti(df_3, 'NY')

    df_4 = data.groupby('State').get_group("TX")
    df_4_new = resampling_byIdenti(df_4, 'TX')

    monthly_result_temp = pd.concat([df_1_new, df_2_new, df_3_new, df_4_new], axis=0)
    return monthly_result_temp

############JAANUARY############
Jan = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/1_Jan.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Jan = data_drop(Jan)
Jan_temp = resampling(Jan)
Jan_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Jan_temp.csv')

#Date change to month dow day hour holiday
Jan_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Jan_temp.csv', encoding='latin-1')
Jan_result = data_change(Jan_temp1)

############Feb############
Feb = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/2_Feb.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Feb = data_drop(Feb)
Feb_temp = resampling(Feb)
Feb_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Feb_temp.csv')

#Date change to month dow day hour holiday
Feb_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Feb_temp.csv', encoding='latin-1')
Feb_result = data_change(Feb_temp1)

############Mar############
Mar = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/3_Mar.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Mar = data_drop(Mar)
Mar_temp = resampling(Mar)
Mar_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Mar_temp.csv')
Mar_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Mar_temp.csv', encoding='latin-1')
Mar_result = data_change(Mar_temp1)

############Apr############
Apr = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/4_Apr.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Apr = data_drop(Apr)
Apr_temp = resampling(Apr)
Apr_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Apr_temp.csv')
Apr_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Apr_temp.csv', encoding='latin-1')
Apr_result = data_change(Apr_temp1)

############May###########
May = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/5_May.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
May = data_drop(May)
May_temp = resampling(May)
May_temp.to_csv('C:/Users/jkim226/Desktop/New folder/May_temp.csv')
May_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/May_temp.csv', encoding='latin-1')
May_result = data_change(May_temp1)

############Jun############
Jun = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/6_Jun.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Jun = data_drop(Jun)
Jun_temp = resampling(Jun)
Jun_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Jun_temp.csv')
Jun_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Jun_temp.csv', encoding='latin-1')
Jun_result = data_change(Jun_temp1)

############Jul############
Jul = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/7_Jul.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Jul = data_drop(Jul)
Jul_temp = resampling(Jul)
Jul_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Jul_temp.csv')
Jul_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Jul_temp.csv', encoding='latin-1')
Jul_result = data_change(Jul_temp1)

############Aug############
Aug = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/8_Aug.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Aug = data_drop(Aug)
Aug_temp = resampling(Aug)
Aug_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Aug_temp.csv')
Aug_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Aug_temp.csv', encoding='latin-1')
Aug_result = data_change(Aug_temp1)

############Sep############
Sep = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/9_Sep.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Sep = data_drop(Sep)
Sep_temp = resampling(Sep)
Sep_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Sep_temp.csv')
Sep_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Sep_temp.csv', encoding='latin-1')
Sep_result = data_change(Sep_temp1)

############Oct############
Oct = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/10_Oct.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Oct = data_drop(Oct)
Oct_temp = resampling(Oct)
Oct_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Oct_temp.csv')
Oct_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Oct_temp.csv', encoding='latin-1')
Oct_result = data_change(Oct_temp1)

############Nov############
Nov = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/11_Nov.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Nov = data_drop(Nov)
Nov_temp = resampling(Nov)
Nov_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Nov_temp.csv')
Nov_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Nov_temp.csv', encoding='latin-1')
Nov_result = data_change(Nov_temp1)

############Dec############
Dec = pd.read_csv('G:/My Drive/5_Federated Learning/Dataset/12_Dec.csv',
                   encoding='latin-1', parse_dates=['DateTime'])
Dec = data_drop(Dec)
Dec_temp = resampling(Dec)
Dec_temp.to_csv('C:/Users/jkim226/Desktop/New folder/Dec_temp.csv')
Dec_temp1 = pd.read_csv('C:/Users/jkim226/Desktop/New folder/Dec_temp.csv', encoding='latin-1')
Dec_result = data_change(Dec_temp1)

monthly_result = pd.concat([Jan_result, Feb_result, Mar_result, Apr_result, May_result, Jun_result, Jul_result, Aug_result, Sep_result, Oct_result, Nov_result, Dec_result], axis=0)
monthly_result.to_csv('C:/Users/jkim226/Desktop/New folder/monthly_result.csv', index = False)