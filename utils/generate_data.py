import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable

feature_cols = ['open','high','low','close','to','vol']

path1 = "./data/csi300.pkl"

df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
relation = os.listdir('./data/relation/')
relation = sorted(relation) #stock correalation matrix를 담은 걸 가져옴
date_unique=df1['dt'].unique()
#stock_trade_data는 주식이 거래된 내역을 의미함. 
stock_trade_data=date_unique.tolist() #리스트로 만들고 날짜 리스트를 정렬
stock_trade_data.sort()

df1['dt']=df1['dt'].astype('datetime64') # datetime으로 바꾸고

def fun(relation_dt, start_dt_month, end_dt_month,df1):
    prev_date_num = 20
    adj_all = pd.read_csv('./data/relation/'+relation_dt+'.csv', index_col=0)
    adj_stock_set = list(adj_all.index)
    
    '''
    Positive 관계로서 연결된 Adjacency matrix과 Negative 관계로서 연결된 Adjacency matrix를 각각 만들어 준다
    '''

    pos_g = nx.Graph(adj_all > 0.1) # 0.1보다 큰 것만 connect. 즉 다 '연결'된 것으로 간주. 이 때 GAT로서 알아서 weight를 조절해줄 것이다.
    pos_adj = nx.adjacency_matrix(pos_g).toarray() # Adjacency matrix로 바꿔줌
    pos_adj = pos_adj - np.diag(np.diag(pos_adj)) # 대각선은 0으로 만들어줌
    pos_adj = torch.from_numpy(pos_adj).type(torch.float32) # tensor로 바꿔줌
    
    '''
    Positive 관계로서 연결된 Adjacency matrix를 만들어 준다.
    '''

    neg_g = nx.Graph(adj_all < -0.1) # 상관관계가 -0.1 미만인 것들에 대해서도 testing.
    neg_adj = nx.adjacency_matrix(neg_g) #Adjacnecy matrix로 바꿔주기
    neg_adj.data = np.ones(neg_adj.data.shape) # 1로 바꿔주기 -> threshold 통화 후 0과 1로 바꿔짐
    neg_adj = neg_adj.toarray() # array로 바꿔주기
    neg_adj = neg_adj - np.diag(np.diag(neg_adj)) # 대각선은 0으로 만들어주기
    neg_adj = torch.from_numpy(neg_adj).type(torch.float32) # tensor로 바꿔주기
    
    print('neg_adj over')
    print(neg_adj.shape)
    # dts -> stock이 거래된 데이터(start~end 사이에서 거래된 날짜들만 가져옴)
    dts = stock_trade_data[stock_trade_data.index(start_dt_month):stock_trade_data.index(end_dt_month)+1]
    print(dts)
    
    for i in tqdm(range(len(dts))):
        end_data=dts[i] # 가장 마지막 날짜
        start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)] # 처음 시작하는 날짜
        df2 = df1.loc[df1['dt'] <= end_data]
        df2 = df2.loc[df2['dt'] >= start_data]
        code = adj_stock_set # 
        feature_all = []
        mask = []
        labels = []
        day_last_code = []
        for j in range(len(code)):
            df3 = df2.loc[df2['code'] == code[j]] # code[j]에 해당하는 데이터만 가져옴. 즉 한 종목에 대해서 모든 date에 대한 데이터가 존재하는지 확인
            y = df3[feature_cols].values # value 가져와서
            if y.T.shape[1] == prev_date_num: # 만약 20일치 데이터가 다 존재한다면 -> 없으면 그 경우 ㅔ대해서는 label을 구하지 않는다.
                one = [] # 이건 뭘 위한 거지?
                feature_all.append(y) # feature_all에 추가
                mask.append(True) #  mask에 True 추가
                label = df3.loc[df3['dt'] == end_data]['label'].values # label은 end_data에 해당하는 label을 가져옴
                labels.append(label[0]) # labels에 추가
                one.append(code[j]) # one에 code[j] 추가
                one.append(end_data) # # one에 end_data 추가
                day_last_code.append(one)# day_last_code에 one 추가 -> 들어갈 떄마다 주식별로 for문 진행하면서 종목과 거래 마지막 날짜 저장
                
        #for 문 다 지나면 모든 종목에 대해서 20일치 데이터가 다 존재하는지 확인하고, 존재한다면 feature_all, mask, labels, day_last_code에 추가해줌
        feature_all = np.array(feature_all)
        features = torch.from_numpy(feature_all).type(torch.float32)
        #mask = [True]*len(labels) # 이건 뭐여 assert 문이어야하는 거 아닌가
        assert len(mask) == len(labels)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        result = {'pos_adj': pos_adj, 'neg_adj': neg_adj,  'features': features,
                  'labels': labels, 'mask': mask} # Varibale은 그냥 Tensor임! 최신 버전의 Pytorch에서는 없어짐
        with open('./data/data_train_predict/'+end_data+'.pkl', 'wb') as f:
            pickle.dump(result, f) #pickle 파일을 작성
        df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code) #csv 파일로는 correlation을 저장
        df.to_csv('./data/daily_stock/'+end_data+'.csv', header=True, index=False, encoding='utf_8_sig')
        
#The first parameter and third parameters indicate the last trading day of each month, and the second parameter indicates the first trading day of each month.

fun('2022-11-30','2022-11-01','2022-11-30',df1)
fun('2022-12-30','2022-12-01','2022-12-30',df1)