import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

feature_cols = ['high','low','close','open','to','vol']

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y)) #요소간 곱한 것을 모두 더하기 위함임. e.g. [1,2,3]*[1,2,3] -> 근데 굳이 이렇게 해야하나...
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n): # ref_dict[code], ref_dict, n
    '''
    각 stock code의 각 feature 별로 상관관계가 계산된다. 이 때 tmp_res에는 각 주식 code별 상관관계가 저장되며 임시 tmp_res는 results에 저장된다.
     '''
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs): # pos는 feature의 index, x는 feature의 20일 동안의 데이터
            y = ys[pos] # 하나의 feature에 대해서 20일 어치의 데이터를 가져온다.
            tmp_res.append(cal_pccs(x, y, n)) # x: 주식 code, y: features, n: 참조할 과거 시계열 길이 -> pearson 상관관계 값을 구한다.
        result.append(tmp_res) # 각code 별로 진행.
    return np.mean(result, axis=1) # mean of all pccs -> 6개의 feature에 대한 평균 pccs -> 이렇게 하면 안되지 않니... 어떻게 했지 

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes] # 전체 코드에 대해서 각각의 코드에 대한 pccs를 구하고 튜플로서 저장
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all] # 저장된 튜플(x,y,n)이 있으면 그것에 대해서 멀
        output = [o.get() for o in results]
        data = np.stack(output) #  하나의 배열로서 합친다.
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))): # 행은 code, 열은 feature
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)

path1 = "./data/csi300.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
#prev_date_num Indicates the number of days in which stock correlation is calculated
prev_date_num = 20
date_unique=df1['dt'].unique()
stock_trade_data=date_unique.tolist()
stock_trade_data.sort()
stock_num=df1.code.unique().shape[0]
#dt is the last trading day of each month
dt=['2022-11-30','2022-12-30']

df1['dt']=df1['dt'].astype('datetime64')

for i in range(len(dt)): # datetime list가 생성되었는데, 이는 각 달의 마지막 날짜를 의미한다.
    df2 = df1.copy()
    end_data = dt[i]
    start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
    df2 = df2.loc[df2['dt'] <= end_data]
    df2 = df2.loc[df2['dt'] >= start_data]
    code = sorted(list(set(df2['code'].values.tolist())))
    test_tmp = {}
    for j in tqdm(range(len(code))):
        df3 = df2.loc[df2['code'] == code[j]]
        y = df3[feature_cols].values
        if y.T.shape[1] == prev_date_num:
            test_tmp[code[j]] = y.T
    t1 = time.time()
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1) 
    # test_tmp: {code: feature} -> code에 대한 feature를 가지고 있는 dict
    # test_tmp.keys(): code list
    # prev_date_num: 20
    # processes: 1 -> 멀티프로세싱을 사용하지 않는다. 차후 멀티프로세싱 사용 시 변경
    result=result.fillna(0)
    for i in range(0,stock_num):
        result.iloc[i,i]=1 # 대각선은 1로 채워준다. -> Adjacency matrix이기 때문에 대각선은 1로 채워주는 작업 assert
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    result.to_csv("./data/relation/"+str(end_data)+".csv")
    
    '''
    그냥 데이터 가지고 
    '''
