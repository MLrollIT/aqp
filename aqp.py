import pandas as pd
import numpy as np
import json
import time


slice_cut = 1
dataSave = []
columns = ['YEAR_DATE', 'UNIQUE_CARRIER', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST',
           'DEST_STATE_ABR', 'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARR_DELAY', 'AIR_TIME', 'DISTANCE']
isData = []
pivots = []

wholeNum = 1000000
breakNum = slice_cut*wholeNum

Tree = {}
flag0 = 0.0
msgSend = {}
Sum = {}
Avg = {}
dataSize = 0
Data = ['YEAR_DATE',  'DEP_DELAY', 'TAXI_OUT',
        'TAXI_IN', 'ARR_DELAY', 'AIR_TIME', 'DISTANCE']
Card = ['UNIQUE_CARRIER', 'ORIGIN',
        'ORIGIN_STATE_ABR', 'DEST', 'DEST_STATE_ABR']
batch_size = 30


def aqp_online(data: pd.DataFrame, Q: list) -> list:
    global flag0, start_clock, dataSize, isData, pivots, columns, msgSend, Sum, Avg, breakNum

    results = []
    querys = [json.loads(i) for i in Q]

    for query in querys:
        if query['groupby'] == '_None_':
            if query['result_col'][0][0] == 'avg':
                now_level = f_avg(query['predicate'], query['result_col'][0][1], Avg[query['result_col'][0][1]])
                results.append([[now_level]])
            elif query['result_col'][0][0] == 'sum':
                now_level = f_sum(query['predicate'], query['result_col'][0][1], Sum[query['result_col'][0][1]])
                results.append([[now_level]])
            else:
                now_level = f_count(query['predicate'], wholeNum)
                results.append([[now_level]])
        else:
            for index, name in enumerate(columns):
                if name == query['groupby']:
                    result = []
                    for word in pivots[index]:
                        msg_data = msgSend[name][word]
                        if query['result_col'][1][0] == 'avg':
                            now_level = f_avg(query['predicate'], query['result_col'][1][1], msg_data[query['result_col'][1][1]]['mean'])
                            result.append([word, now_level])
                        elif query['result_col'][1][0] == 'sum':
                            now_level = f_sum(query['predicate'], query['result_col'][1][1], msg_data[query['result_col'][1][1]]['sum'])
                            result.append([word, now_level])
                        else:
                            now_level = f_count(query['predicate'], msg_data['YEAR_DATE']['count'])
                            result.append([word, now_level])
                    results.append(result)

    results = [json.dumps(result, ensure_ascii=False) for result in results]
    return results



def f_sum(strain, target, whole):
    # print(strain)
    _delay = 0
    _dist = 0
    for trace in strain:
        if trace['col'] == 'ARR_DELAY':
            _delay += 1
        elif trace['col'] == 'DEP_DELAY':
            _delay += 1
        elif trace['col'] == 'DISTANCE':
            _dist += 1
        elif trace['col'] == 'AIR_TIME':
            _dist += 1

    for trace in strain:
        batch = 0.0
        ubatch = 0.0
        last_pivot = 0
        ulast_pivot = 0
        for index, name in enumerate(columns):
            if name == trace['col']:
                if isData[index]:
                    for i in pivots[index]:
                        if str(trace['lb']) == '_None_':
                            trace['lb'] = -1
                        if i <= trace['lb']:
                            batch += 1
                            last_pivot = i
                        else:
                            batch += (trace['lb'] -
                                      last_pivot) / (i - last_pivot)
                            break
                    for i in pivots[index]:
                        if trace['ub'] == '_None_':
                            trace['ub'] = 100000000
                        if i <= trace['ub']:
                            ubatch += 1
                            ulast_pivot = i
                        else:
                            ubatch += (trace['ub'] -
                                       ulast_pivot) / (i - ulast_pivot)
                            break
                    whole = whole * (ubatch - batch)/10
                    # if (target == 'ARR_DELAY' and trace['col'] == 'DEP_DELAY') or (target == 'DEP_DELAY' and trace['col'] == 'ARR_DELAY'):
                    #     whole = whole * (ubatch - batch)
                else:
                    batch = msgSend[name][trace['lb']
                                          ][target]['sum']/Sum[target]
                    whole = whole * batch
    return whole


def f_avg(strain, target, whole):
    for trace in strain:
        batch = 0.0
        for index, name in enumerate(columns):
            if (not isData[index]) and name == trace['col']:
                batch = msgSend[name][trace['lb']][target]['mean']
                whole *= batch / Avg[target]

    return whole


def f_count(strain, whole):
    for trace in strain:
        batch = 0.0
        ubatch = 0.0
        last_pivot = 0
        ulast_pivot = 0
        for index, name in enumerate(columns):
            if name == trace['col']:
                if isData[index]:
                    for i in pivots[index]:
                        if str(trace['lb']) == '_None_':
                            trace['lb'] = -1
                        if i <= trace['lb']:
                            batch += 1
                            last_pivot = i
                        else:
                            batch += (trace['lb'] -
                                      last_pivot) / (i - last_pivot)
                            break
                    for i in pivots[index]:
                        if str(trace['ub']) == '_None_':
                            trace['ub'] = 100000000
                        if i <= trace['ub']:
                            ubatch += 1
                            ulast_pivot = i
                        else:
                            ubatch += (trace['ub'] -
                                       ulast_pivot) / (i - ulast_pivot)
                            break
                    whole = whole * (ubatch - batch)/10
                else:
                    batch = msgSend[name][trace['lb']
                                          ]['YEAR_DATE']['count']/breakNum
                    whole = batch * whole
    return whole


def aqp_offline(data: pd.DataFrame, Q: list) -> None:
    # 遍历表中数据，找到各个列的十分位数，并按十分位数切分数据
    data.sample(100000)
    global flag0, start_clock, dataSize, isData, pivots, columns, msgSend, Sum, Avg, breakNum
    start_clock = time.time()
    dataSize = data.shape[0]
    if flag0 == 1:
        return
    else:
        flag0 = 1

    if 1:
        year_date = data['YEAR_DATE'].values
        temp0 = np.sort(year_date)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp0[int(len(temp0) * (i + 1) / batch_size)])
        pivot.append(temp0[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        pivot = list(set(data['UNIQUE_CARRIER'].values))
        isData.append(0)
        pivots.append(pivot)
        # print(pivot)

        pivot = list(set(data['ORIGIN'].values))
        isData.append(0)
        pivots.append(pivot)
        # print(pivot)

        pivot = list(set(data['ORIGIN_STATE_ABR'].values))
        isData.append(0)
        pivots.append(pivot)
        # print(pivot)

        pivot = list(set(data['DEST'].values))
        pivots.append(pivot)
        isData.append(0)
        # print(pivot)

        pivot = list(set(data['DEST_STATE_ABR'].values))
        isData.append(0)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['DEP_DELAY'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['TAXI_OUT'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['TAXI_IN'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['ARR_DELAY'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['AIR_TIME'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

        dep_data = data['DISTANCE'].values
        temp1 = np.sort(dep_data)
        pivot = []
        for i in range(batch_size-1):
            pivot.append(temp1[int(len(temp1) * (i + 1) / batch_size)])
        pivot.append(temp1[-1])
        isData.append(1)
        pivots.append(pivot)
        # print(pivot)

    data = data.sample(frac=slice_cut).reset_index(drop=True)

    # now_time = time.time()
    for i in range(len(columns)):
        # print(i, time.time()-now_time)
        # now_time = time.time()
        if not isData[i]:
            dict_temp = {}
            dict_real = {}
            for j in range(len(pivots[i])):
                dict_temp[pivots[i][j]] = {}
                dict_real[pivots[i][j]] = {}
                for name in Data:
                    dict_temp[pivots[i][j]][name] = []
                    dict_real[pivots[i][j]][name] = {
                        'count': 0, 'sum': 0, 'mean': 0, 'std': 0, 'mid': 0}
            Tree[columns[i]] = dict_temp
            msgSend[columns[i]] = dict_real

    breakNum = data.shape[0]

    for index, row in data.iterrows():
        if index % 10000 == 0:
            if time.time()-start_clock > 164:
                breakNum = index
                # print('break at', breakNum)
                break
        for i in range(len(pivots)):
            if not isData[i]:
                for name in Data:
                    # print(len(Tree[columns[i]][row[i]][name]))
                    Tree[columns[i]][row[i]][name].append(row[name])
    for i, pivot in enumerate(pivots):
        if not isData[i]:
            for p in pivot:
                for name in Data:
                    msgSend[columns[i]][p][name]['count'] = len(
                        Tree[columns[i]][p][name])
                    msgSend[columns[i]][p][name]['sum'] = np.sum(
                        Tree[columns[i]][p][name])
                    msgSend[columns[i]][p][name]['mean'] = np.mean(
                        Tree[columns[i]][p][name])
                    msgSend[columns[i]][p][name]['std'] = np.std(
                        Tree[columns[i]][p][name])
                    msgSend[columns[i]][p][name]['mid'] = np.median(
                        Tree[columns[i]][p][name])

    for column in columns:
        if column in Data:
            Sum[column] = np.sum(data[column][:breakNum].values)
            Avg[column] = np.mean(data[column][:breakNum].values)

    return
