import pandas as pd
import numpy as np

batch = 241

def get_date(row):
    return row['set_time'][:4]


def get_hour(row):
    return row['set_time'][4:]


def dataProcess(file_path='datasets_final.csv'):
    tourist_data = pd.read_csv(file_path)
    pd.set_option('display.float_format', lambda x: '%d' % x)
    tourist_data = tourist_data.loc[tourist_data['set_time'].notnull(), :]
    tourist_data.sort_values(by='set_time', ascending=True, inplace=True)
    tourist_data.loc[:, 'set_time'] = tourist_data['set_time'].astype(str).str.replace('\\.0', '', regex=True).str[
                                      4:].str[:-2]
    tourist_data.loc[:, 'date'] = tourist_data.apply(get_date, axis=1)
    tourist_data.loc[:, 'hour'] = tourist_data.apply(get_hour, axis=1)

    tourist_data.drop(['id', 'start_port', 'ship'], axis=1, inplace=True)

    group_result = tourist_data.groupby(['date', 'hour'])

    count_result = pd.DataFrame(columns=['date', 'hour', 'people'])
    for name, group in group_result:
        count_result = count_result.append(pd.Series([name[0], name[1], len(group)], index=['date', 'hour', 'people']),
                                           ignore_index=True)

    pre = 0
    count = 0
    exp_mask = []
    i = 0
    for index, row in count_result.iterrows():
        if abs(row['people'] - pre) > 80:
            print(row)
            exp_mask.append(i)
            count += 1
        i += 1
        pre = row['people']
    count_result.drop(labels=exp_mask, axis=0, inplace=True)

    train_column = [i + 1 for i in range(241)]
    train_data = pd.DataFrame(columns=train_column)
    peoples = count_result['people']
    for i in range(len(count_result) - batch + 1):
        train_data = train_data.append(pd.Series(peoples[i:i + batch].tolist(), index=train_column), ignore_index=True)

    label = train_data[241]
    train_data.drop([241], axis=1, inplace=True)
    return np.array(train_data), label
