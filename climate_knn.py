# Takes ~ 30 sec to test five stations

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def get_args ():
    import sys
    return sys.argv[1:len(sys.argv)]

# Reads a text file
# Returns list of lists
def read_txt (file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    txt = []
    for line in lines:
        txt.append(line.split())
    return txt

def format(txt):
    import re
    from sklearn.preprocessing import Imputer
    exp = re.compile('[A-Z]')
    rows = []
    for i in range(len(txt)):
        entries = []
        for j in range(3, 27):
            num = re.sub(exp, '', txt[i][j])
            entries.append(int(num))
        rows.append(entries)
    # Impute values per hour (column) based on global mean
    imp = Imputer(missing_values=-9999, strategy='mean', axis=0)
    final = imp.fit_transform(rows)
    return final

def build_features(df):
    df['missing'] = df.groupby(['station'])['value'].shift(-8759)
    df['previous_hr'] = df.groupby(['station'])['value'].shift(1).fillna(df['missing'])
    df['missing2'] = df.groupby(['station'])['value'].shift(-8736)
    df['previous_day'] = df.groupby(['station'])['value'].shift(24).fillna(df['missing2'])
    df['hr_mean'] = df.groupby(['month','day','variable'])['value'].transform(np.mean)
    df['day_mean'] = ((df.groupby(['station','month','day'])['value'].transform(np.cumsum) - df['value']) / (df['variable'].replace({0:np.nan}))).fillna(df['value'])
    return df.as_matrix(columns=df.columns[[7,9,10,11]])

def calculate_mse (targets, hypotheses):
    mse = 0
    for i in range(len(targets)):
        mse += (targets[i] - hypotheses[i])**2
    return mse / len(targets)

test_list = ['USW00023234','USW00014918','USW00012919','USW00013743','USW00025309']
stations = get_args()
txt = read_txt('hly-temp-normal.txt')
test = []
index = []
for i in range(len(txt)):
    if txt[i][0] in stations:
        test.append(txt[i])
    if txt[i][0] in test_list:
        index.append(i)
train = [i for j, i in enumerate(txt) if j not in index]
train_id = pd.DataFrame(train).loc[:,0:2]
train_id.columns = ['station','month','day']
test_id = pd.DataFrame(test).loc[:,0:2]
test_id.columns = ['station','month','day']
train_temp = pd.DataFrame(format(train))
test_temp = pd.DataFrame(format(test))
df_train = pd.concat([train_id.reset_index(drop=True), train_temp], axis=1)
df_test = pd.concat([test_id.reset_index(drop=True), test_temp], axis=1)
df_train_long = pd.melt(df_train, id_vars=['station','month','day'])
df_test_long = pd.melt(df_test, id_vars=['station','month','day'])
final_train = df_train_long.sort_values(by=['station','month','day','variable'])
final_train = final_train.reset_index()
final_test = df_test_long.sort_values(by=['station','month','day','variable'])
final_test = final_test.reset_index()
train_x = build_features(final_train)
train_y = final_train['value']
test_x = build_features(final_test)
test_y = final_test['value']
neighbours = 5     # Default to k=5
algo = KNeighborsRegressor(n_neighbors=neighbours)
algo.fit(train_x, train_y)
hypotheses = algo.predict(test_x)
print 'TEST MSE:', calculate_mse(test_y, hypotheses)
