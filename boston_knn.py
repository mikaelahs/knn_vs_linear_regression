from sklearn.neighbors import KNeighborsRegressor

def read_csv (file_name):
    import csv
    data = []
    targets = []
    with open(file_name, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        # Skip the first row. There may be a better way of doing this.
        header = True
        for row in lines:
            if header:
                header = False
            else:
                data.append(map(float, row[:-1]))
                targets.append(float(row[-1]))
    return data, targets

def calculate_mse (targets, hypotheses):
    mse = 0
    for i in range(len(targets)):
        mse += (targets[i] - hypotheses[i])**2
    return mse / len(targets)

train = []
test = []
predictors = read_csv('boston.csv')[0]
response = read_csv('boston.csv')[1]
train.append(predictors[:len(predictors) - 50])
train.append(response[:len(response) - 50])
test.append(predictors[len(predictors) - 50:])
test.append(response[len(response) - 50:])
train_x = train[0]
train_y = train[1]
test_x = test[0]
test_y = test[1]
neighbours = 3     # Default to k=3
algo = KNeighborsRegressor(n_neighbors=neighbours)
algo.fit(train_x, train_y)
hypotheses = algo.predict(test_x)
print 'TEST MSE:', calculate_mse(test_y, hypotheses)
