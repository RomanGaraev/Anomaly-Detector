import pandas
import numpy as np
from sklearn import preprocessing
import os

path = "data\Bot\\"

# Получает имена всех .csv файлов в папке по пути "path"
def FindCsv():
    files = []
    for r, d, f in os.walk(path):
        print("Founded files in the folder", path, ":")
        for file in f:
            if '.csv' in file:
                files.append(path + file)
                print(file)
    return files

# Возвращает стандартизированные обучающую, валидационную и тестовую выборки
def LoadSet():
    # Загружаем выборки из .csv таблиц
    files = FindCsv()
    tables = []
    print("\nLoading started...")
    for file in files:
        tables.append(pandas.read_csv(file, sep=","))
        print("File", file, "is added.")

    # Объединяем таблицы
    data_set = pandas.concat(tables, ignore_index=True)
    print("Data set is completely loaded. Total length of samples:", len(data_set))

    # Избавляемся от "плохих" характеристик
    data_set = data_set.drop(['Timestamp', 'Flow Byts/s', 'Flow Pkts/s'], axis=1)
    # Разбиваем и перемешиваем выборки
    index = np.random.permutation(np.arange(len(data_set)))
    train_ind = index[: np.int32(0.70 * len(data_set))]
    valid_ind = index[np.int32(0.70 * len(data_set)) : np.int32(0.85 * len(data_set))]
    test_ind  = index[np.int32(0.85 * len(data_set)) :]

    train_set = preprocessing.scale(data_set.iloc[train_ind, 0:76])
    valid_set = preprocessing.scale(data_set.iloc[valid_ind, 0:76])
    test_set  = preprocessing.scale(data_set.iloc[test_ind,  0:76])

    train_label = data_set.iloc[train_ind, 76]
    valid_label = data_set.iloc[valid_ind, 76]
    test_label  = data_set.iloc[test_ind, 76]
    return train_set, valid_set, test_set, train_label, valid_label, test_label
