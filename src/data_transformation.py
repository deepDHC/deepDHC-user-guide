from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

def power_transform(main_data, train_data, test_data):
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
    power_transformer.fit(train_data)
    return power_transformer.transform(main_data), \
           power_transformer.transform(train_data), \
           power_transformer.transform(test_data), \
           power_transformer


def reverse_power_transform(data, power_transformer):
    return power_transformer.inverse_transform(data)


def standardization(main_data, train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(main_data), \
           scaler.transform(train_data), \
           scaler.transform(test_data), \
           scaler


def reverse_standardization(data, scaler):
    return scaler.inverse_transform(data)


def normalization(main_data, train_data, test_data):
    transformer = MinMaxScaler()
    transformer.fit(train_data)
    return transformer.transform(main_data), \
           transformer.transform(train_data), \
           transformer.transform(test_data), \
           transformer


def reverse_normalization(data, transformer):
    return transformer.inverse_transform(data)


def transform_data(data, power_transformer, standardizer, normalizer):
    if power_transformer:
        data = power_transformer.transform(data)
    if standardizer:
        data = standardizer.transform(data)
    if normalizer:
        data = normalizer.transform(data)
    return data
