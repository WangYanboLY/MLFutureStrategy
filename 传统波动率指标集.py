# 流动性因子
def high_low_5days(data):
    # 近5天内的最高价与最低价之比
    data['high_low_5days'] = data['high'].rolling(window=5).max() / data['low'].rolling(window=5).min()
    return data['high_low_5days']


def high_low_17days(data):
    # 近17天内的最高价与最低价之比
    data['high_low_17days'] = data['high'].rolling(window=17).max() / data['low'].rolling(window=17).min()
    return data['high_low_17days']


def volume_std_11days(data):
    # 近11天内的成交量标准差
    data["volume_std_11days"] = data['volume'].rolling(window=11).std()
    return data["volume_std_11days"]


def volume_std_2days(data):
    # 近2天内的成交量标准差
    data["volume_std_2days"] = data['volume'].rolling(window=2).std()
    return data["volume_std_2days"]


def volume_std_21days(data):
    # 近一个月内的成交量标准差
    data["volume_std_21days"] = data['volume'].rolling(window=21).std()
    return data["volume_std_21days"]


def returns_std_3days(data):
    # 近一个月内的日收益率标准差
    data['real_returns'] = data['close'].pct_change()
    data['returns_std_3days'] = data['real_returns'].rolling(window=3).std()
    return data['returns_std_3days']


def returns_std_27days(data):
    # 近一个月内的日收益率标准差
    data['real_returns'] = data['close'].pct_change()
    data['returns_std_27days'] = data['real_returns'].rolling(window=27).std()
    return data['returns_std_27days']
