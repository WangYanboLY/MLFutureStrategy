def returns_last_5days(data):
    # 近一个月(22个交易日)的收益率
    data['returns_last_5days'] = data['close'].pct_change(5)
    return data['returns_last_5days']


def weighted_avg_return_2(data):
    """
    计算近x日的加权平均收益率，使用距离权重。
    :param prices: 股票的价格序列 (pandas Series)
    :param x: 考虑的天数
    :return: 近x日的加权平均收益率 (pandas Series)
    """
    # 计算日收益率
    returns = data['close'].pct_change()

    # 根据x生成权重。例如，对于x=9，权重是[9, 8, 7, ... 1]
    weights = list(range(2, 0, -1))

    # 确保权重和为1
    weights = [w/sum(weights) for w in weights]

    # 计算加权平均收益率
    data['weighted_avg_return_2'] = returns.rolling(window=2).apply(lambda ret: (ret * weights).sum(), raw=True)

    return data['weighted_avg_return_2']


def weighted_avg_return_42(data):
    """
    计算近x日的加权平均收益率，使用距离权重。
    :param prices: 股票的价格序列 (pandas Series)
    :param x: 考虑的天数
    :return: 近x日的加权平均收益率 (pandas Series)
    """
    # 计算日收益率
    returns = data['close'].pct_change()

    # 根据x生成权重。例如，对于x=9，权重是[9, 8, 7, ... 1]
    weights = list(range(42, 0, -1))

    # 确保权重和为1
    weights = [w/sum(weights) for w in weights]

    # 计算加权平均收益率
    data['weighted_avg_return_42'] = returns.rolling(window=42).apply(lambda ret: (ret * weights).sum(), raw=True)

    return data['weighted_avg_return_42']


def returns_last_18day(data):
    # 18个交易日收益率
    data['returns_last_18day'] = data['close'].pct_change(18)
    return data['returns_last_18day']


def returns_last_2days(data):
    # 近2天的收益率
    data['returns_last_2days'] = data['close'].pct_change(2)
    return data['returns_last_2days']


def returns_daliy_max_2days(data):
    # 近一个月内的日收益率最大值
    data['returns_real'] = data['returns'].shift()
    data["returns_daliy_max_2days"] = data['returns_real'].rolling(window=2).max()
    return data["returns_daliy_max_2days"]


def returns_daliy_max_7days(data):
    # 近7天内的日收益率最大值
    data['returns_real'] = data['returns'].shift()
    data["returns_daliy_max_7days"] = data['returns_real'].rolling(window=7).max()
    return data["returns_daliy_max_7days"]


def returns_daliy_max_last_20days(data):
    # 近20天内内的日收益率最大值
    data['returns_real'] = data['returns'].shift()
    data["returns_daliy_max_last_20days"] = data['returns_real'].rolling(window=20).max()
    return data["returns_daliy_max_last_20days"]


def sma_weight_con_23_37(data):
    # 容量和
    data['SHORT_SUM'] = data['volume'].rolling(window=23).sum()
    data['LONG_SUM'] = data['volume'].rolling(window=37).sum()

    # 以rolling函数和四指标平均计算生成长短移动平均线
    data['SHORT_WEIGHT'] = data['volume'] / data['SHORT_SUM']
    data['LONG_WEIGHT'] = data['volume'] / data['LONG_SUM']

    data['SHORT_PRICE'] = data['SHORT_WEIGHT'] * data['vwap']
    data['LONG_PRICE'] = data['LONG_WEIGHT'] * data['vwap']

    data["SMA_short"] = data['SHORT_PRICE'].rolling(window=23).sum()
    data["SMA_long"] = data['LONG_PRICE'].rolling(window=37).sum()

    data['sma_weight'] = 0
    data['sma_weight_con_23_37'] = (data['SMA_short'].shift(1) - data['SMA_long'].shift(1))-(data['SMA_short'].shift(2) - data['SMA_long'].shift(2))

    return data['sma_weight_con_23_37']


def sma_weight_con_3_62(data):
    # 容量和
    data['SHORT_SUM'] = data['volume'].rolling(window=3).sum()
    data['LONG_SUM'] = data['volume'].rolling(window=62).sum()

    # 以rolling函数和四指标平均计算生成长短移动平均线
    data['SHORT_WEIGHT'] = data['volume'] / data['SHORT_SUM']
    data['LONG_WEIGHT'] = data['volume'] / data['LONG_SUM']

    data['SHORT_PRICE'] = data['SHORT_WEIGHT'] * data['vwap']
    data['LONG_PRICE'] = data['LONG_WEIGHT'] * data['vwap']

    data["SMA_short"] = data['SHORT_PRICE'].rolling(window=3).sum()
    data["SMA_long"] = data['LONG_PRICE'].rolling(window=62).sum()

    data['sma_weight'] = 0
    data['sma_weight_con_3_62'] = (data['SMA_short'].shift(1) - data['SMA_long'].shift(1))-(data['SMA_short'].shift(2) - data['SMA_long'].shift(2))

    return data['sma_weight_con_3_62']


def flipping_weight_29_69(df):
    df['returns_real'] = df['returns'].shift()
    df['weight'] = df['volume'] / df['volume'].rolling(window=29).sum()
    df['returns_real_weight'] = df['weight'] * df['returns_real']
    df['corr'] = df['returns_real_weight'].rolling(window=69).corr(df['returns_real'])
    df['flipping_weight_29_69'] = df['corr'] * df['returns_real_weight']
    return df['flipping_weight_29_69']


def flipping_weight_4_67(df):
    df['returns_real'] = df['returns'].shift()
    df['weight'] = df['volume'] / df['volume'].rolling(window=4).sum()
    df['returns_real_weight'] = df['weight'] * df['returns_real']
    df['corr'] = df['returns_real_weight'].rolling(window=67).corr(df['returns_real'])
    df['flipping_weight_4_67'] = df['corr'] * df['returns_real_weight']
    return df['flipping_weight_4_67']




