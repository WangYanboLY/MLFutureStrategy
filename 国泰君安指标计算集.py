# 来自国泰君安证券研报《基于短周期量价特征的多因子选股体系》
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# 这里首先有四个函数，用于进行在指标计算中常用的功能


def rf(x):
    # 该函数用于计算一个值在过往一段时间中所有值中的排位，即某期货品种与历史上同品种的值中的排位
    # 该函数将用来替代在股票多因子体系中常用的全市场排位函数
    # 调用该函数的示例语句
    # data['rank_abs_diff'] = data['abs_diff'].rolling(window=22, min_periods=22).apply(rf, raw=True)
    return (x[-1] <= x).sum()


def custom_sma(series, n, m):
    # 该函数用于计算给定时间序列的加权移动平均值
    span_value = n / m
    return series.ewm(span=span_value, adjust=False).mean()


def decay_linear(series, d):
    # 该函数用于计算给定时间序列的线性衰减加权移动平均值
    """
    计算DECAYLINEAR
    Args:
    - series: 输入的序列
    - d: 窗口大小

    Returns:
    - DECAYLINEAR值
    """
    # 创建权重从d递减到1
    weights = np.arange(d, 0, -1)

    # 将权重归一化，使它们的和为1
    weights = weights / weights.sum()

    # 计算线性衰减的加权平均
    result = series.rolling(window=d).apply(lambda x: np.dot(x, weights), raw=True)
    return result


def wma(series, n):
    """
    计算WMA (加权移动平均)

    Args:
    - series: pd.Series，需要计算WMA的数据序列
    - n: int，考虑的时期数

    Returns:
    - pd.Series，WMA值
    """
    weights = [0.9 * i for i in range(1, n + 1)]
    return series.rolling(window=n).apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)


def gj_001(data):
    # 排位函数
    # 因子意义：在过去一段时间中交易量的排位与日收益率排位的相关性
    # 因子公式(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    # 计算DELTA(LOG(VOLUME), 1)
    # Compute DELTA(LOG(VOLUME), 1)
    # 这个因子的直观解释是：它试图捕捉交易量变化与价格变动之间的关系。
    # 特别地，这个因子关心的是，如果交易量发生了大的变化（无论是增加还是减少），那么价格是否也会发生相应的变动。
    data['volume_log'] = np.log(data['volume'])
    data['volume_log_1'] = data['volume_log'].diff()
    data['(close-open)/open'] = (data['close'] - data['open']) / data['open']
    data['volume_log_1_rank'] = data['volume_log_1'].rolling(window=5, min_periods=5).\
        apply(rf, raw=True)
    data['(close-open)/open_rank'] = data['(close-open)/open'].rolling(window=5, min_periods=5). \
        apply(rf, raw=True)
    data['gj_001'] = data['volume_log_1_rank'].rolling(window=6).corr(data['(close-open)/open_rank'])

    return data['gj_001']


def gj_002(data):
    # 因子公式-1 * delta((((close-low)-(high-close))/((high-low)),1))
    # 计算加减除
    data['gj_002'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])

    # 计算delta
    data['gj_002'] = -data['gj_002'].diff(periods=1)

    return data['gj_002']


def gj_003(data):
    # alpha_003:SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    # 如果当天的收盘价等于前一天的收盘价，则累加值为0。
    # 如果当天的收盘价不等于前一天的收盘价：
    # 如果当天的收盘价大于前一天的收盘价，累加值为当天的收盘价减去当天的最低价与前一天的收盘价的较小值。
    # 如果当天的收盘价小于前一天的收盘价，累加值为当天的收盘价减去当天的最高价与前一天的收盘价的较大值。
    # 最后，对于这个累加值序列，计算过去6个交易日内的累加和。

    # Calculate the previous day's close price
    delay = data['close'].shift()

    # 产生条件函数
    condition1 = data['close'] > delay
    condition2 = data['close'] < delay

    # 根据条件创建系列列
    data['series'] = np.where(condition1, data['close'] - np.minimum(data['low'], delay),
                              np.where(condition2, data['close'] - np.maximum(data['high'], delay), 0))

    # 计算因子值
    data['gj_003'] = data['series'].rolling(window=6).sum()

    return data['gj_003']


def gj_004(data):
    # 输出离散((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <
    # ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /
    # MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        """
        Compute the alpha feature based on the given formula.

        :param data: DataFrame containing 'close' and 'volume' columns.
        :return: A Series representing the alpha feature for each row in data.
        """

        # Calculate rolling statistics
        sum_close_8 = data['close'].rolling(window=8).sum()
        mean_close_8 = sum_close_8 / 8
        std_close_8 = data['close'].rolling(window=8).std()

        sum_close_2 = data['close'].rolling(window=2).sum()
        mean_close_2 = sum_close_2 / 2

        mean_volume_20 = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'] / mean_volume_20

        # Apply the conditions from the formula
        condition1 = (mean_close_8 + std_close_8) < mean_close_2
        condition2 = mean_close_2 < (mean_close_8 - std_close_8)
        condition3 = volume_ratio >= 1

        # Compute alpha values
        data['gj_004'] = -1  # default
        data['gj_004'].loc[condition1] = -1
        data['gj_004'].loc[~condition1 & condition2] = 1
        data['gj_004'].loc[~condition1 & ~condition2 & condition3] = 1

        return data['gj_004']


def gj_005(data):
    # 因子公式(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    # TSRANK(VOLUME, 5)：计算过去5天内的成交量的时间序列排名。这意味着它会为最近的5天的成交量排序，并为每天赋予一个排名。
    # TSRANK(HIGH, 5)：计算过去5天内的最高价的时间序列排名。与上面的类似，它会为最近5天的最高价排序，并为每天赋予一个排名。
    # CORR(..., ...,5)：计算两个时间序列排名（上述的成交量和最高价的时间序列排名）在过去5天内的相关性。这将给我们一个关于两者是如何随时间变化而变化的信息。
    # TSMAX(..., 3)：从上面计算得出的5天相关性中，找出3天内的最大值。
    # -1 * ...：取上述最大值的负数。如果原始的值是正的，这意味着在过去的3天内，成交量的增长和最高价的增长之间有一个强烈的正相关。通过取负数，这个关系被反转。
    # 对于每一天，计算其及其之前4天（总计5天）的volume和high的排名
    data['volume_tsrank'] = data['volume'].rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                                                                   raw=False)
    data['high_tsrank'] = data['high'].rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                                                               raw=False)

    # 计算两个时间序列排名的5天相关性
    data['correlation_5'] = data['volume_tsrank'].rolling(window=10).corr(data['high_tsrank'])

    # 从上述的相关性中，找出3天内的最大值
    data['gj_005'] = -data['correlation_5'].rolling(window=3).max()

    return data['gj_005']


def gj_006(data):
    # 排位函数 输出离散

    # 因子公式：(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    # ((OPEN * 0.85) + (HIGH * 0.15))：这是对开盘价（OPEN）和最高价（HIGH）的一个加权平均。这里，开盘价的权重为85 %，而最高价的权重为15 %。
    # DELTA(..., 4)：计算上述加权平均与4天前的加权平均的差值（即变化量）。
    # SIGN(...)：对这个差值取其符号。如果差值为正，结果为 + 1；如果差值为0，结果为0；如果差值为负，结果为 - 1。
    # RANK(...)：根据上述符号函数的结果对数据进行排名。此处可能涉及对整个数据集的排序，或者是在特定分组内的排序（例如按股票代码）。
    # ... * -1：取上述排名的负数。
    # 计算加权平均
    data['weighted_avg'] = (data['open'] * 0.85) + (data['high'] * 0.15)

    # 计算4天的差值
    data['delta_4'] = data['weighted_avg'].diff(4)

    # 取差值的符号
    data['sign'] = data['delta_4'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 对结果进行排名
    data['rank'] = data['sign'].rolling(window=5, min_periods=5). \
        apply(rf, raw=True)

    # 取排名的负数
    data['gj_006'] = -1 * data['rank']

    return data['gj_006']


def gj_007(data):
    # 因子公式：((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    # VWAP - CLOSE: 这计算了平均价（VWAP）与收盘价（CLOSE）之间的差值。
    # MAX((VWAP - CLOSE), 3): 在过去的3天内找到上述差值的最大值。
    # MIN((VWAP - CLOSE), 3): 在过去的3天内找到上述差值的最小值。
    # RANK(...): 对数据进行排名。对于前两个RANK函数，它们对于过去3天内的差值的最大值和最小值分别进行了排名。对于第三个RANK函数，它对VOLUME的变化量进行了排名。
    # DELTA(VOLUME, 3): 计算成交量（VOLUME）与3天前的成交量之间的差值。
    # 计算VWAP与CLOSE的差值的3天最大值和最小值
    data['diff_vwap_close'] = data['vwap'] - data['close']
    data['max_3'] = data['diff_vwap_close'].rolling(window=3).max()
    data['min_3'] = data['diff_vwap_close'].rolling(window=3).min()

    # 对上述两个结果进行排名

    data['rank_max'] = data['max_3'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    data['rank_min'] = data['min_3'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # 计算VOLUME的3天变化量并进行排名
    data['delta_volume_3'] = data['volume'].diff(3)
    data['rank_delta_volume'] = data['delta_volume_3'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # 对三个RANK的结果进行乘法操作
    data['gj_007'] = (data['rank_max'] + data['rank_min']) * data['rank_delta_volume']

    return data['gj_007']


def gj_008(data):
    # 因子公式：RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    # (HIGH + LOW) / 2: 这计算了当天的最高价（HIGH）与最低价（LOW）的平均值，通常称为当天的中间价。
    # ((... * 0.2) + (VWAP * 0.8)): 这是对上述中间价与成交量加权平均价（VWAP）的一个加权平均。其中，中间价的权重为20%，而VWAP的权重为80%。
    # DELTA(..., 4): 这计算了上述加权平均价与4天前的加权平均价之间的差值。
    # ... * -1: 取上述差值的负数。
    # RANK(...): 对上述结果进行排名。
    # 计算中间价和其与VWAP的加权平均
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['weighted_avg'] = (data['mid_price'] * 0.2) + (data['vwap'] * 0.8)

    # 计算加权平均价的4天变化量并取其负数
    data['delta_4'] = -1 * data['weighted_avg'].diff(4)

    # 对上述结果进行排名
    data['gj_008'] = data['delta_4'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    return data['gj_008']


def gj_009(data):
    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    # 1.(HIGH + LOW) / 2: 这计算了当天的最高价（HIGH）和最低价（LOW)的平均值，通常称为当天的中间价。
    # 2.DELAY(HIGH, 1)和DELAY(LOW1): 分别计算了前一天的最高价和最低价。
    # 3.(DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2: 这计算了前一天的中间价。
    # 4.(HIGH + LOW) / 2 -..:这计算了当天的中间价与前一天的中间价之间的差值。
    # 5.(HIGH - LOW) / VOLUME: 这计算了当天的最高价与最低价之间的区间(即振幅）相对于当天的成交量的比例。
    # 6.SMA(...7, 2):计算上述两个结果相乘后的7天简单移动平均，其中权重因子为2。
    # SMA的公式为: SMA = (今天的值×N＋昨天的SMA ×(M-N)) / M，其中M为周期数(在这里为7)，N为权重因子(在这里为2)。
    # 计算当天和前一天的中间价
    data['mid_price_today'] = (data['high'] + data['low']) / 2

    # 计算中间价的差值
    data['mid_diff'] = data['mid_price_today'].diff()

    # 计算振幅相对于成交量的比例
    data['amplitude_ratio'] = (data['high'] - data['low']) / data['volume']
    # 初始化sma_feature
    data['sma_feature'] = 0
    # 计算SMA
    m = 7
    n = 2
    data['sma_feature'] = (data['mid_diff'] * data['amplitude_ratio'] * n + data['sma_feature'].shift(1) * (
                m - n)) / m
    data['gj_009'] = data['sma_feature'].rolling(window=7).mean()
    return data['gj_009']


def gj_010(data):
    # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE) ^ 2), 5))
    # RET < 0: 这是一个条件判断，检查收益率RET是否小于0。
    # STD(RET, 20): 这计算了收益率RET的20日标准差。
    # (RET < 0) ? STD(RET, 20): CLOSE: 这是一个条件操作。当RET小于0时，取值为RET的20日标准差；否则，取值为当天的收盘价CLOSE。
    # ... ^ 2: 将上述值平方。
    # MAX(..., 5): 选择上个数和5之间最大的那个。
    # RANK(...): 对上述最大值进行排名。
    # 整体而言，此技术指标似乎在试图度量过去5天中的最大风险（在RET为负时）与价格方向性（当RET为正时）。
    # 根据RET的值选择对应的数据
    # Compute rolling std for the entire 'returns' column
    # 避免使用未来函数
    data['returns_real'] = data['returns'].shift(1)
    data['rolling_std'] = data['returns_real'].rolling(window=20).std()

    # Now, use the computed rolling standard deviation inside the apply function
    data['selected_value'] = data.apply(
        lambda row: (row['returns_real'] < 0) * row['rolling_std'] + (row['returns_real'] >= 0) * row['close'], axis=1)

    # 对选择的值进行平方
    data['squared_result'] = data['selected_value'] ** 2

    # 计算过去5天的最大值
    data['final_value'] = data['squared_result'].apply(lambda x: max(x, 5))

    # 对最大值进行排名
    data['gj_010'] = data['final_value'].rolling(window=5, min_periods=5).apply(rf, raw=True)
    return data['gj_010']


def gj_011(data):
    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    # 这个Alpha因子尝试捕捉近期的价格动量并将其与交易量结合起来。
    # 如果收盘价接近当日最高点，这通常是一个积极的信号，而且如果这种行为持续出现并伴随着高交易量，那么这种上涨趋势可能会更为明显。
    # 相反，如果收盘价接近最低点，这可能是一个负面信号
    # Calculate the relative position of close within the daily range
    relative_position = (data['close'] - data['low']) - (data['high'] - data['close'])

    # Normalize with the daily range
    normalized_position = relative_position / (data['high'] - data['low'])

    # Multiply by volume
    volume_weighted = normalized_position * data['volume']

    # Calculate the rolling sum for 6 days
    data['gj_011'] = volume_weighted.rolling(window=6).sum()

    return data['gj_011']


def gj_012(data):
    # Alpha12 (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))

    # Assuming df is your dataframe with columns: 'OPEN', 'VWAP', and 'CLOSE'
    # 成交量加权平均价（VWAP）反映了投资者当天买入或卖出股票的平均价格，同时考虑了价格和成交量的因素。
    # 开盘价与 VWAP 的 10 天平均值之间的差额可以说明股票相对于近期平均交易价格的表现。
    # 收盘价与 VWAP 之间的绝对差值表明收盘价与当日平均交易价格的偏离程度。
    # 排名用于了解数据点在系列中的相对位置。在此公式中，您要查看的是开盘价偏离 10 天 VWAP 平均值的程度以及收盘价与 VWAP 之间的绝对差值的相互排名。

    # Calculate the 10-day mean of VWAP
    data['VWAP_10_mean'] = data['vwap'].rolling(window=10).mean()

    # Compute the difference between the open price and the 10-day mean of VWAP
    data['open_diff'] = data['open'] - data['VWAP_10_mean']

    # Calculate the absolute difference between the close price and the VWAP
    data['abs_diff'] = (data['close'] - data['vwap']).abs()

    # Calculate the ranks
    data['rank_open_diff'] = data['open_diff'].rolling(window=5, min_periods=5).apply(rf, raw=True)
    data['rank_abs_diff'] = data['abs_diff'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Compute Alpha12
    data['gj_012'] = data['rank_open_diff'] * (-1 * data['rank_abs_diff'])
    return data['gj_012']


def gj_013(data):
    # Alpha13 (((HIGH * LOW)^0.5) - VWAP)
    data["multiple"] = data['high'] * data['low']
    data['gj_013'] = data['multiple'] ** 0.5 - data['vwap']

    return data['gj_013']


def gj_014(data):
    # Alpha14 CLOSE - DELAY(CLOSE, 5)
    data['gj_014'] = data['close'] - data['close'].shift(5)
    return data['gj_014']


def gj_015(data):
    # Alpha15 OPEN / DELAY(CLOSE, 1) - 1
    data['gj_015'] = data['open'] / data['close'].shift(1) - 1
    return data['gj_015']


def gj_016(df):
    # Alpha16 (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    # Assuming df is your dataframe with columns: 'VOLUME' and 'VWAP'

    # vWAP（成交量加权平均价）是一个交易基准，它提供了一个关于股票成交量的平均价格的概念。
    # 通过对 VWAP 和成交量进行排序，我们可以看到它们在一段时间内的相对位置。
    # 通过这些排名值之间的相关性，我们可以了解成交量和价格走势（由 VWAP 表示）是同步还是背离。
    # 使用排序相关性的 TSMAX 函数，我们可以确定 5 天窗口中的最大相关性，从而了解成交量和价格走势最同步的时间。
    # 减去该值（乘以-1）可以优先考虑相关性最大的情况，这表明成交量和价格之间存在分歧。

    # Rank the volume and VWAP values
    df['rank_volume'] = df['volume'].rolling(window=5, min_periods=5).apply(rf, raw=True)
    df['rank_vwap'] = df['vwap'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Calculate the 5-day correlation between the ranked values
    df['corr_5'] = df['rank_volume'].rolling(window=5).corr(df['rank_vwap'])

    # Rank the 5-day correlation values
    df['rank_corr_5'] = df['corr_5'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Identify the 5-day maximum of the ranked correlation
    df['tsmax_5'] = df['rank_corr_5'].rolling(window=5).max()

    # Compute Alpha16
    df['gj_016'] = -1 * df['tsmax_5']
    return df['gj_016']


def gj_017(df):
    # 与期货市场的对应比较难确定，暂时不考虑
    # Alpha17 RANK((VWAP - MAX(VWAP, 15)))*DELTA(CLOSE, 5)
    # VWAP（成交量加权平均价）是一个重要的交易基准，交易者根据成交量和价格来评估期货全天交易的平均价格。
    # 当前 VWAP 与过去 15 天内最大值之间的差值可以说明，与近期历史相比，该股票目前的交易价格是高还是低。
    # 5 天收盘价的 delta 值（或差值）可以衡量期货的动量。
    # 结合这两点，该公式基本上可以检查近期交易量/价格动态与股票动量之间的关系。
    # 如果股价与近期历史相比处于相对低位（diff_vwap_max 为负），且具有正动量（delta_close_5 为正），则 Alpha17 的值将为正且高。
    df['max_value'] = df['vwap'].rolling(window=15).max()

    # Compute the difference between the current VWAP and max_value
    df['diff_vwap_max'] = df['vwap'] - df['max_value']

    # Calculate the change in the closing price over 5 days
    df['delta_close_5'] = df['close'].diff(5)

    # Rank the diff_vwap_max values
    df['rank_diff_vwap_max'] = df['diff_vwap_max'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Compute Alpha17
    df['gj_017'] = df['rank_diff_vwap_max'] * df['delta_close_5']
    return df['gj_017']


def gj_018(df):
    # Alpha18 CLOSE/DELAY(CLOSE,5)
    df['gj_018'] = df['close'] / df['close'].shift(5)
    return df['gj_018']


def gj_019(df):
    # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    # 该公式用于衡量 5 天内的价格动量。它衡量的是收盘价相对于 5 天前收盘价的百分比变化。
    # 这种变化的标准化方式（除以旧价或现价）取决于价格是上涨还是下跌。这可以帮助我们了解近期价格变动的重要性。
    df['delay_close_5'] = df['close'].shift(5)

    # Compute the conditional expression for Alpha19
    def compute_alpha19(row):
        close = row['close']
        delay_close = row['delay_close_5']

        if close < delay_close:
            return (close - delay_close) / delay_close
        elif close == delay_close:
            return 0
        else:
            return (close - delay_close) / close

    df['gj_019'] = df.apply(compute_alpha19, axis=1)
    return df['gj_019']


def gj_020(df):
    # Alpha20 (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    # Alpha20 用于衡量股票在 6 天内的动量。它表明股票在过去 6 天内按百分比计算的涨跌幅度。
    # 这种指标通常用于金融领域，以了解短期价格走势。如果 Alpha20 为正值，则表示该股票在过去 6 天内上涨；如果为负值，则表示该股票下跌。
    # Compute the percentage change over 6 days
    df['gj_020'] = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)

    return df['gj_020']


def gj_021(df):
    """
    Alpha21 REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    Alpha21 用于衡量过去 6 天内收盘价的趋势。正的 β 表示上升趋势，而负的 β 表示下降趋势。
    REGBETA(A, B, n) 在前 n 期的样本 A 对 B 进行回归得到的回归系数。
    SEQUENCE(n) 生成 1~n 的等差序列。
    """

    # Compute the mean of the closing prices over 6 days
    df['mean_close_6'] = df['close'].rolling(window=6).mean()

    # Compute the regression coefficient beta
    def regbeta(array):
        X = np.arange(1, 7).reshape(-1, 1)  # This is our SEQUENCE(6)
        y = array
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    df['gj_021'] = df['mean_close_6'].rolling(window=6).apply(regbeta, raw=True)

    return df['gj_021']


def gj_022(df):
    """
    Alpha22 SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    Alpha22 用于衡量收盘价与其短期（6 天）平均值之间的偏差的动量。
    通过比较当前的偏差与 3 天前的偏差，我们可以了解这种动量是增加还是减少。
    使用 12 天的简单移动平均线平滑这些差异，以减少日常噪声并提供更稳定的动量测量。
    """

    # Compute the mean of the closing prices over 6 days
    df['mean_close_6'] = df['close'].rolling(window=6).mean()

    # Compute the relative difference between the current close and its 6-day mean
    df['relative_diff'] = (df['close'] - df['mean_close_6']) / df['mean_close_6']

    # Calculate the difference from the relative difference 3 days ago
    df['diff_from_delayed'] = df['relative_diff'] - df['relative_diff'].shift(3)

    # Compute Alpha22 using custom Simple Moving Average
    df['gj_022'] = custom_sma(df['diff_from_delayed'], n=12, m=1)

    return df['gj_022']


def gj_023(df):
    """
    Alpha23
    SMA((CLOSE > DELAY(CLOSE,1) ? STD(CLOSE, 20) : 0), 20, 1)
    /
    (SMA((CLOSE > DELAY(CLOSE,1) ? STD(CLOSE, 20) : 0), 20, 1) + SMA((CLOSE <= DELAY(CLOSE,1) ? STD(CLOSE, 20) : 0), 20, 1)) * 100

    Alpha23 用于比较期货近期收盘价上涨和下跌时的波动性。这可能对于捕捉期货的短期动态变化有用。
    """

    # 计算20日滚动标准差
    rolling_std_20 = df['close'].rolling(window=20).std()

    # 使用向量化操作计算temp1和temp2
    condition1 = df['close'] > df['close'].shift(1)
    condition2 = df['close'] <= df['close'].shift(1)

    df['temp1'] = np.where(condition1, rolling_std_20, 0)
    df['temp2'] = np.where(condition2, rolling_std_20, 0)

    # 使用自定义的SMA函数计算SMA
    df['SMA_temp1'] = custom_sma(df['temp1'], n=20, m=1)
    df['SMA_temp2'] = custom_sma(df['temp2'], n=20, m=1)

    # 计算Alpha23
    df['gj_023'] = (df['SMA_temp1'] / (df['SMA_temp1'] + df['SMA_temp2'])) * 100

    # 删除临时列
    df.drop(columns=['temp1', 'temp2', 'SMA_temp1', 'SMA_temp2'], inplace=True)

    return df['gj_023']


def gj_024(df):
    """
    Alpha24 SMA(CLOSE-DELAY(CLOSE,5),5,1)
    Alpha24 用于衡量期货的短期动量。通过查看当前收盘价与5天前的收盘价之间的差异，它捕捉到期货运动的近期趋势。
    """
    # 计算CLOSE与5天前的CLOSE之间的差异
    df['difference'] = df['close'] - df['close'].shift(5)

    # 使用自定义的SMA函数计算5天窗口的平均值
    df['gj_024'] = custom_sma(df['difference'], n=5, m=1)

    return df['gj_024']


def gj_025(df):
    """
    Alpha25:
    (-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250)))

    Alpha25 combines several indicators:
    1. Price momentum through the difference in close prices.
    2. Volume dynamics by comparing current volume to a 20-day average.
    3. Long term returns with a sum of 250-day returns.

    This strategy provides a composite score for futures based on price momentum, volume dynamics, and long-term returns.

    Args:
    - df: DataFrame containing 'close', 'volume', and 'ret' columns.

    Returns:
    - Series of Alpha25 values.

    """

    # 计算当前收盘价与7天前的收盘价之间的差异
    delta_close_7 = df['close'].diff(7)

    # 计算20天平均成交量
    mean_volume_20 = df['volume'].rolling(window=20).mean()

    # 计算当前成交量与20天平均成交量的比率的线性衰减
    volume_ratio = df['volume'] / mean_volume_20
    decayed_volume_ratio = decay_linear(volume_ratio, 9)

    df['returns_real'] = df['returns'].shift(1)
    # 组合指标并计算最终的Alpha25值
    rank1 = delta_close_7 * (1 - decayed_volume_ratio.rolling(window=5, min_periods=5).apply(rf, raw=True))
    rank2 = df['returns_real'].rolling(window=250).sum().rolling(window=5, min_periods=5).apply(rf, raw=True)

    df['gj_025'] = (-1 * rank1.rolling(window=5, min_periods=5).apply(rf, raw=True)) * (1 + rank2)

    return df['gj_025']


def gj_026(df):
    """
    Alpha26: ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 22))))

    Alpha26分析了以下几点:
    1. 当前收盘价与过去7天的平均收盘价之间的偏差。这可以帮助我们了解价格是否偏离了短期的平均水平。
    2. VWAP与5天前的收盘价在过去230天内的相关性。这可能提供了有关交易量如何与价格变化关联的见解。
    第一部分计算当前收盘价与 7 天平均收盘价的偏差。
    第二部分测量的是 230 天窗口内成交量加权平均价（VWAP）与 5 天前收盘价之间的相关性。
    该公式将这两个指标结合在一起，有助于深入了解价格动量以及成交量与价格变化之间的关系。
    Args:
    - df: 包含'close'和'vwap'列的DataFrame.

    Returns:
    - DataFrame with the new column 'gj_026'.
    """

    # 计算过去7天的平均收盘价
    mean_close_7 = df['close'].rolling(window=7).mean()

    # 计算VWAP与5天前的收盘价在过去22天内的相关性
    corr_vwap_close = df['vwap'].rolling(window=22).corr(df['close'].shift(5))

    # 组合指标以得到Alpha26
    df['gj_026'] = (mean_close_7 - df['close']) + corr_vwap_close

    return df['gj_026']


def gj_027(df):
    """
    Alpha27: WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)

    Alpha27分析了以下几点:
    1. 价格与3天前及6天前的价格的相对变化。
    2. 使用WMA平滑这些变化，从而考虑到更近的变化对当前市场情况的影响较大。

    Args:
    - df: 包含'close'列的DataFrame.

    Returns:
    - DataFrame with the new column 'gj_027'.
    """

    # 计算3天和6天的价格变化
    relative_change_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    relative_change_6 = (df['close'] - df['close'].shift(6)) / df['close'].shift(6) * 100

    combined_change = relative_change_3 + relative_change_6

    # 使用WMA计算Alpha27
    df['gj_027'] = wma(combined_change, 12)

    return df['gj_027']


def gj_028(df):
    """
    Alpha28: 3 * SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9)) * 100, 3, 1)
            - 2 * SMA(SMA((CLOSE - TSMIN(LOW, 9)) / (MAX(HIGH, 9) - TSMAX(LOW, 9)) * 100, 3, 1), 3, 1)

    Alpha28 的逻辑如下:
    1. 计算当前价格与过去9天的最低价的相对差异。
    2. 计算过去9天的最高价与最低价的差异。
    3. 使用上述两者计算一个百分比值，代表价格在其9天范围内的位置。
    4. 使用这个百分比值计算两个不同的SMA，并结合这两个SMA来得到最终的 Alpha28 值。

    Args:
    - df: 包含 'close', 'low', 和 'high' 列的 DataFrame.

    Returns:
    - DataFrame with the new column 'gj_028'.
    """

    # 1 & 2. 计算相对差异和9天范围
    relative_difference = df['close'] - df['low'].rolling(window=9).min()
    nine_day_range = df['high'].rolling(window=9).max() - df['low'].rolling(window=9).min()

    # 3. 计算位置百分比
    position_percentage = (relative_difference / nine_day_range) * 100

    # 4. 使用这个百分比值计算两个不同的SMA
    sma1 = custom_sma(position_percentage, n=3, m=1)
    sma2 = custom_sma(sma1, n=3, m=1)

    # 结合这两个SMA来得到 Alpha28 值
    df['gj_028'] = 3 * sma1 - 2 * sma2

    return df['gj_028']


def gj_029(df):
    """
    Alpha29: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME

    Alpha29 的逻辑如下:
    - 计算当前的收盘价与6天前的收盘价之间的百分比变化。
    - 将这个百分比变化与当天的交易量相乘，得到一个调整后的动量值。

    这个指标的目的是捕捉近期的价格动态，并通过考虑交易量来给出一个关于市场强度的指标。
    如果交易量很高，那么这种价格变化可能更有意义。

    Args:
    - df: 包含 'close' 和 'volume' 列的 DataFrame.

    Returns:
    - DataFrame with the new column 'gj_029'.
    """

    # 计算当前的收盘价与6天前的收盘价之间的百分比变化
    price_change_percentage = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)

    # 将这个百分比变化与当天的交易量相乘
    df['gj_029'] = price_change_percentage * df['volume']

    return df['gj_029']


def gj_031(df):
    """
    Alpha31 测量当前收盘价与其12天平均值的偏差。这一偏差以12天平均值的百分比形式表示。

    如果该指标的值为正，这表示当前的收盘价高于近期的平均水平，
    暗示可能的上升动力或被高估。相反，负值可能表示下行动力或被低估。
    """

    # Calculate the 12-day mean of the closing prices
    df['mean_close_12'] = df['close'].rolling(window=12).mean()

    # Compute Alpha31
    df['gj_031'] = (df['close'] - df['mean_close_12']) / df['mean_close_12'] * 100

    return df['gj_031']


def gj_032(df):
    """
    Alpha32: -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)

    Alpha32 旨在捕捉短期内高价和成交量之间的关系。
    它首先分别对高价和成交量进行排名。
    然后，该指标测量这些排名在3天窗口内的相关性。
    最后，它对这些短期相关性进行排名，并在另一个3天的窗口内进行累积。

    如果该指标呈现高正值，这可能表明高价和高成交量之间存在一致的相关性，暗示着强劲的市场动力。
    相反，负值可能表示高价和成交量之间存在逆向关系，这可能突显出市场的异常或趋势反转。
    """

    # Rank high prices and volumes
    df['rank_high'] = df['high'].rolling(window=5, min_periods=5).apply(rf, raw=True)
    df['rank_volume'] = df['volume'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Compute the rolling correlation of the ranks over a 3-day window
    df['corr_rank'] = df['rank_high'].rolling(window=3).corr(df['rank_volume'])

    # Rank the rolling correlations
    df['rank_corr'] = df['corr_rank'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Sum up the ranks of the correlations over a 3-day window
    df['gj_032'] = -1 * df['rank_corr'].rolling(window=3).sum()

    return df['gj_032']


def gj_033(df):
    """
    Alpha33: ((((-1 * TSMIN(LOW, 5))
    + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 100) - SUM(RET, 20)) / 80))) * TSRANK(VOLUME, 5)

    Alpha33: 该公式综合考虑了短期价格下跌、长期净动量（排除最近一个月的数据）以及近期的成交量。具体来说，公式包括以下几个部分：
    首先计算过去5天内最低价的最小值（TSMIN(LOW, 5)），并取其负值；
    然后计算5天前该最小值（DELAY(TSMIN(LOW, 5), 5)）；
    计算过去100天的收益总和减去过去20天的收益总和后，再除以80天，得到一个期间内的平均收益率，然后对其进行排名（RANK）；
    最后，计算过去5天内的成交量的时间序列排名（TSRANK(VOLUME, 5)）。
    这个指标综合以上因素，用于捕捉短期价格下跌、长期净动量和近期成交量之间的相互作用。当该指标呈现高正值时，表明最近的价格出现下跌，过去一年的净动量较强，并且近日的成交量较高。
    """

    # Compute the 5-day minimum low price and its delayed value
    df['tsmin_low'] = df['low'].rolling(window=5).min()
    df['delay_tsmin_low'] = df['tsmin_low'].shift(5)

    # Compute the net momentum measure
    df['ret_240'] = df['close'].pct_change().rolling(window=100).sum()
    df['ret_20'] = df['close'].pct_change().rolling(window=20).sum()
    df['momentum'] = (df['ret_240'] - df['ret_20']) / 80

    # Rank the net momentum measure
    df['rank_momentum'] = df['momentum'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Compute the time-series rank of volume over a 5-day window
    df['tsrank_volume'] = df['volume'].rolling(window=5).apply(lambda x: pd.Series(x).rolling(window=5, min_periods=5).apply(rf, raw=True).iloc[-1])

    # Compute Alpha33
    df['gj_033'] = ((-1 * df['tsmin_low'] + df['delay_tsmin_low']) * df['rank_momentum']) * df['tsrank_volume']

    return df['gj_033']


def gj_034(df):
    """
    Alpha34: MEAN(CLOSE,12)/CLOSE

    Alpha36: 测量过去六天内成交量排名和成交量加权平均价（VWAP）排名之间的相关性。
    该指标通过基于成交量的数据提供对交易动态以及价格趋势的强度或弱度的洞察。
    简言之，它分析成交量和成交量加权平均价在一定时期内的相互关系，以判断市场趋势。
    """

    # Compute the 12-day mean close price
    df['mean_close_12'] = df['close'].rolling(window=12).mean()

    # Compute Alpha34
    df['gj_034'] = df['mean_close_12'] / df['close']

    return df['gj_034']


def gj_035(df):
    """
    Alpha35
    (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +
    (OPEN *0.35)), 17),7))) * -1)
    Alpha35 结合了开盘价的动量及其与成交量的关系。该指标捕捉了价格变动的本质及其与成交量动态的关联。
    这意味着它考虑了开盘价的趋势变化，并分析这些变化如何受到成交量因素的影响，从而提供对市场行为更深入的理解。
    """

    # Compute the change in OPEN from the previous day
    df['delta_open'] = df['open'].diff()

    # Compute correlation between VOLUME and transformed OPEN
    df['corr_vol_open'] = df['volume'].rolling(window=17).corr(df['open'] * 0.65 + df['open'] * 0.35)

    # Compute DECAYLINEAR values using the provided function
    df['decay_delta_open'] = decay_linear(df['delta_open'], 15)
    df['decay_corr'] = decay_linear(df['corr_vol_open'], 7)

    # Compute RANK values
    df['rank_decay_delta_open'] = df['decay_delta_open'].rolling(window=5, min_periods=5).apply(rf, raw=True)
    df['rank_decay_corr'] = df['decay_corr'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # Compute Alpha35
    df['gj_035'] = df[['rank_decay_delta_open', 'rank_decay_corr']].min(axis=1) * -1

    return df['gj_035']


def gj_036(df):
    """
    Alpha36 RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
    Alpha36 通过计算6天时间窗口内成交量排名和成交量加权平均价（VWAP）排名之间的相关性。
    它通过考虑基于成交量的价格趋势的强弱，提供了对交易动态的深入洞察。

    具体来说，Alpha36 首先对成交量和VWAP分别进行排名，然后测量这两者在6天内的相关性，并对这些相关性的总和进行排名。
    这一过程有助于识别成交量与价格表现之间的相互关系，进而判断市场趋势的强度。
    """

    # Compute ranks for VOLUME and VWAP
    df['rank_volume'] = df['volume'].rolling(window=10, min_periods=10).apply(rf, raw=True)
    df['rank_vwap'] = df['vwap'].rolling(window=10, min_periods=10).apply(rf, raw=True)

    # Compute rolling correlation between the ranks over a window of 6 days
    df['corr_rank'] = df['rank_volume'].rolling(window=6).corr(df['rank_vwap'])

    # Sum the correlations over a window of 2 days
    df['sum_corr'] = df['corr_rank'].rolling(window=2).sum()

    # Compute RANK of the summed correlation values
    df['gj_036'] = df['sum_corr'].rolling(window=10, min_periods=10).apply(rf, raw=True)

    return df['gj_036']


def gj_037(df):
    """
    Alpha37 (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    Alpha37: 使用过去5天的开盘价之和与同期的收益率之和的乘积，然后将这个乘积与其在10天前的值相比较。
    具体来说，该公式首先计算过去5天开盘价的总和与同期收益率的总和的乘积，然后从这个值中减去10天前该乘积的值，并对结果取反的排名。
    Alpha37 的功能是结合了过去5天的开盘价总和和同期的收益率总和。
    这个特征试图捕捉开盘价和收益率乘积的动量。它用于识别可能正在经历短期回调或者上涨的情况。
    """
    df['returns_real'] = df['returns'].shift(1)
    # Calculate the product of SUM(OPEN, 5) and SUM(RET, 5)
    df['product_5'] = df['open'].rolling(window=5).sum() * df['returns_real'].rolling(window=5).sum()

    # Subtract the delayed product from the current product
    df['diff_product'] = df['product_5'] - df['product_5'].shift(10)

    # Rank the difference and multiply by -1
    df['gj_037'] = -1 * df['diff_product'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    return df['gj_037']


def gj_038(df):
    """
    Alpha38: (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)

    该因子考虑了当最近的高价超过其过去20日的平均高价时的高价变化。如果最近的高价确实超过了该均值，则我们考虑两天前高价的变化；否则，因子值为0。
    这可能是在试图捕捉价格超越其近期范围时的动量变化。
    """
    # 计算过去20日的高价均值
    df['mean_high_20'] = df['high'].rolling(window=20).mean()

    # 计算高价的2日变化
    df['delta_high_2'] = df['high'].diff(2)

    # 根据条件计算Alpha38
    df['gj_038'] = df.apply(lambda row: -1 * row['delta_high_2'] if row['high'] > row['mean_high_20'] else 0, axis=1)

    return df['gj_038']


def gj_039(df):
    """
    Alpha39: ((RANK(DECAYLINEAR(DELTA(CLOSE, 2),8)) - RANK(DECAYLINEAR(CORR((VWAP * 0.3) + (OPEN * 0.7),
    SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1

    这个因子首先考虑了CLOSE价格的两天变化的线性衰减。
    接着，它考虑了VWAP和OPEN的加权组合与过去180天的平均交易量之和的相关性，并对这个相关性进行了线性衰减。
    最后，这个因子考虑了上述两个值的排序，并从第一个值中减去第二个值。
    """

    # 计算CLOSE的2日变化
    df['delta_close'] = df['close'].diff(2)

    # 计算VWAP和OPEN的加权组合
    df['weighted_combo'] = (df['vwap'] * 0.3) + (df['open'] * 0.7)

    # 计算过去180天的平均交易量之和
    df['mean_vol_180'] = df['volume'].rolling(window=180).mean()
    df['sum_mean_vol_180'] = df['mean_vol_180'].rolling(window=37).sum()

    # 计算相关性
    df['corr'] = df['weighted_combo'].rolling(window=14).corr(df['sum_mean_vol_180'])

    # 计算线性衰减
    df['decay_delta_close'] = decay_linear(df['delta_close'], 8)
    df['decay_corr'] = decay_linear(df['corr'], 12)

    # 使用rank计算Alpha39
    df['gj_039'] = (df['decay_delta_close'].rank() - df['decay_corr'].rank()) * -1

    return df['gj_039']


def gj_040(df):
    """
    Alpha40: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100

    解释:
    当今天的收盘价大于昨天的收盘价时，我们将计算这些日子的成交量总和。
    同样，当今天的收盘价小于或等于昨天的收盘价时，我们也将计算这些日子的成交量总和。
    最后，我们将取前者的总和与后者的总和的比例，并乘以100。

    这个因子通过比较上涨和下跌日的成交量来提供市场情绪的量度。它可以用来衡量买方或卖方在过去的交易日中的优势。
    """

    # 当今天的收盘价大于昨天的收盘价时，获取成交量，否则为0
    df['up_volume'] = df.apply(lambda row: row['volume'] if row['close'] > df['close'].shift(1)[row.name] else 0,
                               axis=1)

    # 当今天的收盘价小于或等于昨天的收盘价时，获取成交量，否则为0
    df['down_volume'] = df.apply(lambda row: row['volume'] if row['close'] <= df['close'].shift(1)[row.name] else 0,
                                 axis=1)

    # 计算上涨和下跌日的成交量的26日总和
    df['up_volume_sum'] = df['up_volume'].rolling(window=26).sum()
    df['down_volume_sum'] = df['down_volume'].rolling(window=26).sum()

    # 计算Alpha40
    df['gj_040'] = (df['up_volume_sum'] / df['down_volume_sum']) * 100

    return df['gj_040']


def gj_041(df):
    """
    Alpha41: (RANK(MAX(DELTA((VWAP), 3), 5))* -1)

    解释:
    1. 计算VWAP在过去3天内的变化率。
    2. 找出这个变化率在过去5天内的最大值。
    3. 将这个最大值赋予一个排名。
    4. 最后，乘以-1调整排名的方向，使得变化最大的值得到最低的排名。

    这个指标可以帮助我们识别近期内VWAP变化最大的日子，而这可能意味着市场上有重要的新闻或事件影响了股价。
    """

    # 计算VWAP在过去3天内的变化率
    df['delta_vwap'] = df['vwap'].diff(3)

    # 找出这个变化率在过去5天内的最大值
    df['max_delta_vwap'] = df['delta_vwap'].rolling(window=5).max()

    # 赋予一个排名
    df['rank_vwap'] = df['max_delta_vwap'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 乘以-1
    df['gj_041'] = df['rank_vwap'] * -1

    return df['gj_041']


def gj_042(df):
    """
    Alpha42: ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))

    解释:
    1. 计算过去10天的最高价的标准差来衡量价格的波动性。
    2. 计算过去10天的最高价和成交量之间的相关性来评估价格与成交量之间的关系。
    3. 将最高价的10天标准差进行排名，然后乘以-1。
    4. 上述的排名与最高价和成交量的10天相关性相乘得到因子值。

    这个指标考察了价格的波动性和价格与成交量之间的关系。如果波动性较低且价格与成交量高度相关，这可能意味着市场在一个健康的上涨趋势中。
    """

    # 计算过去10天的最高价的标准差
    df['std_high'] = df['high'].rolling(window=10).std()

    # 计算过去10天的最高价和成交量之间的相关性
    df['corr_high_volume'] = df['high'].rolling(window=10).corr(df['volume'])

    # 将最高价的10天标准差进行排名
    df['rank_std_high'] = df['std_high'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 上述的排名乘以-1后与最高价和成交量的10天相关性相乘得到因子值
    df['gj_042'] = (-1 * df['rank_std_high']) * df['corr_high_volume']

    return df['gj_042']


def gj_043(df):
    """
    Alpha43: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)

    解释:
    1. 当天的收盘价高于前一天的收盘价时，我们取当天的成交量。
    2. 当天的收盘价低于前一天的收盘价时，我们取当天的成交量的负值。
    3. 当天的收盘价与前一天的收盘价相等时，取值为0。

    这个指标考察了价格变动背后的成交量动力，可以帮助我们了解上涨或下跌背后的成交量动力。
    """

    # 根据上述逻辑计算成交量的调整值
    df['adjusted_volume'] = np.where(df['close'] > df['close'].shift(1), df['volume'],
                                     np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))

    # 对过去6天的调整后的成交量求和
    df['gj_043'] = df['adjusted_volume'].rolling(window=6).sum()

    return df['gj_043']


def gj_044(df):
    """
    Alpha44: (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) +
             TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
    """

    # 计算7天内LOW（最低价）与10日移动平均成交量之间的相关性
    corr_low_volume = df['low'].rolling(window=7).corr(df['volume'].rolling(window=10).mean())

    # 对这个相关性应用6天期限的DECAYLINEAR（线性衰减）
    decay_corr = decay_linear(corr_low_volume, 6)

    # 对衰减后的相关性计算4天的时间序列排名
    tsrank_decay_corr = decay_corr.rolling(window=4).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)

    # 计算VWAP（成交量加权平均价）3天的变化
    delta_vwap = df['vwap'].diff(3)

    # 对VWAP的变化应用10天期限的DECAYLINEAR（线性衰减）
    decay_delta_vwap = decay_linear(delta_vwap, 10)

    # 对衰减后的VWAP变化计算15天的时间序列排名
    tsrank_decay_delta_vwap = decay_delta_vwap.rolling(window=15).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                                                                        raw=True)

    # 计算Alpha44的最终值
    df['gj_044'] = tsrank_decay_corr + tsrank_decay_delta_vwap

    return df['gj_044']


def gj_045(df):
    """
    Alpha45: (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
    """

    # 计算（收盘价 * 0.6 + 开盘价 * 0.4）加权和的1天变化量
    weighted_price_delta = ((df['close'] * 0.6) + (df['open'] * 0.4)).diff(1)

    # 对加权价格变化量进行排名
    rank_weighted_price_delta = weighted_price_delta.rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 计算VWAP与150日成交量移动平均在15日窗口内的相关性
    corr_vwap_volume = df['vwap'].rolling(window=15).corr(df['volume'].rolling(window=150).mean())

    # 对相关性进行排名
    rank_corr = corr_vwap_volume.rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 计算Alpha45的最终值
    df['gj_045'] = rank_weighted_price_delta * rank_corr

    return df['gj_045']


def gj_046(df):
    """
    Alpha46: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    这个特征是一种相对强度指数的类型。
    通过计算四种不同移动平均的平均值，并将此值与当前收盘价进行比较，
    它衡量当前价格相对于其近期历史平均水平是高还是低。
    如果值大于1，则表明当前收盘价低于其近期历史平均水平，
    这可能被视为看涨信号（即，资产被低估）。
    相反，如果值小于1，则表明当前收盘价高于其近期历史平均水平，
    表示资产可能被高估。
    """

    # 计算四个移动平均
    mean_3 = df['close'].rolling(window=3).mean()
    mean_6 = df['close'].rolling(window=6).mean()
    mean_12 = df['close'].rolling(window=12).mean()
    mean_24 = df['close'].rolling(window=24).mean()

    # 计算这四个移动平均的平均值
    mean_avg = (mean_3 + mean_6 + mean_12 + mean_24) / 4

    # 将此平均值除以当前的收盘价
    df['gj_046'] = mean_avg / df['close']

    return df['gj_046']


def gj_047(df):
    """
    Alpha47: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)

    此公式计算的是在过去6天内最高价的最大值与收盘价的差值，相对于这6天内最高价的最大值和最低价的最小值之间差值的比例，然后将该比例乘以100。
    接着对这个比例值应用9天窗口的简单移动平均（SMA）。
    """

    # 计算过去6天内的最高价的最大值
    high_6 = df['high'].rolling(window=6).max()

    # 计算过去6天内的最低价的最小值
    low_6 = df['low'].rolling(window=6).min()

    # 计算收盘价在6天高低区间内的相对位置
    relative_position = (high_6 - df['close']) / (high_6 - low_6) * 100

    # 应用9天窗口的简单移动平均
    df['gj_047'] = custom_sma(relative_position, n=9, m=1)

    return df['gj_047']


def gj_048(df):
    """
    Alpha48: (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2))) +
              SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20)
    此指标利用过去3天的动量和相对成交量活动度计算一个值。具体来说，它首先使用SIGN函数来计算过去3天的价格动量，
    然后计算5天总成交量与20天总成交量的比率，最后将这两个值相乘并取反得到最终的Alpha48值。
    """

    # 使用SIGN函数计算过去3天的动量
    momentum_3d = (
            df['close'].diff(1).apply(np.sign) +
            df['close'].diff(2).apply(np.sign) +
            df['close'].diff(3).apply(np.sign)
    )

    # 计算相对成交量活动度
    volume_ratio = (
            df['volume'].rolling(window=5).sum() /
            df['volume'].rolling(window=20).sum()
    )

    # 计算最终的Alpha48值
    df['gj_048'] = (-1 * momentum_3d.rolling(window=5, min_periods=5).apply(rf, raw=True) * volume_ratio)

    return df['gj_048']


def gj_049(df):
    """
    Alpha49 特征计算
    Alpha49
    SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    /(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    +SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    描述：
    此特征计算过去12天内上涨价格动力与总价格动力的比率。比率较高表明价格上涨趋势较强。
    """
    # 计算上涨和下跌条件
    df['upward_condition'] = (df['high'] + df['low']) >= (df['high'].shift(1) + df['low'].shift(1))
    df['downward_condition'] = (df['high'] + df['low']) <= (df['high'].shift(1) + df['low'].shift(1))

    # 计算上涨和下跌动力
    df['upward_movement'] = np.where(df['upward_condition'], 0,
                                     np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                np.abs(df['low'] - df['low'].shift(1))))

    df['downward_movement'] = np.where(df['downward_condition'], 0,
                                       np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                  np.abs(df['low'] - df['low'].shift(1))))

    # 计算过去12天的上涨和下跌动力总和
    df['sum_upward_12'] = df['upward_movement'].rolling(window=12).sum()
    df['sum_downward_12'] = df['downward_movement'].rolling(window=12).sum()

    # 计算Alpha49的值
    df['gj_049'] = df['sum_upward_12'] / (df['sum_upward_12'] + df['sum_downward_12'])

    return df['gj_049']


def gj_050(df):
    """
    Alpha50 特征计算
    计算公式：
    SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    -SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    /(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    +SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

    描述：
    此特征计算过去12天内下跌价格动力与上涨价格动力比率的差值。
    较高（正）的值表明价格下跌趋势较强，
    而较低（负）的值表明价格上涨趋势较强。
    """
    # 计算上涨和下跌条件
    df['downward_condition'] = (df['high'] + df['low']) <= (df['high'].shift(1) + df['low'].shift(1))
    df['upward_condition'] = (df['high'] + df['low']) >= (df['high'].shift(1) + df['low'].shift(1))

    # 计算上涨和下跌动力
    df['downward_movement'] = np.where(df['downward_condition'], 0,
                                       np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                  np.abs(df['low'] - df['low'].shift(1))))

    df['upward_movement'] = np.where(df['upward_condition'], 0,
                                     np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                np.abs(df['low'] - df['low'].shift(1))))

    # 计算过去12天的上涨和下跌动力总和
    df['sum_downward_12'] = df['downward_movement'].rolling(window=12).sum()
    df['sum_upward_12'] = df['upward_movement'].rolling(window=12).sum()

    # 计算Alpha50的两部分值
    part1 = df['sum_downward_12'] / (df['sum_downward_12'] + df['sum_upward_12'])
    part2 = df['sum_upward_12'] / (df['sum_upward_12'] + df['sum_downward_12'])

    # 计算Alpha50的值
    df['gj_050'] = part1 - part2

    return df['gj_050']


def gj_051(df):
    """
    Alpha51 特征计算
    计算公式：
    SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
    +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    描述：
    此特征计算过去12天内下跌价格动力与总价格动力的比率。较高的值表明相对于整体趋势，价格下跌趋势较强，
    而较低的值表明相对于整体趋势，价格下跌趋势较弱。
    """

    # 计算下跌动力
    downward_condition = (df['high'] + df['low']) <= (df['high'].shift(1) + df['low'].shift(1))
    df['downward_movement'] = np.where(downward_condition,
                                       np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                  np.abs(df['low'] - df['low'].shift(1))),
                                       0)

    # 计算上涨动力
    upward_condition = (df['high'] + df['low']) >= (df['high'].shift(1) + df['low'].shift(1))
    df['upward_movement'] = np.where(upward_condition,
                                     np.maximum(np.abs(df['high'] - df['high'].shift(1)),
                                                np.abs(df['low'] - df['low'].shift(1))),
                                     0)

    # 计算Alpha51公式的分子和分母
    numerator = df['downward_movement'].rolling(window=12).sum()
    denominator = numerator + df['upward_movement'].rolling(window=12).sum()

    # 计算Alpha51的值
    df['gj_051'] = numerator / denominator

    return df['gj_051']


def gj_052(df):
    """
    Alpha52 特征计算
    计算公式：
    SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100
    描述：
    此特征捕捉过去26天内上涨价格动力相对于总价格动力的强度。较高的值表明相对于总价格动力，价格上涨趋势较强，
    而较低的值表明上涨趋势较弱。
    """

    # 计算高价、低价和收盘价的平均价
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3

    # 计算正向价格动力
    df['positive_movement'] = np.maximum(0, df['high'] - df['avg_price'].shift(1))

    # 计算负向价格动力
    df['negative_movement'] = np.maximum(0, df['avg_price'].shift(1) - df['low'])

    # 计算Alpha52公式的分子和分母
    numerator = df['positive_movement'].rolling(window=26).sum()
    denominator = df['negative_movement'].rolling(window=26).sum()

    # 计算Alpha52的值
    df['gj_052'] = (numerator / denominator) * 100

    return df['gj_052']


def gj_053(df):
    """
    Alpha53 特征计算
    Alpha53 COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    描述：
    此特征计算过去12天内当前收盘价高于前一日收盘价的天数占比。结果代表了价格上涨动力或强度。
    """

    # 识别收盘价高于前一日收盘价的日子
    df['close_gt_prev'] = (df['close'] > df['close'].shift(1)).astype(int)

    # 在12天窗口内计算此类日子的数量
    df['count_close_gt_prev'] = df['close_gt_prev'].rolling(window=12).sum()

    # 计算这12天周期内此类日子的百分比
    df['gj_053'] = (df['count_close_gt_prev'] / 12) * 100

    return df['gj_053']


def gj_054(df):
    """
    Alpha54 特征计算
    (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    描述：
    此特征结合了日内价格波动性、日内净价格运动以及10天周期内开盘价与收盘价之间的相关性的度量。然后对所有资产的这些综合度量进行排名，以指示哪些资产具有最可预测的日内价格行为。
    """

    # 计算日内波动性、日内价格运动以及10天开收盘价的相关性
    df['intraday_volatility'] = df['close'].sub(df['open']).abs().rolling(window=10).std()
    df['intraday_movement'] = df['close'] - df['open']
    df['corr_close_open'] = df['close'].rolling(window=10).corr(df['open'])

    # 计算日内价格行为的综合度量
    df['combined_measure'] = df['intraday_volatility'] + df['intraday_movement'] + df['corr_close_open']

    # 对综合度量进行排名
    df['gj_054'] = -1 * df['combined_measure'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    return df['gj_054']


def gj_055(df):
    # 由于调试不成功，该指标暂时没有被使用
    """
    Alpha55 特征计算
    计算公式：
    SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))
    >ABS(LOW-DELAY(CLOSE,1)) &ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))
    +ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))
    &ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2
    +ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))
    *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    描述：
    这个特征看似是一个基于量价趋势的变种指标。它评估相对于前期的价格变化，特别是收盘价，并结合了成交量动态。
    """

    # 计算考虑到当日开盘-收盘区间和前一日开盘-收盘区间的价格变化
    df['price_change'] = df['close'] - df['close'].shift(1) + (df['close'] - df['open'])/2 + df['close'].shift(1) - df['open'].shift(1)

    # 计算波动性度量
    df['vol1'] = abs(df['high'] - df['close'].shift(1))
    df['vol2'] = abs(df['low'] - df['close'].shift(1))
    df['vol3'] = abs(df['high'].shift(1) - df['low'].shift(1))

    # 根据波动性度量的相对大小计算加权价格变化的分母
    conditions = [
        (df['vol1'] > df['vol2']) & (df['vol1'] > df['vol3']),
        (df['vol2'] > df['vol3']) & (df['vol2'] > df['vol1'])
    ]
    choices = [
        df['vol1'] + df['vol2']/2 + abs(df['close'].shift(1) - df['open'].shift(1))/4,
        df['vol2'] + df['vol1']/2 + abs(df['close'].shift(1) - df['open'].shift(1))/4
    ]
    df['denominator'] = np.select(conditions, choices, default=df['vol3'] + abs(df['close'].shift(1) - df['open'].shift(1))/4)

    # 计算加权价格变化
    df['weighted_price_change'] = 16 * df['price_change'] / df['denominator']

    # 乘以最大波动性度量并在最近20个周期内求和
    df['gj_055'] = df['weighted_price_change'] * df[['vol1', 'vol2']].max(axis=1).rolling(window=20).sum()
    df['gj_055'] = df['gj_055']
    return df['gj_055']


def gj_056(df):
    """
    Alpha56:
    (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),
    SUM(MEAN(VOLUME,40), 19), 13))^5)))

    描述：
    此特征比较了最近价格动量（开盘价减去其12天最低值）的排名与最近价格平均值（19天内高低价平均值之和）
    和成交量动态（19天内40天平均成交量之和）相关性的排名之间的关系。它是价格动量与成交量动态之间的相互作用。
    """

    # 计算条件的第一部分
    df['part1'] = df['open'] - df['open'].rolling(window=12).min()
    df['rank_part1'] = df['part1'].rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 计算19天内高低价平均值之和
    df['hl_avg_19'] = ((df['high'] + df['low']) / 2).rolling(window=19).sum()

    # 计算19天内40天平均成交量之和
    df['vol_avg_19'] = df['volume'].rolling(window=40).mean().rolling(window=19).sum()

    # 在13天窗口内计算这两个序列的相关性
    df['corr_part'] = df['hl_avg_19'].rolling(window=13).corr(df['vol_avg_19'])

    # 对相关性进行排名，将其提高到5次幂，然后再次排名
    df['rank_part2'] = df['corr_part'].rolling(window=5, min_periods=5).apply(rf, raw=True).pow(5).rank()

    # 比较排名得到Alpha56的值
    df['gj_056'] = (df['rank_part1'] < df['rank_part2']).astype(int)

    return df['gj_056']


def gj_057(df):
    """
    Alpha57: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)

    描述：
    此特征计算收盘价相对于由9周期最高价和9周期最低价定义的区间的位置，并使用简单移动平均（SMA）对这个值进行平滑处理。
    它衡量收盘价在定义区间内的相对强度或弱度。
    """

    # 计算9周期的最低低点和9周期的最高高点
    min_low_9 = df['low'].rolling(window=9).min()
    max_high_9 = df['high'].rolling(window=9).max()

    # 计算比率
    ratio = (df['close'] - min_low_9) / (max_high_9 - min_low_9) * 100

    # 应用简单移动平均
    df['gj_057'] = custom_sma(ratio, n=3, m=1)

    return df['gj_057']


def gj_058(df):
    """
    Alpha58: COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100

    描述：
    此特征计算在过去20天内股票收盘价高于前一天的天数占比。它是衡量短期看涨趋势或资产强度的指标。
    """

    # 检查当前收盘价是否高于前一天的收盘价
    close_above_prev = (df['close'] > df['close'].shift(1)).astype(int)

    # 统计在过去20天内满足条件的天数
    count_20 = close_above_prev.rolling(window=20).sum()

    # 转换为百分比
    df['gj_058'] = (count_20 / 20) * 100

    return df['gj_058']


def gj_059(df):
    """
    Alpha59:SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)

    Alpha59: 捕捉基于收盘价与当前和前一日高/低点偏差的波动性，这取决于价格移动的方向。

    描述：
    此特征衡量收盘价与当前和前一日的高/低点的偏差量，这取决于价格运动的方向。它可以提供资产波动性的洞察。
    """

    # 计算条件性调整后的收盘价
    conditions = [
        df['close'] == df['close'].shift(1),
        df['close'] > df['close'].shift(1)
    ]
    choices = [
        0,
        df['close'] - np.minimum(df['low'], df['close'].shift(1))
    ]
    df['adjusted_close'] = np.select(conditions, choices,
                                     default=df['close'] - np.maximum(df['high'], df['close'].shift(1)))

    # 计算过去20天的总和
    df['gj_059'] = df['adjusted_close'].rolling(window=20).sum()

    return df['gj_059']


def gj_060(df):
    """
    Alpha60 SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    Alpha60: 衡量收盘价在当天总价格波动范围内的位置的加权指标。

    描述：
    此特征捕捉资产在20天期间内倾向于收盘于其高点或低点的趋势，并通过成交量进行加权。它可以提供关于资产动量和价格运动强度的洞察。
    """

    # 计算收盘价在当天范围内的位置
    df['positional_value'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])

    # 通过成交量对这个位置进行加权
    df['weighted_positional_value'] = df['positional_value'] * df['volume']

    # 计算过去20天的总和
    df['gj_060'] = df['weighted_positional_value'].rolling(window=20).sum()

    return df['gj_060']


def gj_061(df):
    """
    Alpha61(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
    80太大，实际上以22计算
    Alpha61 特征计算
    描述：
    此特征结合了来自VWAP变化的动量信息以及成交量-价格关系的动态信息。
    它首先计算VWAP变化的线性衰减加权，然后计算LOW价格与成交量移动平均值的相关性的线性衰减加权排名。
    取这两个排名值的最大值，并将结果乘以-1以调整信号方向。
    """

    # 计算VWAP的变化
    df['delta_vwap'] = df['vwap'].diff()

    # 计算VWAP变化的线性衰减加权
    decayed_vwap = decay_linear(df['delta_vwap'], 12)

    # 计算LOW价格与22周期移动平均成交量的相关性
    corr_low_volume = df['low'].rolling(window=22).corr(df['volume'])

    # 对相关性进行排名并应用线性衰减加权
    ranked_corr = corr_low_volume.rolling(window=20, min_periods=20).apply(rf, raw=True)
    decayed_corr = decay_linear(ranked_corr, 17)

    # 对线性衰减序列进行排名
    rank_decayed_vwap = decayed_vwap.rolling(window=20, min_periods=20).apply(rf, raw=True)
    rank_decayed_corr = decayed_corr.rolling(window=20, min_periods=20).apply(rf, raw=True)

    # 取两个排名值的最大值并调整信号方向
    df['gj_061'] = -1 * np.maximum(rank_decayed_vwap, rank_decayed_corr)

    return df['gj_061']


def gj_062(df):
    """
    Alpha62: (-1 * CORR(HIGH, RANK(VOLUME), 5))

    描述：
    此特征测量了在5天周期内高价和成交量排名之间的相关性。
    负相关可能意味着当高价上升时，成交量排名下降，反之亦然。
    这可能是市场情绪变化的一个标志，因为较高的价格并没有得到较高成交量排名的支持。
    此特征通过将相关性乘以-1，可能突出这些负相关性为正值。
    """

    df['ranked_volume'] = df['volume'].rolling(window=5).apply(rf, raw=True)
    df['gj_062'] = df['high'].rolling(window=5).corr(df['ranked_volume']) * -1

    return df['gj_062']


def gj_064(df):
    """
    Alpha64:
    -1 * max(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
             RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14)))

    描述：
    此特征捕捉了成交量加权平均价格（VWAP）的趋势以及以成交量为权重的收盘价趋势。
    通过结合这两种趋势，此特征旨在识别具有一致价格和成交量趋势的股票。
    """

    # 计算第一和第二个线性衰减函数的相关值
    corr1 = df['vwap'].rolling(window=5, min_periods=5).apply(rf, raw=True).rolling(window=4).corr(df['volume'].rank())
    corr2 = df['close'].rolling(window=5, min_periods=5).\
        apply(rf, raw=True).rolling(window=4).corr(df['volume'].rolling(window=60).mean().rank())

    # 计算线性衰减值
    decay1 = decay_linear(corr1, 4)
    decay2 = decay_linear(corr2.rolling(window=13).max(), 14)

    # 对线性衰减值进行排名
    rank1 = decay1.rolling(window=5, min_periods=5).apply(rf, raw=True)
    rank2 = decay2.rolling(window=5, min_periods=5).apply(rf, raw=True)

    # 结合排名值并乘以-1
    df['gj_064'] = (-1) * np.maximum(rank1, rank2)

    return df['gj_064']


def gj_063(df):
    """
    Alpha63: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100

    描述：
    此特征是一个动量指标，用于测量过去6个周期内上涨动作的强度，作为价格总体波动幅度的百分比。
    高值表示强劲的上升动力，而低值表示强劲的下降动力。
    """

    # 计算收盘价之间的正差值
    df['pos_diff'] = (df['close'] - df['close'].shift(1)).clip(lower=0)

    # 计算收盘价之间的绝对差值
    df['abs_diff'] = abs(df['close'] - df['close'].shift(1))

    # 使用自定义SMA函数计算正差值和绝对差值的SMA
    df['sma_pos_diff'] = custom_sma(df['pos_diff'], 6, 1)
    df['sma_abs_diff'] = custom_sma(df['abs_diff'], 6, 1)

    # 计算比率并乘以100
    df['gj_063'] = (df['sma_pos_diff'] / df['sma_abs_diff']) * 100

    return df['gj_063']


def gj_065(df):
    # Alpha65 MEAN(CLOSE,6)/CLOSE
    df['gj_065'] = df['close'].rolling(window=6).mean() / df['close']
    return df['gj_065']


def gj_066(df):
    """
    Alpha66: (CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) * 100

    描述：
    此特征计算当前收盘价与其6日移动平均的百分比偏差。
    正值表明股票交易价高于短期平均水平，暗示潜在的上升动力；
    负值则表明股票交易价低于短期平均水平，暗示潜在的下降动力。
    """

    # 计算CLOSE的6日移动平均
    mean_close_6 = df['close'].rolling(window=6).mean()

    # 计算百分比偏差
    df['gj_066'] = (df['close'] - mean_close_6) / mean_close_6 * 100

    return df['gj_066']
