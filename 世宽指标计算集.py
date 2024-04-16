import numpy as np
import pandas as pd


def decay_linear(data, d):
    """
    计算DECAYLINEAR
    Args:
    - data: 输入的数据 (可以是Series或DataFrame)
    - d: 窗口大小

    Returns:
    - DECAYLINEAR值 (Series或DataFrame)
    """

    # 创建权重从d递减到1
    weights = np.arange(d, 0, -1)

    # 将权重归一化，使它们的和为1
    weights = weights / weights.sum()

    # 如果输入是Series
    if isinstance(data, pd.Series):
        result = data.rolling(window=d).apply(lambda x: np.dot(x, weights), raw=True)
    # 如果输入是DataFrame
    elif isinstance(data, pd.DataFrame):
        result = data.copy()
        for column in data.columns:
            result[column] = data[column].rolling(window=d).apply(lambda x: np.dot(x, weights), raw=True)
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame.")

    return result


def rf(x):
    return (x[-1] <= x).sum()
# 调用该函数的示例语句
#    data['rank_abs_diff'] = data['abs_diff'].rolling(window=5, min_periods=5).apply(rf, raw=True)


def wq_001(data):
    """
    指标公式: ts_rank(Ts_ArgMax(SignedPower((returns < 0) ? stddev(returns, 20) : close, 2.), 5)) - 0.5
    说明: 该指标将收益率与20日的收益率标准差或收盘价进行比较，并计算其在过去5天内的最大值时序排名。
          结果从0.5中减去，可能是为了中心化数据。
    """
    data['returns_real'] = data['returns'].shift(1)
    # 计算 SignedPower
    conditions = [data['returns_real'] < 0, data['returns_real'] >= 0]
    choices = [data['returns_real'].rolling(window=20).std(), data['close']]
    data['signed_power'] = np.power(np.select(conditions, choices), 2)

    # 计算 Ts_ArgMax
    data['argmax_signed_power'] = data['signed_power'].rolling(window=5).apply(lambda x: np.argmax(x) + 1)

    # 计算 ts_rank 使用 rf 函数
    data['wq_001'] = data['argmax_signed_power'].rolling(window=5).apply(rf, raw=True) - 0.5

    return data['wq_001']


def wq_002(data):
    """
    指标公式: Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    说明: 该指标评估了两个不同时间尺度的价格动量和交易量动量之间的相关性。
    """

    # 计算 log(volume) 的两天差分
    data['log_volume_diff'] = data['volume'].apply(np.log).diff(2)

    # 计算 ((close - open) / open)
    data['price_change_ratio'] = (data['close'] - data['open']) / data['open']

    # 使用 rf 函数对步骤1和步骤2的结果进行排名
    data['rank_log_volume_diff'] = data['log_volume_diff'].rolling(window=6).apply(rf, raw=True)
    data['rank_price_change_ratio'] = data['price_change_ratio'].rolling(window=6).apply(rf, raw=True)

    # 对上述两个排名进行6天窗口的相关性计算
    data['correlation'] = data['rank_log_volume_diff'].rolling(window=6).corr(data['rank_price_change_ratio'])

    # 乘以-1
    data['wq_002'] = -1 * data['correlation']

    return data['wq_002']


def wq_003(data):
    """
    指标公式: Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    说明: 该指标评估了开盘价和交易量之间在一个10天的窗口内的相关性。
    """

    # 使用 rf 函数对 open 和 volume 进行排名
    data['rank_open'] = data['open'].rolling(window=10).apply(rf, raw=True)
    data['rank_volume'] = data['volume'].rolling(window=10).apply(rf, raw=True)

    # 对上述两个排名进行10天窗口的相关性计算
    data['correlation'] = data['rank_open'].rolling(window=10).corr(data['rank_volume'])

    # 乘以-1
    data['wq_003'] = -1 * data['correlation']

    return data['wq_003']


def wq_004(data):
    """
    指标公式: Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    说明: 该指标对过去9天的最低价进行排名，并对这些排名进行时间序列排名，然后乘以-1。
    """

    # 使用 rf 函数对 low 进行排名
    data['rank_low'] = data['low'].rolling(window=9).apply(rf, raw=True)

    # 对上述的排名进行时间序列排名
    data['ts_rank'] = data['rank_low'].rolling(window=9).apply(rf, raw=True)

    # 乘以-1
    data['wq_004'] = -1 * data['ts_rank']

    return data['wq_004']


def wq_005(data):
    """
    指标公式: Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    说明: 该指标考虑了开盘价与10天平均VWAP的差值的排名，以及收盘价与VWAP之间差异的绝对值排名，并将两者相乘。
    """

    # 计算10天的VWAP平均值
    data['avg_vwap_10'] = data['vwap'].rolling(window=10).mean()

    # 计算开盘价与10天平均VWAP的差值的排名
    data['rank_open_vwap'] = (data['open'] - data['avg_vwap_10']).rolling(window=10).apply(rf, raw=True)

    # 计算收盘价与VWAP的差值的绝对值排名
    data['abs_rank_close_vwap'] = abs(data['close'] - data['vwap']).rolling(window=10).apply(rf, raw=True)

    # 根据给定的公式计算最终的指标值
    data['wq_005'] = data['rank_open_vwap'] * (-1 * data['abs_rank_close_vwap'])

    return data['wq_005']


def wq_006(data):
    """
    指标公式: Alpha#6: (-1 * correlation(open, volume, 10))
    说明: 该指标计算了开盘价与成交量之间的10日相关性，并取其负值。
    """

    # 使用pandas内建的corr方法计算10日相关性
    data['correlation_open_volume'] = data['open'].rolling(window=10).corr(data['volume'])

    # 根据给定的公式计算最终的指标值
    data['wq_006'] = -1 * data['correlation_open_volume']

    return data['wq_006']


def wq_007(data):
    """
    指标公式: Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    说明: 该指标在20日平均成交量低于当天成交量时，使用近60日内对7日收盘价变化绝对值的时序排名乘以收盘价变化的符号。否则返回-1。
    """

    # 计算20日平均成交量
    data['adv20'] = data['volume'].rolling(window=20).mean()

    # 计算7日的收盘价变化
    data['delta_close_7'] = data['close'].diff(7)

    # 使用之前定义的rf函数计算60日的时序排名
    data['ts_rank_delta_close_7'] = data['delta_close_7'].abs().rolling(window=60).apply(rf, raw=True)

    # 根据条件计算指标值
    data['wq_007'] = np.where(data['adv20'] < data['volume'],
                              -1 * data['ts_rank_delta_close_7'] * np.sign(data['delta_close_7']),
                              -1)

    return data['wq_007']


def wq_008(data):
    """
    指标公式: Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    说明: 该指标计算近5天的开盘价之和与收益率之和的乘积，并与10天前该值的差异进行排名。
    """

    # 创建实际收益率的列
    data['returns_real'] = data['returns'].shift(1)

    # 计算5天内的开盘价和实际收益率的和
    data['sum_open_5'] = data['open'].rolling(window=5).sum()
    data['sum_returns_real_5'] = data['returns_real'].rolling(window=5).sum()

    # 计算乘积
    data['product_5'] = data['sum_open_5'] * data['sum_returns_real_5']

    # 计算10天前的乘积值
    data['delayed_product_5_10'] = data['product_5'].shift(10)

    # 计算差异
    data['product_diff'] = data['product_5'] - data['delayed_product_5_10']

    # 使用之前定义的rf函数计算排名
    data['wq_008'] = -1 * data['product_diff'].rolling(window=20).apply(rf, raw=True)

    return data['wq_008']


def wq_009(data):
    """
    指标公式: Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    说明: 该指标考虑了过去5天中收盘价的最大和最小变化来决定单日收盘价变化的方向。
    """

    # 计算close价格的单日变化
    data['delta_close_1'] = data['close'].diff(1)

    # 计算过去5天中`close`价格变化的最大和最小值
    data['ts_min_delta_5'] = data['delta_close_1'].rolling(window=5).min()
    data['ts_max_delta_5'] = data['delta_close_1'].rolling(window=5).max()

    # 根据给定的逻辑计算指标值
    conditions = [data['ts_min_delta_5'] > 0, data['ts_max_delta_5'] < 0]
    choices = [data['delta_close_1'], data['delta_close_1']]
    data['wq_009'] = np.select(conditions, choices, -1 * data['delta_close_1'])

    return data['wq_009']


def wq_010(data):
    """
    指标公式: Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    说明: 该指标考虑了过去4天中收盘价的最大和最小变化来决定单日收盘价变化的方向，并对结果进行排名。
    """

    # 计算close价格的单日变化
    data['delta_close_1'] = data['close'].diff(1)

    # 计算过去4天中`close`价格变化的最大和最小值
    data['ts_min_delta_4'] = data['delta_close_1'].rolling(window=4).min()
    data['ts_max_delta_4'] = data['delta_close_1'].rolling(window=4).max()

    # 根据给定的逻辑计算指标值
    conditions = [data['ts_min_delta_4'] > 0, data['ts_max_delta_4'] < 0]
    choices = [data['delta_close_1'], data['delta_close_1']]
    data['alpha_10_temp'] = np.select(conditions, choices, -1 * data['delta_close_1'])

    # 对结果进行排名
    data['wq_010'] = data['alpha_10_temp'].rolling(window=20).apply(rf, raw=True)

    return data['wq_010']


def wq_011(data):
    """
    指标公式: Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    说明: 该指标考虑了过去3天中(vwap - close)的最大和最小值，并结合过去3天的交易量变化进行排名。
    """

    # 计算 vwap 与 close 之间的差值
    data['vwap_close_diff'] = data['vwap'] - data['close']

    # 计算过去3天中 vwap_close_diff 的最大和最小值
    data['ts_max_vwap_close_3'] = data['vwap_close_diff'].rolling(window=3).max()
    data['ts_min_vwap_close_3'] = data['vwap_close_diff'].rolling(window=3).min()

    # 计算交易量的3天变化
    data['delta_volume_3'] = data['volume'].diff(3)

    # 使用 rf 函数对结果进行排名
    rank_ts_max = data['ts_max_vwap_close_3'].rolling(window=20).apply(rf, raw=True)
    rank_ts_min = data['ts_min_vwap_close_3'].rolling(window=20).apply(rf, raw=True)
    rank_delta_volume = data['delta_volume_3'].rolling(window=20).apply(rf, raw=True)

    # 根据给定的公式计算指标值
    data['wq_011'] = (rank_ts_max + rank_ts_min) * rank_delta_volume

    return data['wq_011']


def wq_012(data):
    """
    指标公式: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    说明: 该指标计算了交易量变化的方向与收盘价的反向变化的乘积。
    """

    # 因为您的returns已经是计算的第二天的收益率，为了避免使用未来数据，这里使用了shift。
    data['delta_volume'] = data['volume'].diff(1)
    data['delta_close'] = data['close'].diff(1)

    # 计算指标值
    data['wq_012'] = np.sign(data['delta_volume']) * (-1 * data['delta_close'])

    return data['wq_012']


def wq_013(data):
    """
    指标公式: -1 * rank(covariance(rank(close), rank(volume), 5))
    说明: 该指标计算收盘价与交易量的排名协方差的负排名值。
    """

    # 计算排名
    data['rank_close'] = data['close'].rolling(window=20).apply(rf, raw=True)
    data['rank_volume'] = data['volume'].rolling(window=20).apply(rf, raw=True)

    # 计算协方差并计算负排名
    data['cov_rank'] = data['rank_close'].rolling(window=5).cov(data['rank_volume'])
    data['wq_013'] = -1 * data['cov_rank'].rolling(window=20).apply(rf, raw=True)

    return data['wq_013']


def wq_014(data):
    """
    指标公式: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    说明: 该指标计算了收益率的3日排名变化的负排名值与开盘价和交易量的10日相关性的乘积。
    """

    # 创建实际收益率的列
    data['returns_real'] = data['returns'].shift(1)

    # 计算收益率的3日排名变化
    data['rank_delta_returns'] = data['returns_real'].rolling(window=3).apply(rf, raw=True)

    # 计算相关性
    data['correlation_open_volume'] = data['open'].rolling(window=10).corr(data['volume'])

    # 计算指标值
    data['wq_014'] = (-1 * data['rank_delta_returns']) * data['correlation_open_volume']

    return data['wq_014']


def wq_015(data):
    # 相当优秀
    """
    Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    Calculate wq_015 (Alpha#15) using a rolling window of 20 for ranking.
    """

    # Step 1: Rank high and volume over 20 days using the custom rank function
    data['rank_high'] = data['high'].rolling(window=3, min_periods=3).apply(rf, raw=True)
    data['rank_volume'] = data['volume'].rolling(window=3, min_periods=3).apply(rf, raw=True)

    # Step 2: Calculate correlation between the ranked series over 3 days
    data['correlation'] = data['rank_high'].rolling(window=3).corr(data['rank_volume'])

    # Step 3: Rank the correlation using the custom rank function over 20 days
    data['rank_correlation'] = data['correlation'].rolling(window=3, min_periods=3).apply(rf, raw=True)

    # Step 4: Sum the rank of correlation over 3 days
    data['wq_015'] = data['rank_correlation'].rolling(window=3).sum()

    return data['wq_015']


def wq_016(data):
    """
    指标公式: -1 * rank(covariance(rank(high), rank(volume), 5))
    说明: 计算高价与交易量排名的协方差的负排名值。
    """

    # 计算排名
    data['rank_high'] = data['high'].rolling(window=20).apply(rf, raw=True)
    data['rank_volume'] = data['volume'].rolling(window=20).apply(rf, raw=True)

    # 计算协方差并计算排名
    data['cov_rank'] = data['rank_high'].rolling(window=5).cov(data['rank_volume'])
    data['wq_016'] = -1 * data['cov_rank'].rolling(window=20).apply(rf, raw=True)

    return data['wq_016']


def wq_017(data):
    """
    指标公式: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    说明: 该指标是三个子指标的乘积，包括收盘价的10日时间序列排名、收盘价的1日差分的1日差分、以及交易量与20日平均交易量之比的5日时间序列排名。
    """

    # 计算时间序列排名
    data['ts_rank_close'] = data['close'].rolling(window=10).apply(rf, raw=True)
    data['ts_rank_volume_adv20'] = (data['volume'] / data['volume'].rolling(window=20).mean()).rolling(window=5).apply(
        rf, raw=True)

    # 计算差分
    data['delta_close'] = data['close'].diff(1)
    data['delta_delta_close'] = data['delta_close'].diff(1)

    # 计算排名
    data['rank_ts_rank_close'] = data['ts_rank_close'].rolling(window=20).apply(rf, raw=True)
    data['rank_delta_delta_close'] = data['delta_delta_close'].rolling(window=20).apply(rf, raw=True)
    data['rank_ts_rank_volume_adv20'] = data['ts_rank_volume_adv20'].rolling(window=20).apply(rf, raw=True)

    # 组合指标
    data['wq_017'] = (-1 * data['rank_ts_rank_close']) * data['rank_delta_delta_close'] * data[
        'rank_ts_rank_volume_adv20']

    return data['wq_017']


def wq_018(data):
    """
    指标公式: -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    说明: 该指标结合了收盘价与开盘价之间的差值的绝对值的5日标准差、收盘价与开盘价之间的差值以及收盘价与开盘价的10日相关性。
    """

    # 计算收盘价与开盘价之间的差值的绝对值的5日标准差
    data['std_abs_diff'] = data['close'].sub(data['open']).abs().rolling(window=5).std()

    # 计算收盘价与开盘价之间的差值
    data['diff_close_open'] = data['close'].sub(data['open'])

    # 计算收盘价与开盘价的10日相关性
    data['correlation_close_open'] = data['close'].rolling(window=10).corr(data['open'])

    # 将上述三个部分结合起来并计算排名
    data['wq_018'] = -1 * (data['std_abs_diff'] + data['diff_close_open'] + data['correlation_close_open']).rolling(window=20).apply(rf, raw=True)

    return data['wq_018']


def wq_019(data):
    """
    指标公式: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    说明: 该指标结合了收盘价与7天前的收盘价的差值，收盘价的7天变化，以及250天收益率的累积和的排名。
    250会使我们丧失很多数据，这里我们使用66，也即三个月
    """
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']

    # 计算close与7天前的close的差值
    data['delay_7_close'] = data['close'].shift(7)

    # 计算收盘价的7天变化
    data['delta_7_close'] = data['close'].diff(7)

    # 计算累积和的排名
    data['sum_returns_66'] = data['returns_real'].rolling(window=66).sum()
    data['rank_sum_returns'] = (1 + data['sum_returns_66']).rolling(window=20).apply(rf, raw=True)

    # 结合上述部分以计算最终的指标值
    data['wq_019'] = (-1 * (data['close'] - data['delay_7_close'] + data['delta_7_close'])
                      .apply(np.sign)) * (1 + data['rank_sum_returns'])

    return data['wq_019']


def wq_020(data):
    """
    指标公式: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    说明: 该指标结合了开盘价与1天前的高、低、收盘价的差值的排名。
    """

    # 计算开盘价与1天前的高、低、收盘价的差值
    data['delay_1_high'] = data['high'].shift(1)
    data['delay_1_low'] = data['low'].shift(1)
    data['delay_1_close'] = data['close'].shift(1)

    data['diff_open_high'] = data['open'] - data['delay_1_high']
    data['diff_open_low'] = data['open'] - data['delay_1_low']
    data['diff_open_close'] = data['open'] - data['delay_1_close']

    # 对差值进行排名
    data['rank_diff_open_high'] = data['diff_open_high'].rolling(window=20).apply(rf, raw=True)
    data['rank_diff_open_low'] = data['diff_open_low'].rolling(window=20).apply(rf, raw=True)
    data['rank_diff_open_close'] = data['diff_open_close'].rolling(window=20).apply(rf, raw=True)

    # 结合上述部分以计算最终的指标值
    data['wq_020'] = (-1 * data['rank_diff_open_high'] * data['rank_diff_open_close'] * data['rank_diff_open_low'])

    return data['wq_020']


def wq_021(data):
    """
    指标公式: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) :
              (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 :
              (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
    说明: 此指标结合了收盘价的短期和长期均值与标准差，以及交易量与20日平均交易量的关系。
    """

    # 计算收盘价的均值和标准差
    data['mean_close_8'] = data['close'].rolling(window=8).mean()
    data['stddev_close_8'] = data['close'].rolling(window=8).std()
    data['mean_close_2'] = data['close'].rolling(window=2).mean()

    # 计算交易量与20日平均交易量的比值
    data['volume_adv20_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()

    # 根据公式条件进行计算
    cond1 = (data['mean_close_8'] + data['stddev_close_8']) < data['mean_close_2']
    cond2 = data['mean_close_2'] < (data['mean_close_8'] - data['stddev_close_8'])
    cond3 = (data['volume_adv20_ratio'] > 1) | (data['volume_adv20_ratio'] == 1)

    data['wq_021'] = np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))

    return data['wq_021']


def wq_022(data):
    """
    指标公式: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    说明: 该指标结合了高价与交易量的5日相关性变化与收盘价的20日标准差的排名。
    """

    # 计算高价与交易量的5日相关性
    data['correlation_high_volume'] = data['high'].rolling(window=5).corr(data['volume'])

    # 计算相关性的5日变化
    data['delta_correlation'] = data['correlation_high_volume'].diff(5)

    # 计算收盘价的20日标准差的排名
    data['rank_stddev_close'] = data['close'].rolling(window=20).std().rolling(window=20).apply(rf, raw=True)

    # 根据公式进行计算
    data['wq_022'] = -1 * data['delta_correlation'] * data['rank_stddev_close']

    return data['wq_022']


def wq_023(data):
    """
    指标公式: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    说明: 若20日高价均值小于当前高价，则取高价的2日变化的负值，否则为0。
    """

    # 计算20日高价均值
    data['avg_high_20'] = data['high'].rolling(window=20).mean()

    # 计算高价的2日变化
    data['delta_high_2'] = data['high'].diff(2)

    # 根据公式进行计算
    data['wq_023'] = data.apply(lambda row: -1 * row['delta_high_2'] if row['avg_high_20'] < row['high'] else 0, axis=1)

    return data['wq_023']


def wq_024(data):
    """
    指标公式:
    ((((delta((sum(close, 66) / 66), 66) / delay(close, 66)) < 0.05) ||
    ((delta((sum(close, 66) / 66), 66) / delay(close, 66)) == 0.05)) ?
    (-1 * (close - ts_min(close, 66))) : (-1 * delta(close, 3)))
    说明: 根据过去66日收盘价的均值与66日前的收盘价的变化与条件进行判断，并得到最终的指标值。
    """

    # 计算66日收盘价均值
    data['avg_close_66'] = data['close'].rolling(window=66).mean()

    # 计算66日收盘价均值的66日变化
    data['delta_avg_close_66'] = data['avg_close_66'].diff(66)

    # 计算66日前的收盘价
    data['delay_close_66'] = data['close'].shift(66)

    # 计算66日最低收盘价
    data['ts_min_close_66'] = data['close'].rolling(window=66).min()

    # 计算收盘价的3日变化
    data['delta_close_3'] = data['close'].diff(3)

    # 根据条件进行计算
    data['wq_024'] = data.apply(lambda row: -1 * (row['close'] - row['ts_min_close_66']) if
                                (row['delta_avg_close_66'] / row['delay_close_66'] < 0.05 or
                                 row['delta_avg_close_66'] / row['delay_close_66'] == 0.05)
                                else -1 * row['delta_close_3'], axis=1)

    return data['wq_024']


def wq_025(data):
    """
    指标公式: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    说明: 该指标结合了收益率、20日平均日成交量、成交均价和最高价与收盘价之间的差异来计算排名。
    """
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['close'].shift()
    # 因为我们先前已经定义了'returns_real'为真实的收益率，我们直接使用它
    data['adv20'] = data['close'] * data['volume'].rolling(window=20).mean()
    data['product'] = (-1 * data['returns_real']) * data['adv20'] * data['vwap'] * (data['high'] - data['close'])

    data['wq_025'] = data['product'].rolling(window=20).apply(rf, raw=True)

    return data['wq_025']


def wq_026(data):
    """
    指标公式: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    说明: 该指标计算了交易量和最高价的5天时序排名的相关性在过去3天中的最大值的负值。
    """

    # 计算交易量和最高价的5天时序排名
    data['ts_rank_volume'] = data['volume'].rolling(window=10).apply(rf, raw=True)
    data['ts_rank_high'] = data['high'].rolling(window=10).apply(rf, raw=True)

    # 计算这两个时序排名的5天相关性
    data['correlation_rank'] = data['ts_rank_volume'].rolling(window=10).corr(data['ts_rank_high'])

    # 找到这个相关性在过去3天中的最大值
    data['ts_max_correlation'] = data['correlation_rank'].rolling(window=3).max()

    # 计算最终指标值
    data['wq_026'] = -1 * data['ts_max_correlation']

    return data['wq_026']


def wq_027(data):
    """
    指标公式: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    说明: 该指标根据交易量和VWAP的6天排名的相关性的2天累积和的排名值来决定指标值。
    """

    # 计算交易量和VWAP的排名
    data['rank_volume'] = data['volume'].rolling(window=20).apply(rf, raw=True)
    data['rank_vwap'] = data['vwap'].rolling(window=20).apply(rf, raw=True)

    # 计算这两个排名的6天相关性
    data['correlation_rank'] = data['rank_volume'].rolling(window=6).corr(data['rank_vwap'])

    # 计算这个相关性的2天累积和的一半
    data['sum_correlation'] = data['correlation_rank'].rolling(window=2).sum() / 2.0

    # 对这个值进行排名
    data['rank_sum_correlation'] = data['sum_correlation'].rolling(window=20).apply(rf, raw=True)

    # 根据排名值决定指标值
    data['wq_027'] = data['rank_sum_correlation'].apply(lambda x: -1 if x > 0.5 else 1)

    return data['wq_027']


def wq_028(data):
    """
    指标公式: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    说明: 该指标考虑了adv20与最低价的5天相关性，然后与当天的平均价相加，最后减去收盘价并进行缩放。
    """
    data['adv20'] = data['volume'].rolling(window=20).mean()
    # 计算adv20与最低价的5天相关性
    data['correlation_adv20_low'] = data['adv20'].rolling(window=5).corr(data['low'])

    # 计算当天的平均价
    data['average_price'] = (data['high'] + data['low']) / 2

    # 结合上述部分并减去收盘价
    data['unscaled_wq_028'] = data['correlation_adv20_low'] + data['average_price'] - data['close']

    # 缩放这个值
    data['wq_028'] = (data['unscaled_wq_028'] / abs(data['unscaled_wq_028']).sum())

    return data['wq_028']


def wq_029(data):
    """
    指标公式: min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5)
    说明: 该指标结合了多个排名、尺度变换、对数、产品、时间序列最小值和延迟。
    """

    # 计算收益率
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']

    # 分解指标公式
    data['delta_close'] = data['close'].diff(5)
    data['rank_delta'] = data['delta_close'].rolling(window=20).apply(rf, raw=True)
    data['double_ranked'] = data['rank_delta'].rolling(window=20).apply(rf, raw=True)
    data['min_double_ranked'] = data['double_ranked'].rolling(window=2).min()
    data['scaled_log'] = np.log(data['min_double_ranked'])
    data['scaled'] = (data['scaled_log'] - data['scaled_log'].min()) / (
                data['scaled_log'].max() - data['scaled_log'].min())
    data['product_term'] = data['scaled'].rolling(window=5).apply(lambda x: np.prod(x), raw=True)

    # 结合公式的各部分
    data['ts_rank'] = data['returns_real'].shift(6).rolling(window=5).apply(rf, raw=True)
    data['wq_029'] = data['product_term'] + data['ts_rank']

    return data['wq_029']


def wq_030(data):
    """
    指标公式: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +
    sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    说明: 该指标结合了连续三天收盘价的变化方向与成交量的关系。
    """

    # 计算连续差值并取其符号
    data['sign_diff_1'] = (data['close'] - data['close'].shift(1)).apply(np.sign)
    data['sign_diff_2'] = (data['close'].shift(1) - data['close'].shift(2)).apply(np.sign)
    data['sign_diff_3'] = (data['close'].shift(2) - data['close'].shift(3)).apply(np.sign)

    # 对这些符号求和
    data['sign_sum'] = data['sign_diff_1'] + data['sign_diff_2'] + data['sign_diff_3']

    # 计算上述值的排名
    data['rank_sign_sum'] = data['sign_sum'].rolling(window=20).apply(rf, raw=True)

    # 计算5天和20天的成交量之和
    data['sum_vol_5'] = data['volume'].rolling(window=5).sum()
    data['sum_vol_20'] = data['volume'].rolling(window=20).sum()

    # 使用上述步骤得到的结果计算最终的指标值
    data['wq_030'] = ((1.0 - data['rank_sign_sum']) * data['sum_vol_5']) / data['sum_vol_20']

    return data['wq_030']


def wq_031(data):
    """
    指标公式: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *
    delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    说明: 该指标结合了多个排名、收盘价的差值、线性衰减和成交量与最低价的相关性。
    """
    data['adv20'] = data['volume'].rolling(window=20).mean()
    # 计算收盘价的10日差值，并进行连续排名
    data['delta_close_10'] = data['close'].diff(10)
    data['rank_delta_close_10'] = data['delta_close_10'].rolling(window=20).apply(rf, raw=True)
    data['double_rank_delta_close_10'] = data['rank_delta_close_10'].rolling(window=20).apply(rf, raw=True)

    # 对上述结果应用线性衰减函数
    data['decay'] = decay_linear(-1 * data['double_rank_delta_close_10'], 10)
    data['rank_decay'] = data['decay'].rolling(window=20).apply(rf, raw=True)

    # 计算收盘价的3日差值
    data['delta_close_3'] = -1 * data['close'].diff(3)
    data['rank_delta_close_3'] = data['delta_close_3'].rolling(window=20).apply(rf, raw=True)

    # 使用adv20和最低价计算12日相关性
    data['correlation_adv20_low'] = data['adv20'].rolling(window=12).corr(data['low'])

    # 将所有结果组合在一起，得到最终的指标值
    data['wq_031'] = data['rank_decay'] + data['rank_delta_close_3'] + data['correlation_adv20_low'].apply(np.sign)

    return data['wq_031']


def wq_032(data):
    """
    指标公式: scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))
    说明: 该指标结合了近7天收盘价的均值与当天收盘价的差，以及vwap与5天前的close在230天窗口内的相关性。
    230太多，改成66.
    """

    # 计算7天的close均值与close的差，并进行标准化
    data['mean_close_7'] = data['close'].rolling(window=7).mean()
    data['diff_mean_close'] = data['mean_close_7'] - data['close']
    data['scaled_diff_mean_close'] = (data['diff_mean_close'] - data['diff_mean_close'].mean()) / data[
        'diff_mean_close'].std()

    # 计算vwap与5天前的close的230天相关性，并进行标准化
    data['delay_close_5'] = data['close'].shift(5)
    data['correlation_vwap_delayclose'] = data['vwap'].rolling(window=66).corr(data['delay_close_5'])
    data['scaled_correlation_vwap_delayclose'] = (data['correlation_vwap_delayclose'] - data[
        'correlation_vwap_delayclose'].mean()) / data['correlation_vwap_delayclose'].std()

    # 计算最终的指标值
    data['wq_032'] = data['scaled_diff_mean_close'] + 20 * data['scaled_correlation_vwap_delayclose']

    return data['wq_032']


def wq_033(data):
    """
    指标公式: rank((-1 * ((1 - (open / close))^1)))
    说明: 该指标基于开盘价和收盘价之间的比值来进行排名。
    """

    # 计算开盘价与收盘价的比值的差距与1的差值
    data['open_close_gap'] = (1 - (data['open'] / data['close']))

    # 对差值进行排名
    data['wq_033'] = (-1 * data['open_close_gap']).rolling(window=20).apply(rf, raw=True)

    return data['wq_033']


def wq_034(data):
    """
    指标公式: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    说明: 该指标结合了短期与长期收益率波动性的比值的排名的补数和收盘价与前一日收盘价差的排名的补数。
    """
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']
    # 计算2日与5日收益率的标准差
    data['stddev_returns_2'] = data['returns_real'].rolling(window=2).std()
    data['stddev_returns_5'] = data['returns_real'].rolling(window=5).std()

    # 计算标准差的比值的排名的补数
    data['stddev_ratio_rank'] = (data['stddev_returns_2'] / data['stddev_returns_5']).rolling(window=20).apply(rf, raw=True)
    data['stddev_ratio_complement'] = 1 - data['stddev_ratio_rank']

    # 计算收盘价与前一日收盘价差的排名的补数
    data['delta_close'] = data['close'].diff(1)
    data['delta_close_complement'] = 1 - data['delta_close'].rolling(window=20).apply(rf, raw=True)

    # 结合上述部分以计算最终的指标值
    data['wq_034'] = data['stddev_ratio_complement'] + data['delta_close_complement']

    return data['wq_034']


def wq_035(data):
    """
    指标公式: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    说明: 该指标结合了过去32日的交易量排名、过去16日的(close + high - low)的排名的补数和过去32日的收益率排名的补数的乘积。
    """
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']
    # 计算各部分的时间序列排名
    data['ts_rank_volume'] = data['volume'].rolling(window=32).apply(rf, raw=True)
    data['ts_rank_close_high_low'] = (data['close'] + data['high'] - data['low']).rolling(window=16).apply(rf, raw=True)
    data['ts_rank_returns'] = data['returns_real'].rolling(window=32).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_035'] = data['ts_rank_volume'] * (1 - data['ts_rank_close_high_low']) * (1 - data['ts_rank_returns'])

    return data['wq_035']


def wq_036(data):
    """
    指标公式: Alpha#36Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open
    - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6))))
    + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    说明: 该指标结合了多个部分的加权和，包括与延迟交易量的相关性、开盘价与收盘价之差、时间序列排名、vwap与adv20的相关性等。
    """
    # 计算实际收益率
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']
    # 计算adv20
    data['adv20'] = data['volume'].rolling(window=20).mean()

    # 计算各部分
    data['correlation_close_open_volume'] = data['close'].sub(data['open']).rolling(window=15).corr(data['volume'].shift(1))
    data['rank_correlation_close_open_volume'] = data['correlation_close_open_volume'].rolling(window=20).apply(rf, raw=True)

    data['rank_open_close'] = (data['open'] - data['close']).rolling(window=20).apply(rf, raw=True)

    data['ts_rank_delay_returns'] = (-1 * data['returns_real'].shift(6)).rolling(window=5).apply(rf, raw=True)

    data['correlation_vwap_adv20'] = data['vwap'].rolling(window=6).corr(data['adv20'])
    data['rank_abs_correlation_vwap_adv20'] = abs(data['correlation_vwap_adv20']).rolling(window=20).apply(rf, raw=True)

    data['rank_sum_close_open'] = ((data['close'].rolling(window=200).mean() - data['open']) * (data['close'] - data['open'])).rolling(window=20).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_036'] = (2.21 * data['rank_correlation_close_open_volume']
                      + 0.7 * data['rank_open_close']
                      + 0.73 * data['ts_rank_delay_returns']
                      + data['rank_abs_correlation_vwap_adv20']
                      + 0.6 * data['rank_sum_close_open'])

    return data['wq_036']


def wq_037(data):
    """
    指标公式: Alpha#37
    Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    说明: 该指标考虑了开盘价与收盘价的差值与其一天的延迟值的相关性，再加上开盘价与收盘价的差值的排名。
    200太大，改成132
    """

    # 计算实际收益率
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']

    # 计算相关性
    data['correlation_open_close'] = data['open'].sub(data['close']).shift(1).rolling(window=132).corr(data['close'])
    data['rank_correlation_open_close'] = data['correlation_open_close'].rolling(window=20).apply(rf, raw=True)

    # 计算开盘价与收盘价的差值的排名
    data['rank_open_minus_close'] = (data['open'] - data['close']).rolling(window=20).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_037'] = data['rank_correlation_open_close'] + data['rank_open_minus_close']

    return data['wq_037']


def wq_038(data):
    """
    指标公式: Alpha#38
    Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    说明: 该指标考虑了收盘价的10日排名与收盘价与开盘价的比值的排名的乘积。
    """

    # 计算实际收益率
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']

    # 计算收盘价的10日排名
    data['rank_ts_close'] = data['close'].rolling(window=10).apply(rf, raw=True)

    # 计算收盘价与开盘价的比值的排名
    data['rank_close_over_open'] = (data['close'] / data['open']).rolling(window=20).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_038'] = -1 * data['rank_ts_close'] * data['rank_close_over_open']

    return data['wq_038']


def wq_039(data):
    """
    指标公式: Alpha#39
    Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +
    rank(sum(returns, 250))))
    说明: 该指标考虑了收盘价的7日变化、交易量与adv20的9日线性衰减，以及250日的收益率的排名。
    250太大，改为132
    """
    data['adv20'] = data['volume'].rolling(window=20).mean()
    data['decay'] = decay_linear(data['volume'] / data['adv20'], 9)
    data['returns_real'] = (data['close'] - data['close'].shift()) / data['open']

    # 计算最终的指标值
    data['wq_039'] = (-1 * data['close'].diff(7) * (1 - data['decay'].rolling(window=20).apply(rf, raw=True))) * (
                1 + data['returns_real'].rolling(window=250).sum())

    return data['wq_039']


def wq_040(data):
    """
    指标公式: Alpha#40
    Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    说明: 该指标考虑了高价的10日标准差与高价和交易量的10日相关性。
    """
    data['std_high_10'] = data['high'].rolling(window=10).std()
    data['correlation_high_volume'] = data['high'].rolling(window=10).corr(data['volume'])

    # 计算最终的指标值
    data['wq_040'] = -1 * data['std_high_10'].rolling(window=20).apply(rf, raw=True) * data['correlation_high_volume']

    return data['wq_040']


def wq_041(data):
    """
    指标公式: Alpha#41
    Alpha#41: (((high * low)^0.5) - vwap)
    说明: 该指标考虑了高价和低价的几何平均数与vwap的差。
    """
    data['geom_mean'] = np.sqrt(data['high'] * data['low'])

    # 计算最终的指标值
    data['wq_041'] = data['geom_mean'] - data['vwap']

    return data['wq_041']


def wq_042(data):
    """
    指标公式: Alpha#42
    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    说明: 该指标考虑了vwap与收盘价的差的排名与vwap与收盘价的和的排名的比值。
    """
    data['rank_diff'] = (data['vwap'] - data['close']).rolling(window=20).apply(rf, raw=True)
    data['rank_sum'] = (data['vwap'] + data['close']).rolling(window=20).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_042'] = data['rank_diff'] / data['rank_sum']

    return data['wq_042']


def wq_043(data):
    """
    指标公式: Alpha#43
    Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    说明: 该指标考虑了交易量与adv20的20日排名与收盘价的7日变化的8日排名的乘积。
    """
    data['adv20'] = data['volume'].rolling(window=20).mean()
    data['ts_rank_volume'] = (data['volume'] / data['adv20']).rolling(window=20).apply(rf, raw=True)
    data['ts_rank_close'] = data['close'].diff(7).rolling(window=8).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_043'] = data['ts_rank_volume'] * data['ts_rank_close']

    return data['wq_043']


def wq_044(data):
    """
    指标公式: (-1 * correlation(high, rank(volume), 5))
    说明: 该指标计算了高价与交易量排名的5日相关性的负值。
    因计算缘故，将5日相关性改为20日相关性
    """
    # 计算volume的排名
    data['rank_volume'] = data['volume'].rolling(window=20).apply(rf, raw=True)

    # 计算高价与交易量排名的20日相关性
    data['correlation_high_rank_volume'] = data['high'].rolling(window=20).corr(data['rank_volume'])

    # 计算指标值
    data['wq_044'] = -1 * data['correlation_high_rank_volume']

    return data['wq_044']


def wq_045(data):
    """
    指标公式: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
    rank(correlation(sum(close, 5), sum(close, 20), 2))))
    因计算缘故，将2日相关性改为22日相关性
    """
    # 计算delay(close, 5)的20日累计和
    data['sum_delay_close_20'] = data['close'].shift(5).rolling(window=20).sum() / 20

    # 计算close和volume的22日相关性
    data['correlation_close_volume_2'] = data['close'].rolling(window=22).corr(data['volume'])

    # 计算close的5日累计和和20日累计和的22日相关性
    data['sum_close_5'] = data['close'].rolling(window=5).sum()
    data['sum_close_20'] = data['close'].rolling(window=20).sum()
    data['correlation_sum_close_5_20_2'] = data['sum_close_5'].rolling(window=22).corr(data['sum_close_20'])

    # 计算指标值
    data['wq_045'] = -1 * (data['sum_delay_close_20'].rolling(window=20).apply(rf, raw=True) *
                           data['correlation_close_volume_2'] *
                           data['correlation_sum_close_5_20_2'].rolling(window=20).apply(rf, raw=True))

    return data['wq_045']


def wq_046(data):
    """
    Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
    (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
    (-1 * 1) * (close - delay(close, 1))))
    """
    temp1 = (data['close'].shift(20) - data['close'].shift(10)) / 10
    temp2 = (data['close'].shift(10) - data['close']) / 10

    condition1 = (0.25 < (temp1 - temp2))
    condition2 = (temp1 - temp2) < 0

    data['wq_046'] = np.where(condition1, -1, np.where(condition2, 1, -1 * (data['close'] - data['close'].shift(1))))

    return data['wq_046']


def wq_047(data):
    """
    Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /
    5))) - rank((vwap - delay(vwap, 5))))
    """
    # 计算 adv20
    data['adv20'] = data['volume'].rolling(window=20).mean()

    # 计算其他部分
    data['rank_inv_close'] = data['close'].rolling(window=20).apply(rf, raw=True)
    data['rank_high_minus_close'] = (data['high'] - data['close']).rolling(window=20).apply(rf, raw=True)

    term1 = data['rank_inv_close'] * data['volume'] / data['adv20']
    term2 = (data['high'] * data['rank_high_minus_close']) / (data['high'].rolling(window=5).sum() / 5)
    term3 = data['vwap'].diff(5).rolling(window=20).apply(rf, raw=True)

    data['wq_047'] = term1 * term2 - term3

    return data['wq_047']

# wq_048在期货市场中没有对应，不再实现


def wq_049(data):
    """
    Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
    0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    temp1 = (data['close'].shift(20) - data['close'].shift(10)) / 10
    temp2 = (data['close'].shift(10) - data['close']) / 10

    condition = (temp1 - temp2) < (-1 * 0.1)

    data['wq_049'] = np.where(condition, 1, -1 * (data['close'] - data['close'].shift(1)))

    return data['wq_049']


def wq_050(data):
    """
    Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    """
    data['rank_volume'] = data['volume'].rolling(window=20).apply(rf, raw=True)
    data['rank_vwap'] = data['vwap'].rolling(window=20).apply(rf, raw=True)

    data['correlation_rank_volume_vwap'] = data['rank_volume'].rolling(window=5).corr(data['rank_vwap'])

    data['wq_050'] = -1 * data['correlation_rank_volume_vwap'].rolling(window=5).max()

    return data['wq_050']


def wq_051(data):
    """
    Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
    0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    condition = ((data['close'].shift(20) - data['close'].shift(10)) / 10 - (
                data['close'].shift(10) - data['close']) / 10) < -0.05
    data['wq_051'] = np.where(condition, 1, (-1 * (data['close'] - data['close'].shift(1))))

    return data['wq_051']


def wq_052(data):
    """
    Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -
    sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    240太大，修改为132
    """
    term1 = -data['low'].rolling(window=5).min() + data['low'].rolling(window=5).min().shift(5)
    term2 = ((data['returns'].rolling(window=132).sum() - data['returns'].rolling(window=20).sum()) / 220).rolling(
        window=20).apply(rf, raw=True)
    term3 = data['volume'].rolling(window=5).apply(rf, raw=True)
    data['wq_052'] = term1 * term2 * term3

    return data['wq_052']


def wq_053(data):
    """
    Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    """
    data['wq_053'] = -1 * (((data['close'] - data['low']) - (data['high'] - data['close'])) / (
                data['close'] - data['low'])).diff(9)

    return data['wq_053']


def wq_054(data):
    """
    Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    """
    data['wq_054'] = (-1 * (data['low'] - data['close']) * (data['open'] ** 5)) / (
                (data['low'] - data['high']) * (data['close'] ** 5))

    return data['wq_054']


def wq_055(data):
    """
    Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,
    12)))), rank(volume), 6))
    由于计算问题，将相关天数改为22
    """
    rank_term1 = ((data['close'] - data['low'].rolling(window=12).min()) /
                  (data['high'].rolling(window=12).max() - data['low'].rolling(window=12).min())).rolling(
        window=20).apply(rf, raw=True)
    rank_term2 = data['volume'].rolling(window=20).apply(rf, raw=True)
    data['wq_055'] = -1 * rank_term1.rolling(window=22).corr(rank_term2)

    return data['wq_055']

# wq_056在期货市场中没有对应，不再实现


def wq_057(data):
    """
    Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    """
    # Calculate position of max close over past 30 days
    data['argmax_close_30'] = data['close'].rolling(window=30).apply(np.argmax, raw=True) + 1
    data['rank_argmax_close_30'] = data['argmax_close_30'].rolling(window=20).apply(rf, raw=True)

    # Apply decay linear on rank of argmax close
    data['decay_linear_rank'] = decay_linear(data['rank_argmax_close_30'], 2)

    # Calculate Alpha#57
    data['wq_057'] = 0 - (1 * (data['close'] - data['vwap']) / data['decay_linear_rank'])

    return data['wq_057']

# wq_058在期货市场中没有对应，不再实现
# wq_059在期货市场中没有对应，不再实现


def wq_060(data):
    """
    Alpha#60: 0 - (1 * ((2 * scale(rank((((close - low) - (high - close)) / (high - low)) * volume)) - scale(rank(ts_argmax(close, 10)))))
    """
    # Calculate inner term
    data['inner_term'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])

    # Rank and scale the term
    data['ranked_term'] = data['inner_term'].rolling(window=20).apply(rf, raw=True)
    data['scaled_ranked_term'] = (data['ranked_term'] * data['volume'] - data['ranked_term'].mean()) / data[
        'ranked_term'].std()

    # ts_argmax for close
    data['argmax_close_10'] = data['close'].rolling(window=10).apply(np.argmax, raw=True) + 1
    data['rank_argmax_close_10'] = data['argmax_close_10'].rolling(window=20).apply(rf, raw=True)
    data['scaled_rank_argmax'] = (data['rank_argmax_close_10'] - data['rank_argmax_close_10'].mean()) / data[
        'rank_argmax_close_10'].std()

    # Calculate Alpha#60
    data['wq_060'] = 0 - (1 * ((2 * data['scaled_ranked_term']) - data['scaled_rank_argmax']))

    return data['wq_060']


def wq_061(data):
    """
    Alpha#61: rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282))
    """
    # 计算滚动的最小 vwap 值
    data['ts_min_vwap_16'] = data['vwap'].rolling(window=int(16.1219)).min()

    # 计算 adv180
    data['adv180'] = data['volume'].rolling(window=180).mean()

    # 计算vwap和adv180的17.9282日相关系数
    data['corr_vwap_adv180'] = data['vwap'].rolling(window=int(17.9282)).corr(data['adv180'])

    # 将结果分别进行排名
    data['rank_vwap_diff'] = data['vwap'].sub(data['ts_min_vwap_16']).rolling(window=20).apply(rf, raw=True)
    data['rank_corr_vwap_adv180'] = data['corr_vwap_adv180'].rolling(window=20).apply(rf, raw=True)

    # 根据条件返回结果
    data['wq_061'] = (data['rank_vwap_diff'] < data['rank_corr_vwap_adv180']).astype(int)

    return data['wq_061']


def wq_062(data):
    # 62暂时跳过
    """
    Alpha#62: rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open))
    < (rank(((high + low) / 2)) + rank(high))) * -1
    """
    # 计算 adv20
    data['adv20'] = data['volume'].rolling(window=20).mean()

    # 计算vwap和adv20的9.91009日相关系数
    data['corr_vwap_adv20'] = data['vwap'].rolling(window=10).corr(data['adv20'].rolling(window=22).sum())

    # 将结果分别进行排名
    data['rank_corr_vwap_adv20'] = data['corr_vwap_adv20'].rolling(window=20).apply(rf, raw=True)
    data['rank_open'] = data['open'].rolling(window=20).apply(rf, raw=True)
    data['rank_high_low_avg'] = ((data['high'] + data['low']) / 2).rolling(window=20).apply(rf, raw=True)
    data['rank_high'] = data['high'].rolling(window=20).apply(rf, raw=True)

    # 根据条件返回结果
    condition = (2 * data['rank_open'] < data['rank_high_low_avg'] + data['rank_high'])
    data['wq_062'] = (data['rank_corr_vwap_adv20'] < condition).astype(int) * -1

    return data['wq_062']
# wq_063在期货市场中没有对应，不再实现


def wq_064(data):
    """
    Alpha#64: rank(correlation(sum((open * 0.178404 + low * (1 - 0.178404)), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((high + low) / 2) * 0.178404 + vwap * (1 - 0.178404), 3.69741)) * -1
    """
    # 计算 adv120
    data['adv120'] = data['volume'].rolling(window=120).mean()

    # 计算权重加权后的 open 和 low 的和
    data['weighted_open_low'] = data['open'] * 0.178404 + data['low'] * (1 - 0.178404)

    # 计算 correlation
    data['corr_weighted_open_low_adv120'] = data['weighted_open_low'].rolling(window=int(12.7054)).sum().rolling(
        window=int(16.6208)).corr(data['adv120'].rolling(window=int(12.7054)).sum())

    # 计算 delta
    data['delta_vwap_high_low_avg'] = (
                ((data['high'] + data['low']) / 2) * 0.178404 + data['vwap'] * (1 - 0.178404)).diff(int(3.69741))

    # 将结果分别进行排名
    data['rank_corr_weighted_open_low_adv120'] = data['corr_weighted_open_low_adv120'].rolling(window=20).apply(rf,
                                                                                                                raw=True)
    data['rank_delta_vwap_high_low_avg'] = data['delta_vwap_high_low_avg'].rolling(window=20).apply(rf, raw=True)

    # 根据条件返回结果
    data['wq_064'] = (data['rank_corr_weighted_open_low_adv120'] < data['rank_delta_vwap_high_low_avg']).astype(
        int) * -1

    return data['wq_064']


def wq_065(data):
    """
    Alpha#65: rank(correlation(open * 0.00817205 + vwap * (1 - 0.00817205), sum(adv60, 8.6911), 6.40374)) < rank(open - ts_min(open, 13.635)) * -1
    """
    # 计算 adv60
    data['adv60'] = data['volume'].rolling(window=60).mean()

    # 计算权重加权后的 open 和 vwap 的和
    data['weighted_open_vwap'] = data['open'] * 0.00817205 + data['vwap'] * (1 - 0.00817205)

    # 计算 correlation
    data['corr_weighted_open_vwap_adv60'] = data['weighted_open_vwap'].rolling(window=int(6.40374)).corr(
        data['adv60'].rolling(window=int(8.6911)).sum())

    # 计算 ts_min
    data['ts_min_open_13'] = data['open'].rolling(window=int(13.635)).min()

    # 将结果分别进行排名
    data['rank_corr_weighted_open_vwap_adv60'] = data['corr_weighted_open_vwap_adv60'].rolling(window=20).apply(rf,
                                                                                                                raw=True)
    data['rank_open_ts_min'] = (data['open'] - data['ts_min_open_13']).rolling(window=20).apply(rf, raw=True)

    # 根据条件返回结果
    data['wq_065'] = (data['rank_corr_weighted_open_vwap_adv60'] < data['rank_open_ts_min']).astype(int) * -1

    return data['wq_065']


def wq_066(data):
    """
    Alpha#066: (-1 * ((CLOSE - DELTA(CLOSE, 1)) < 0) * TS_RANK(DECAYLINEAR(DELTA(VWAP, 4),7), 16))
    """
    # 计算CLOSE的一阶差分
    data['delta_close'] = data['close'].diff(1)

    # 计算VWAP的四阶差分
    data['delta_vwap'] = data['vwap'].diff(4)

    # 计算DECAYLINEAR
    data['decay_delta_vwap'] = decay_linear(data['delta_vwap'], 7)

    # 计算TS_RANK
    data['ts_rank_decay_delta_vwap'] = data['decay_delta_vwap'].rolling(window=16, min_periods=16).apply(rf, raw=True)

    # 将条件与TS_RANK结合
    data['wq_066'] = -1 * np.where(data['delta_close'] < 0, 1, 0) * data['ts_rank_decay_delta_vwap']

    return data['wq_066']


def wq_068(data):
    """
    Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <
    rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    """
    # Calculate adv15 - the average daily trading volume over the past 15 days
    adv15 = data['volume'].rolling(window=15).mean()
    data['high_rank'] = data['high'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Calculate the correlation
    corr = data['high_rank'].rolling(window=9).corr(adv15.rolling(window=22, min_periods=22).apply(rf, raw=True))

    # Calculate Ts_Rank for the correlation
    ts_rank_corr = corr.rolling(window=16, min_periods=16).apply(rf, raw=True)

    # Calculate the delta
    delta_val = ((data['close'] * 0.518371) + (data['low'] * (1 - 0.518371))).diff(1)

    # Final calculation
    data['wq_068'] = np.where(ts_rank_corr < delta_val.rank(), 1, 0) * -1

    return data['wq_068']

# 69无法实现


def wq_070(data):
    """
    Alpha#70:
    ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(close, adv50, 17.8256), 17.9171)) * -1)
    """
    data['adv50'] = data['volume'].rolling(window=50).mean()
    data['delta_vwap'] = data['vwap'].diff(2)  # approximating 1.29456 as 2 for simplicity
    data['corr_close_adv50'] = data['close'].rolling(window=18).corr(
        data['adv50'])  # approximating 17.8256 as 18 for simplicity
    data['ts_rank_corr'] = data['corr_close_adv50'].rolling(window=18, min_periods=18).apply(rf, raw=True)
    # approximating 17.9171 as 18 for simplicity
    data['wq_070'] = -1 * (data['delta_vwap'].rolling(window=22, min_periods=22).apply(rf, raw=True) **
                           data['ts_rank_corr'])

    return data['wq_070']


def wq_071(data):
    """
    指标公式: Alpha#71
    Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,
    12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +
    vwap)))^2), 16.4662), 4.4388))
    说明: 该指标考虑了股价的时间序列排名、交易量与adv180的时间序列相关性的线性衰减、以及开盘价、最低价与vwap之间的关系。
    """
    # 计算adv180 adv180太大，改为66
    data['adv66'] = data['volume'].rolling(window=66).mean()

    # 计算时间序列排名
    data['ts_rank_close'] = data['close'].rolling(window=3).apply(rf, raw=True)
    data['ts_rank_adv180'] = data['adv66'].rolling(window=12).apply(rf, raw=True)

    # 计算时间序列相关性
    correlation_part = data[['ts_rank_close', 'ts_rank_adv180']].\
        rolling(window=18).apply(lambda x: np.corrcoef(x.reshape(-1, 2).T)[0, 1], raw=True)

    # 计算线性衰减值
    decayed_correlation = decay_linear(correlation_part, 4)

    # 计算时间序列排名
    data['ts_rank_decayed_correlation'] = decayed_correlation.iloc[:, 0].rolling(window=16).apply(rf, raw=True)

    # 计算第二部分的值
    data['low_open_vwap'] = ((data['low'] + data['open']) - (2 * data['vwap'])) ** 2
    data['ranked_low_open_vwap'] = data['low_open_vwap'].rank()
    decayed_low_open_vwap = decay_linear(data['ranked_low_open_vwap'], 16)
    data['ts_rank_decayed_low_open_vwap'] = decayed_low_open_vwap.rolling(window=4).apply(rf, raw=True)

    # 计算最终的指标值
    data['wq_071'] = data[['ts_rank_decayed_correlation', 'ts_rank_decayed_low_open_vwap']].max(axis=1)

    return data['wq_071']


def wq_072(data):
    # First, calculate necessary simple metrics
    # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
    # rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),
    # 2.95011)))
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['adv40'] = data['volume'].rolling(window=40).mean()

    # First correlation part
    data['correlation_part1'] = data['mid_price'].rolling(window=int(8.93345)).corr(data['adv40'])
    data['decayed_correlation1'] = decay_linear(data['correlation_part1'], int(10.1519))

    # Second correlation part
    data['vwap_rank'] = data['vwap'].rolling(window=22).apply(rf, raw=True)
    data['volume_rank'] = data['volume'].rolling(window=22).apply(rf, raw=True)
    data['correlation_part2'] = data['vwap_rank'].rolling(window=int(6.86671)).corr(data['volume_rank'])
    data['decayed_correlation2'] = decay_linear(data['correlation_part2'], int(2.95011))

    # Final calculation using ts_rank
    data['wq_072'] = data['decayed_correlation1'].rolling(window=22).apply(rf, raw=True) / data[
        'decayed_correlation2'].rolling(window=22).apply(rf, raw=True)
    return data['wq_072']


def wq_073(data):
    """
    Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *
    0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)

    Logic:
    1. Calculate the change in VWAP over a period.
    2. Apply a linear decay to the change in VWAP.
    3. Modify the open and low prices using given weights.
    4. Compute the change in the combined weighted price.
    5. Apply a linear decay to the negative change in the combined weighted price.
    6. Rank the values obtained from steps 2 and 5.
    7. Take the maximum rank of the values obtained from the two ranks.
    8. Multiply the result by -1.

    Returns:
    Data series of the computed Alpha#73 values.
    """

    # Calculate delta for vwap
    data['delta_vwap'] = data['vwap'].diff(int(4.72775))

    # Apply a linear decay to the change in VWAP
    decayed_delta_vwap = decay_linear(data['delta_vwap'], int(2.91864))

    # Modify the open and low prices using the given weights
    data['mod_open'] = data['open'] * 0.147155
    data['mod_low'] = data['low'] * (1 - 0.147155)
    data['combined'] = data['mod_open'] + data['mod_low']

    # Compute the change in the combined weighted price
    data['delta_combined'] = data['combined'].diff(int(2.03608)) / data['combined']

    # Apply a linear decay to the negative change in the combined weighted price
    decayed_delta_combined = decay_linear(data['delta_combined'] * -1, int(3.33829))

    # Calculate the time series rank for the decayed values
    ts_rank_decayed_delta_vwap = decayed_delta_vwap.rolling(window=22).apply(rf, raw=True)
    ts_rank_decayed_delta_combined = decayed_delta_combined.rolling(window=22).apply(rf, raw=True)

    # Take the maximum rank of the values obtained from the two ranks
    data['wq_073'] = np.maximum(ts_rank_decayed_delta_vwap, ts_rank_decayed_delta_combined) * -1

    return data['wq_073']


def wq_074(data):
    """
    Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661)
    + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    Alpha#74: Compares the time series rank of the correlation of close prices with the sum of the past 37.4843 days'
    average traded volume against the time series rank of the correlation of a weighted sum of high prices and vwap
    with volume.


    Args:
    data (pd.DataFrame): DataFrame containing 'close', 'adv30', 'high', 'vwap', and 'volume' columns.

    Returns:
    pd.Series: Alpha#74 values.
    """

    # 计算过去37.4843天的平均交易量之和
    data['adv30'] = data['volume'].rolling(window=30).sum()
    data['adv30_sum'] = data['adv30'].rolling(window=37).sum()

    # 计算close和adv30_sum的15.1365天窗口内的相关性
    correlation_part1 = data['close'].rolling(window=15).corr(data['adv30_sum'])

    # 为(high * 0.0261661) + (vwap * (1 - 0.0261661))与volume计算11.4791天窗口内的相关性
    expression = (data['high'] * 0.0261661) + (data['vwap'] * (1 - 0.0261661))
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=20).apply(rf, raw=True)
    correlation_part2 = expression.rolling(window=11).corr(data['volume_rank'])

    # 比较两个排名
    data['wq_074'] = np.where(correlation_part1 < correlation_part2, data['volume_rank'], 11.4791)

    return data['wq_074']


def wq_075(data):
    """
    Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))

    Arguments:
    data : pd.DataFrame : a DataFrame with ['vwap', 'volume', 'low']

    Returns:
    pd.Series : a pd.Series contains boolean values
    """

    # Step 1: Calculate the correlation of vwap and volume with window 4.24304
    corr_vwap_volume = data['vwap'].rolling(window=4).corr(data['volume'])

    # Step 2: Rank of low
    rank_low = data['low'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 3: Calculate adv50
    adv50 = data['volume'].rolling(window=50).mean()

    # Step 4: Rank of adv50
    rank_adv50 = adv50.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 5: Calculate the correlation of rank_low and rank_adv50 with window 12.4413
    corr_rank_low_adv50 = rank_low.rolling(window=12).corr(rank_adv50)

    # Step 6: Compare the rank of two correlations
    result = corr_vwap_volume.rank() < corr_rank_low_adv50.rank()
    data['wq_075'] = np.where(result, rank_adv50, 12.4413)

    return data['wq_075']


# wq_076由于在期货市场中无对应，不再实现。
def wq_077(data):
    """
    Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    rank(decay_linear(correlation(((high + low) / 2), adv40), 3.1614), 5.64125)))

    Arguments:
    data : pd.DataFrame : a DataFrame with ['high', 'low', 'vwap', 'volume']

    Returns:
    pd.Series : a pd.Series contains the result of the factor calculation
    """

    # Step 1: Calculate ((((high + low) / 2) + high) - (vwap + high))
    combined_value = (((data['high'] + data['low']) / 2) + data['high']) - (data['vwap'] + data['high'])

    # Step 2: Apply decay_linear and rank
    rank1 = decay_linear(combined_value, 20).rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 3: Calculate correlation with adv40
    adv40 = data['volume'].rolling(window=40).mean()
    corr_value = ((data['high'] + data['low']) / 2).rolling(window=3).corr(adv40)

    # Step 4: Apply decay_linear and rank
    rank2 = decay_linear(corr_value, 6).rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 5: Get min rank
    data['wq_077'] = np.minimum(rank1, rank2)

    return data['wq_077']


def wq_078(data):
    """
    Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))

    Arguments:
    data : pd.DataFrame : a DataFrame with ['low', 'vwap', 'volume']

    Returns:
    pd.Series : a pd.Series contains the result of the factor calculation
    """

    # Step 1: Calculate sum of ((low * 0.352233) + (vwap * (1 - 0.352233))) for 19.7428 days
    combined_value = (data['low'] * 0.352233) + (data['vwap'] * (1 - 0.352233))
    sum_combined_value = combined_value.rolling(window=int(19.7428)).sum()

    # Step 2: Calculate sum of adv40 for 19.7428 days
    adv40 = data['volume'].rolling(window=40).mean()
    sum_adv40 = adv40.rolling(window=int(19.7428)).sum()

    # Step 3: Calculate correlation and rank
    rank1 = sum_combined_value.rolling(window=int(6.83313)).corr(sum_adv40).rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 4: Calculate correlation of rank of vwap and volume, then rank
    data['vwap_rank'] = data['vwap'].rolling(window=22, min_periods=22).apply(rf, raw=True)
    data['volume_rank'] = data['volume'].rolling(window=22, min_periods=22).apply(rf, raw=True)
    corr = data['vwap_rank'].rolling(window=int(5.77492)).corr(data['volume_rank'])
    rank2 = corr.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 5: Multiply the ranks
    data['wq_078'] = rank1 ** rank2

    return data['wq_078']

# wq_079、wq_080由于在期货市场中无对应，不再实现。


def wq_081(data):
    """
    Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),
    8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1

    Arguments:
    data : pd.DataFrame : a DataFrame with ['vwap', 'volume']

    Returns:
    pd.Series : a pd.Series contains the result of the factor calculation
    """

    # Step 1: Calculate correlation and rank
    adv10 = data['volume'].rolling(window=10).mean()
    sum_adv10 = adv10.rolling(window=int(49.6054)).sum()
    rank1 = data['vwap'].rolling(window=int(8.47743)).corr(sum_adv10).rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 2: Rank of the rank
    rank2 = rank1.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 3: Power of 4
    rank_powered = rank2**4

    # Step 4: Log of the product of ranks
    log_product = np.log(rank_powered.rolling(window=int(14.9655)).apply(np.product))
    rank3 = log_product.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 5: Calculate correlation and rank
    data['vwap_rank'] = data['vwap'].rolling(window=22, min_periods=22).apply(rf, raw=True)
    data['volume_rank'] = data['volume'].rolling(window=22, min_periods=22).apply(rf, raw=True)
    rank_corr = data['vwap_rank'].rolling(window=int(5.07914)).corr(data['volume_rank'])
    rank4 = rank_corr.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 6 & 7: Compare the ranks and return -1 or 1
    data['wq_081'] = (rank3 < rank4) * -1

    return data['wq_081']

# wq_082由于在期货市场中无对应，不再实现。


def wq_083(data):
    """
    Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -
    low) / (sum(close, 5) / 5)) / (vwap - close)))

    Arguments:
    data : pd.DataFrame : a DataFrame with ['high', 'low', 'close', 'vwap', 'volume']

    Returns:
    pd.Series : a pd.Series contains the result of the factor calculation
    """

    # Step 1: Calculate the average of sum of close over 5 days
    avg_close_5 = data['close'].rolling(window=5).sum() / 5

    # Step 2: Calculate the ratio of high-low to the avg_close_5
    hl_avg_ratio = (data['high'] - data['low']) / avg_close_5

    # Step 3: Delay the hl_avg_ratio by 2 days
    delayed_hl_avg_ratio = hl_avg_ratio.shift(2)

    # Step 4: Rank of the delayed ratio
    rank1 = delayed_hl_avg_ratio.rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 5: Rank of volume
    rank2 = data['volume'].rolling(window=22, min_periods=22).apply(rf, raw=True)

    # Step 6: Calculate the ratio of hl_avg_ratio to the difference of vwap and close
    hl_vw_c_ratio = hl_avg_ratio / (data['vwap'] - data['close'])

    # Step 7: Calculate the final factor value
    data['wq_083'] = rank1 * rank2 / hl_vw_c_ratio

    return data['wq_083']


def wq_084(data):
    # 调不好，放弃了
    """
    Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))

    Args:
    - data (pd.DataFrame): A dataframe containing at least 'vwap' and 'close' columns.

    Returns:
    - pd.Series: The computed Alpha#84 values.
    """
    # 1. Calculate the difference between VWAP and its max over the past 15 days
    vwap_diff = data['vwap'] - data['vwap'].rolling(window=15).max()

    # 2. Rank the difference over the past 21 days
    ts_rank_vwap_diff = (
        vwap_diff.rolling(window=21, min_periods=21)
        .apply(rf, raw=True)
    )

    # 3. Compute the delta of close over 5 days
    delta_close = data['close'].diff(5)
    # Check for very large values
    print((np.abs(ts_rank_vwap_diff) > 1e10).sum())

    # Check for very large positive values in delta_close
    print((delta_close > 50).sum())

    # Check for situations where ts_rank_vwap_diff is close to 0 and delta_close is negative
    print(((np.abs(ts_rank_vwap_diff) < 1e-10) & (delta_close < 0)).sum())
    # 4. Calculate the signed power using the computed values
    data['wq_084'] = np.sign(ts_rank_vwap_diff) * np.power(np.abs(ts_rank_vwap_diff), delta_close)

    return data['wq_084']


def wq_085(data):
    """
    Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,
    9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),
    7.11408)))

    Args:
    - data (pd.DataFrame): A dataframe containing at least 'high', 'close', 'low', and 'volume' columns.

    Returns:
    - pd.Series: The computed Alpha#85 values.
    """
    # 1. Calculate (high * 0.876703) + (close * (1 - 0.876703))
    high_close_combo = (data['high'] * 0.876703) + (data['close'] * (1 - 0.876703))

    # 2. Compute the correlation of the computed value with the average volume over 30 days for the past 10 days
    adv30 = data['volume'].rolling(window=30).mean()
    corr1 = high_close_combo.rolling(window=10).corr(adv30)

    # 3. Calculate Ts_Rank of (high + low) / 2 over 4 days
    ts_rank_high_low = (data['high'] + data['low']).rolling(window=4).apply(rf, raw=True) / 2

    # 4. Calculate Ts_Rank of volume over 11 days
    ts_rank_volume = data['volume'].rolling(window=11).apply(rf, raw=True)

    # 5. Compute the correlation of the two Ts_Ranks over 8 days
    corr2 = ts_rank_high_low.rolling(window=8).corr(ts_rank_volume)

    # 6. Compute the product of the ranks of the two correlations
    data['wq_085'] = (
            corr1.rolling(window=22, min_periods=22).apply(rf, raw=True) *
            corr2.rolling(window=22, min_periods=22).apply(rf, raw=True)
    )

    return data['wq_085']


def wq_086(data):
    # Alpha 86: ((Ts_Rank(correlation(close, sum(adv20, 14), 6), 20) < rank(((open + close) - (vwap + open)))) *-1)
    data['adv20'] = data['vwap'].rolling(window=20).mean()
    data['sum_adv20'] = data['adv20'].rolling(window=14).sum()
    data['correlation'] = data['close'].rolling(window=6).corr(data['vwap'])
    data['rank_1'] = data['correlation'].rolling(window=20).apply(rf, raw=True)
    data['rank_expr'] = data['open'] + data['close'] - data['vwap'] - data['open']
    data['rank_2'] = data['rank_expr'].rolling(window=20).apply(rf, raw=True)
    data['wq_086'] = np.where(data['rank_1'] < data['rank_2'], -1, 1)
    return data['wq_086']


def wq_088(df):
    """Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),
    8)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(adv60,
    20), 8), 6), 3))"""
    df['adv60'] = df['volume'].rolling(window=60).mean()

    # 计算rank值
    rank_open = df['open'].rolling(window=20).apply(rf, raw=True)
    rank_low = df['low'].rolling(window=20).apply(rf, raw=True)
    rank_high = df['high'].rolling(window=20).apply(rf, raw=True)
    rank_close = df['close'].rolling(window=20).apply(rf, raw=True)

    # 计算rank的组合值
    combined_rank = rank_open + rank_low - rank_high - rank_close

    # 使用decay_linear函数
    decayed_combined_rank = decay_linear(combined_rank, 8)

    # 计算Ts_Rank
    ts_rank_close = df['close'].rolling(window=8).apply(rf, raw=True)
    ts_rank_adv60 = df['adv60'].rolling(window=20).apply(rf, raw=True)

    # 计算correlation
    corr_ts_rank = ts_rank_close.rolling(window=8).corr(ts_rank_adv60)

    # 使用decay_linear函数
    decayed_corr_ts_rank = decay_linear(corr_ts_rank, 6)

    # 计算Ts_Rank for decayed correlation
    ts_rank_decayed_corr = decayed_corr_ts_rank.rolling(window=3).apply(rf, raw=True)

    # 计算Alpha#88
    df['wq_088'] = pd.concat([decayed_combined_rank, ts_rank_decayed_corr], axis=1).min(axis=1)

    return df['wq_088']









