import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

def feature_compute(data, feature_compute_funcs):
    # 进行指标计算的函数，输入一个记录了日期、开盘价、最高价、最低价、收盘价、结算价、成交量、持有量、id、otc收益率的dataframe，以及一个函数列表
    # 函数列表里面是一个个指标计算函数

    for func in feature_compute_funcs:

        # 读取指标计算函数的名称，后续会将这个名称作为新计算的指标列的名称
        feature_name = func.__name__

        # 打印出即将进行计算的指标名，方便在写作新的指标计算代码时debug，找到出错的那个
        print(f'即将计算{feature_name}')

        # 对每个期货品种（id）应用函数
        result_series = data.groupby('id').apply(func)
        result_df = result_series.reset_index()

        # 中间增加一步整理数据标签的步骤，防止数据标签出错
        data['temp_index'] = data.index
        result_df['temp_index'] = result_df['level_1']
        data = pd.merge(data, result_df[['temp_index', feature_name]], on='temp_index')
        data.drop(columns=['temp_index'], inplace=True)

        # 使用传统正态进行指标标准化，对每个期货品种单独进行
        # mean_value = data.groupby('id')[feature_name].transform('mean')
        #
        # std_value = data.groupby('id')[feature_name].transform('std')
        #
        # data[feature_name] = (data[feature_name] - mean_value) / std_value

    return data


def truncated_normalize(data, feature_cols):
    def normalize_with_past_data(group):
        for col in feature_cols:
            cum_mean = group[col].expanding().mean()
            cum_std = group[col].expanding().std()
            group[col] = (group[col]-cum_mean)/cum_std
        return group
    return data.groupby('id').apply(normalize_with_past_data).reset_index(drop=True)


def information_coefficient(data, feature_list):
    # 该函数输入的是已经存储了各种指标的dataframe，以及需要计算ic的指标的名称
    ic_values = {}

    for unique_id in data['id'].unique():
        subset_data = data[data['id'] == unique_id]
        ic_values[unique_id] = {}

        for factor in feature_list:
            # Consider only the factor of interest and 'returns' for the correlation computation
            factor_data = subset_data[['returns', factor]]
            ic = factor_data[factor].corr(factor_data['returns'], method='spearman')
            ic_values[unique_id][factor] = ic

    ic_df = pd.DataFrame.from_dict(ic_values, orient='index')

    return ic_df


def factor_filter(data, ic, number_bar, ic_bar):
    result = {}

    for index, row in ic.iterrows():
        selected_indicators = row[row.abs() > ic_bar].index.tolist()
        if selected_indicators:
            result[index] = selected_indicators

    for name, selected_factors in result.items():
        subset_df = data[data["id"] == name]
        for factor in selected_factors.copy():
            nan_ratio = subset_df[factor].isna().sum() / len(subset_df)
            if nan_ratio > number_bar:
                result[name].remove(factor)
    # 返回一个字典，键是期货的品种代码，键的值是筛选得到的因子列表
    return result


def factor_filter_rank(data, ic, number_bar, top_n):
    result = {}

    for index, row in ic.iterrows():
        subset_df = data[data["id"] == index]

        # 先移除缺失值比率过高的指标
        factors_to_consider = []
        for factor in row.index:
            nan_ratio = subset_df[factor].isna().sum() / len(subset_df)
            if nan_ratio <= number_bar:
                factors_to_consider.append(factor)

        # 对剩余指标的IC值的绝对值进行排序并选取前top_n个
        sorted_indicators = row[factors_to_consider].abs().sort_values(ascending=False).head(top_n).index.tolist()
        result[index] = sorted_indicators

    return result


def factor_filter_double(data, ic, number_bar, ic_bar, top_n):
    result = {}

    for index, row in ic.iterrows():
        subset_df = data[data["id"] == index]

        # 1. 移除缺失值比率超过10%的指标
        factors_with_low_nan = []
        for factor in row.index:
            nan_ratio = subset_df[factor].isna().sum() / len(subset_df)
            if nan_ratio <= number_bar:
                factors_with_low_nan.append(factor)

        # 2. 选出IC绝对值大于ic_bar的指标
        factors_with_high_ic = [factor for factor in factors_with_low_nan if abs(row[factor]) > ic_bar]

        # 3. 从这些指标中选出IC绝对值前top_n的指标
        sorted_indicators = row[factors_with_high_ic].abs().sort_values(ascending=False).head(top_n).index.tolist()
        result[index] = sorted_indicators

    return result


def linear_regression_rolling(df, length):
    # 滚动训练所用函数， 将模型改成别的可以使用多元线性回归之外的其他模型
    # 使用样本点数为length的训练集滚动训练
    features_delete = ['id', 'date', 'returns', 'label', 'prediction']
    # 确定特征列
    features = list(set(df.columns) - set(features_delete))
    for index, row in df.loc[df['label'] == 1].iterrows():
        past_df = df[df.index < index] 
        # Apr01
        # start_index = index - 286
        # end_index = index
        # past_df = df.loc[start_index:end_index-1]
        features = list(set(past_df.columns) - set(features_delete))

        past_feature_df = past_df[features]
        past_return = past_df['returns']
        correlations = {}
        # 对每一列计算等级相关系数
        for column in past_feature_df.columns:
            # 计算等级相关系数
            corr, _ = spearmanr(past_feature_df[column], past_return)
            # 将结果存储在字典中
            correlations[column] = corr
        features = [feature for feature, correlation in correlations.items() if abs(correlation) > 0.07125]
        if len(features) == 0:
            continue
        # 训练模型
        model = LinearRegression()
        training_end_position = df.index.get_loc(index)
        if len(df.loc[:training_end_position]) <= length:
            model.fit(df[features].iloc[:training_end_position], df['returns'].iloc[:training_end_position])
        else:
            model.fit(df[features].iloc[training_end_position - length:training_end_position]
                      , df['returns'].iloc[training_end_position - length:training_end_position])
        # 记录预测结果
        df_prediction = pd.DataFrame([row[features]], columns=features)
        df.loc[index, 'prediction'] = model.predict(df_prediction)
    return df

def linear_regression_rolling_l(df, length):
    # 滚动训练所用函数， 将模型改成别的可以使用多元线性回归之外的其他模型
    # 使用样本点数为length的训练集滚动训练
    features_delete = ['id', 'date', 'returns', 'label', 'prediction']
    # 确定特征列
    features = list(set(df.columns) - set(features_delete))
    for index, row in df.loc[df['label'] == 1].iterrows():
        # past_df = df[df.index < index] 
        # Apr01
        start_index = index - 286
        end_index = index
        past_df = df.loc[start_index:end_index-1]
        features = list(set(past_df.columns) - set(features_delete))

        past_feature_df = past_df[features]
        past_return = past_df['returns']
        correlations = {}
        # 对每一列计算等级相关系数
        for column in past_feature_df.columns:
            # 计算等级相关系数
            corr, _ = spearmanr(past_feature_df[column], past_return)
            # 将结果存储在字典中
            correlations[column] = corr
        features = [feature for feature, correlation in correlations.items() if abs(correlation) > 0.07125]
        if len(features) == 0:
            continue
        # 训练模型
        model = LinearRegression()
        training_end_position = df.index.get_loc(index)
        if len(df.loc[:training_end_position]) <= length:
            model.fit(df[features].iloc[:training_end_position], df['returns'].iloc[:training_end_position])
        else:
            model.fit(df[features].iloc[training_end_position - length:training_end_position]
                      , df['returns'].iloc[training_end_position - length:training_end_position])
        # 记录预测结果
        df_prediction = pd.DataFrame([row[features]], columns=features)
        df.loc[index, 'prediction'] = model.predict(df_prediction)
    return df

def lasso_regression_rolling(df, length):
    features_delete = ['id', 'date', 'returns', 'label', 'prediction']
    features = list(set(df.columns) - set(features_delete))

    for index, row in df.loc[df['label'] == 1].iterrows():
        past_df = df[df.index < index] 
        features = list(set(past_df.columns) - set(features_delete))

        past_feature_df = past_df[features]
        past_return = past_df['returns']
        correlations = {}

        for column in past_feature_df.columns:
            corr, _ = spearmanr(past_feature_df[column], past_return)
            correlations[column] = corr

        features = [feature for feature, correlation in correlations.items() if abs(correlation) > 0.07125]
        if len(features) == 0:
            continue

        # Use Lasso regression
        model = Lasso(alpha=0.1)  # Example alpha, adjust based on your needs
        training_end_position = df.index.get_loc(index)

        if len(df.loc[:training_end_position]) <= length:
            model.fit(df[features].iloc[:training_end_position], df['returns'].iloc[:training_end_position])
        else:
            model.fit(df[features].iloc[training_end_position - length:training_end_position],
                      df['returns'].iloc[training_end_position - length:training_end_position])

        df_prediction = pd.DataFrame([row[features]], columns=features)
        df.loc[index, 'prediction'] = model.predict(df_prediction)

    return df

def signals_day(data, opening):
    # 训练结束后，使用这份文件产生交易记录
    # 该产生交易记录的函数仅针对AI多指标选期策略，其他策略请针对性写作其他函数

    signals = pd.DataFrame(
        columns=["Asset", "Open_Time", "Open_Price", "Direction", "Close_Time", 
                 "Close_Price", "Predicted_Return", "Position"])
    for stock_id in data['id'].unique():
        data_copy = data.loc[data['id'] == stock_id]
        for i in range(len(data_copy) - 1):  # Subtract 1 to avoid index out-of-range on the last iteration
            row = data_copy.iloc[i]
            next_row = data_copy.iloc[i + 1]
            if abs(row['prediction']) >= opening:
                # 最后生成的交易记录有6列，依次记录了标的名称，开仓时间，开仓价格，开仓方向，平仓时间，平仓价格
                one_signals = pd.DataFrame(
                    columns=["Asset", "Open_Time", "Open_Price", "Direction", 
                             "Close_Time", "Close_Price", "Predicted_Return", "Position"])

                one_signals.loc[0, "Asset"] = next_row['id']
                one_signals.loc[0, "Open_Time"] = next_row['date']
                one_signals.loc[0, "Open_Price"] = next_row['open']
                one_signals.loc[0, "Direction"] = np.sign(row['prediction']) 
                one_signals.loc[0, "Close_Time"] = next_row['date']
                one_signals.loc[0, "Close_Price"] = next_row['close']
                one_signals.loc[0, "Predicted_Return"] = row['prediction']
                one_signals.loc[0, "Position"] = 0
                signals = pd.concat([signals, one_signals], ignore_index=True)

    return signals

def new_signals_day(data, opening):
    # 训练结束后，使用这份文件产生交易记录
    # 该产生交易记录的函数仅针对AI多指标选期策略，其他策略请针对性写作其他函数

    signals = pd.DataFrame(
        columns=["Asset", "Open_Time", "Open_Price", "Direction", "Close_Time", 
                 "Close_Price", "Predicted_Return", "Position", "Real_direction"])
    for stock_id in data['id'].unique():
        count = 0
        count_win = 0
        win_rate = 1
        data_copy = data.loc[data['id'] == stock_id]
        for i in range(len(data_copy) - 1):  # Subtract 1 to avoid index out-of-range on the last iteration
            row = data_copy.iloc[i]
            next_row = data_copy.iloc[i + 1]
            if abs(row['prediction']) >= opening:
                count = count + 1
                # 最后生成的交易记录有6列，依次记录了标的名称，开仓时间，开仓价格，开仓方向，平仓时间，平仓价格
                one_signals = pd.DataFrame(
                    columns=["Asset", "Open_Time", "Open_Price", "Direction", 
                             "Close_Time", "Close_Price", "Predicted_Return", "Position", "Real_direction"])

                one_signals.loc[0, "Asset"] = next_row['id']
                one_signals.loc[0, "Open_Time"] = next_row['date']
                one_signals.loc[0, "Open_Price"] = next_row['open']
                one_signals.loc[0, "Real_direction"] = next_row['real_direction']
                outcomes = [-1, 1]  # 0 for loss, 1 for win
                rv = np.random.choice(outcomes, p=[1-win_rate, win_rate])
                one_signals.loc[0, "Direction"] = np.sign(row['prediction']) * rv
                if one_signals.loc[0, "Direction"] == one_signals.loc[0, 'Real_direction']:
                    count_win = count_win + 1
                one_signals.loc[0, "Close_Time"] = next_row['date']
                one_signals.loc[0, "Close_Price"] = next_row['close']
                one_signals.loc[0, "Predicted_Return"] = row['prediction']
                one_signals.loc[0, "Position"] = 0
                win_rate = count_win / count
                signals = pd.concat([signals, one_signals], ignore_index=True)

    return signals

def count_indicators_usage(df):
    """
    该函数用于计算每个指标被多少个期货品种使用
    使用ic筛选器生成存储了每个品种所使用的指标的字典之后，将字典传入该函数，返回一个记录了各个指标使用次数的字典

    参数:
    df (pandas.DataFrame): 包含期货品种指标的DataFrame

    返回:
    dict: 指标及其被使用的期货品种数量的字典
    """
    # 创建一个空字典来存储指标及其被使用的期货品种数量
    indicator_count = {}

    # 遍历每一列（期货品种）并统计指标的使用情况
    for col in df.columns:
        indicators = df[col].tolist()
        for indicator in indicators:
            if pd.notnull(indicator):  # 排除空值
                indicator_count[indicator] = indicator_count.get(indicator, 0) + 1
    sorted_indicator_count = dict(sorted(indicator_count.items(), key=lambda item: item[1], reverse=True))
    return sorted_indicator_count




# def create_rnn_dataset(df, features, target, window_size):
#     dataX, dataY = [], []
#     for i in range(len(df) - window_size):
#         a = df[features].iloc[i:(i + window_size)].values
#         dataX.append(a)
#         dataY.append(df[target].iloc[i + window_size])
#     return np.array(dataX), np.array(dataY)

# def rolling_rnn(df, length):
#     features_delete = ['id', 'date', 'returns', 'label', 'prediction']
#     features = list(set(df.columns) - set(features_delete))
#     scaler = MinMaxScaler(feature_range=(0, 1))

#     for index, row in df.loc[df['label'] == 1].iterrows():
#         start_index = max(0, index - length)
#         end_index = index
#         past_df = df.loc[start_index:end_index-1]
#         past_df_scaled = scaler.fit_transform(past_df[features])

#         correlations = {}
#         for feature in features:
#             corr, _ = spearmanr(past_df_scaled[feature], past_df['returns'])
#             correlations[feature] = corr
#         selected_features = [feature for feature, correlation in correlations.items() if abs(correlation) > 0.07125]

#         if not selected_features:
#             continue

#         # Prepare dataset for RNN
#         X, y = create_rnn_dataset(past_df, selected_features, 'returns', length)

#         # RNN Model
#         model = Sequential()
#         model.add(LSTM(50, activation='relu', input_shape=(length, len(selected_features))))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')

#         # Train the model (consider adding validation data here)
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)

#         # Make prediction for current index
#         current_X = scaler.transform(df[selected_features].iloc[index:index+1].values.reshape(1, -1))
#         df.loc[index, 'prediction'] = model.predict(np.array([current_X]))

#     return df


def svm_rolling(df, length):
    # 滚动训练所用函数， 将模型改成别的可以使用SVM之外的其他模型
    # 使用样本点数为length的训练集滚动训练
    features_delete = ['id', 'date', 'returns', 'label', 'prediction', 'returns_sign']
    # 确定特征列
    features = list(set(df.columns) - set(features_delete))

    ratio_up = 0.5
    for index, row in df.loc[df['label'] == 1].iterrows():

        past_df = df[df.index < index] 
        ratio_up = past_df['returns_sign'].sum() / (2 *len(past_df)) + 0.5
        # Apr01
        # start_index = index - 286
        # end_index = index
        # past_df = df.loc[start_index:end_index-1]
        features = list(set(past_df.columns) - set(features_delete))

        past_feature_df = past_df[features]
        past_return = past_df['returns']
        correlations = {}
        # 对每一列计算等级相关系数
        for column in past_feature_df.columns:
            # 计算等级相关系数
            corr, _ = spearmanr(past_feature_df[column], past_return)
            # 将结果存储在字典中
            correlations[column] = corr
        features = [feature for feature, correlation in 
                    correlations.items() if abs(correlation) > 0.07]
        if len(features) == 0:
            df.loc[index, 'prediction'] = np.random.choice([-1, 1], p=[1-ratio_up, ratio_up])
            continue
        # 训练模型
        model = SVC()
        training_end_position = df.index.get_loc(index)
        if len(df.loc[:training_end_position]) <= length:
            model.fit(df[features].iloc[:training_end_position], 
                      df['returns_sign'].iloc[:training_end_position])
        else:
            model.fit(df[features].iloc[training_end_position - length:training_end_position]
                      , df['returns_sign'].iloc[training_end_position - length:training_end_position])
        # 记录预测结果
        df_prediction = pd.DataFrame([row[features]], columns=features)
        df.loc[index, 'prediction'] = model.predict(df_prediction)
    return df


def lasso_regression_classification(df, length, alpha):
    features_delete = ['id', 'date', 'returns', 'label', 'prediction', 'returns_sign']
    features = list(set(df.columns) - set(features_delete))
    correct_rate = []
    for index, row in df.loc[df['label'] == 1].iterrows():
        start_index = index - 286
        end_index = index
        past_df = df.loc[start_index:end_index-1]
        # past_df = df[df.index < index] 
        features = list(set(past_df.columns) - set(features_delete))

        past_feature_df = past_df[features]
        past_return = past_df['returns']
        correlations = {}

        for column in past_feature_df.columns:
            corr, _ = spearmanr(past_feature_df[column], past_return)
            correlations[column] = corr
        # Use Lasso regression
        model = Lasso(alpha) 
        # features = [feature for feature, correlation in correlations.items() if abs(correlation) > 0.07125]
        # if len(features) == 0:
        #     continue
        training_end_position = df.index.get_loc(index)

        if len(df.loc[:training_end_position]) <= length:
            model.fit(df[features].iloc[:training_end_position], df['returns'].iloc[:training_end_position])
        else:
            model.fit(df[features].iloc[training_end_position - length:training_end_position],
                      df['returns'].iloc[training_end_position - length:training_end_position])

        df_prediction = pd.DataFrame([row[features]], columns=features)
        df.loc[index, 'prediction'] = np.sign(model.predict(df_prediction))

        correct_rate.append(past_df['returns_sign'].sum()  / (2 * len(past_df)) + 0.5)
        # coef = model.coef_

        # Count non-zero coefficients (i.e., features used by the model)
        # num_features_used = np.sum(coef != 0)

        # print(f"Number of features used: {num_features_used}")

    plt.figure(figsize=(10, 6))
    plt.plot(correct_rate, marker='o')
    plt.title('Correct Rate over Time')
    plt.xlabel('Time')
    plt.ylabel('Correct Rate')
    plt.grid(True)
    plt.show()
    return df