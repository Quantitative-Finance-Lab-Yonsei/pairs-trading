import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


def find_cointegrated_pairs(data):
    """
    Summary:
        This function finds cointegrated pairs from the given data.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data of multiple assets.

    Returns:
        np.ndarray: A matrix of p-values for the cointegration test between each pair of assets.
        list: A list of tuples containing the pairs of assets that are cointegrated along with their p-values.
    """
    n = data.shape[1]
    p_value_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            series1 = data.loc[:, keys[i]]
            series2 = data.loc[:, keys[j]]
            result = coint(series1, series2)
            p_value = result[1]
            p_value_matrix[i, j] = p_value
            if p_value < 0.05:  # Using a threshold of 0.05 for significance
                pairs.append((keys[i], keys[j], p_value))
        pairs.sort(key=lambda x: x[2])
    return p_value_matrix, pairs


def pairs_trade(
    data_df,
    asset_a,
    asset_b,
    window_size=252,
    sigma=2,
    transaction_cost=0.005,
    plot=True,
    figsize=(12, 4),
):
    """
    Function to perform pairs trading on two assets.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        asset_a (str): The first asset for pairs trading. Must be one of the columns in the data_df.
        asset_b (str): The second asset for pairs trading. Must be one of the columns in the data_df.
        window_size (int, optional): The window size for rolling calculations. Defaults to 252.
        sigma (int, optional): The coefficient for standard deviation. Defaults to 2.
        transaction_cost (float, optional): The transaction cost. Defaults to 0.01.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        figsize (tuple, optional): The figure size for the plot. Defaults to (12, 4).

    Returns:
        pd.DataFrame: DataFrame containing the pairs trading results.
        list: List of pair returns.
    """

    assert (
        asset_a in data_df.columns
    ), f"asset_a='{asset_a}' not found in the data_df. Pick one of {list(data_df.columns)}."
    assert (
        asset_b in data_df.columns
    ), f"'asset_b={asset_b}' not found in the data_df. Pick one of {list(data_df.columns)}."
    assert (
        len(data_df) >= window_size + 126
    ), f"Data length must be >= window size + 6-month-trading days (126 days). Right now, data length is {len(data_df)} < {window_size+126}."

    pair_df = data_df[[asset_a, asset_b]].reset_index(drop=True)
    pair_df.columns = ["asset_a", "asset_b"]

    # find the rolling window correlation coefficient
    pair_df["corr"] = (
        pair_df["asset_a"].rolling(window=window_size).corr(pair_df["asset_b"])
    )

    # find the spread
    pair_df["spread"] = (
        pair_df["asset_a"] / pair_df["asset_a"].iloc[0]
        - pair_df["asset_b"] / pair_df["asset_b"].iloc[0]
    )

    # find the equilibrium spread
    pair_df["equi_spread"] = (
        pair_df["spread"].rolling(window=window_size, min_periods=0).mean()
    )
    pair_df["normalized_spread"] = pair_df["spread"] - pair_df["equi_spread"]

    # find the std
    pair_df["std"] = (
        pair_df["normalized_spread"].rolling(window=window_size).std() * sigma
    )
    pair_df["neg_std"] = -pair_df["std"]

    # find signal

    pos_cross_points = []
    neg_cross_points = []
    zero_cross_points = []
    status = 0  # not trading
    pair_cum_return = 1
    pair_df["pair_cum_return"] = 1
    pair_return_list = []

    for i in range(len(pair_df)):

        if i < len(pair_df) - 1:
            # if position is not open, check for crossing points
            if status == 0:
                # short asset_a, long asset_b
                if pair_df["normalized_spread"].iloc[i] > pair_df["std"].iloc[i]:
                    pos_cross_points.append(i)
                    status = 1
                    asset_a_short = pair_df["asset_a"].iloc[i]  # record asset_a price
                    asset_b_long = pair_df["asset_b"].iloc[i]  # record asset_b price
                    pair_cum_return = pair_cum_return * (
                        1 - transaction_cost
                    )  # return after transaction cost

                # long asset_a, short asset_b
                elif pair_df["normalized_spread"].iloc[i] < pair_df["neg_std"].iloc[i]:
                    neg_cross_points.append(i)
                    status = -1
                    asset_a_long = pair_df["asset_a"].iloc[i]  # record asset_a price
                    asset_b_short = pair_df["asset_b"].iloc[i]  # record asset_b price
                    pair_cum_return = pair_cum_return * (1 - transaction_cost)
                    
            # if position is open, check for zero crossing points
            elif status == 1:
                if pair_df["normalized_spread"].iloc[i] < 0:
                    zero_cross_points.append(i)
                    status = 0
                    asset_a_return_on_the_dollar = (
                        asset_a_short / pair_df["asset_a"].iloc[i + 1]
                    )  # positions are closed on the next day to account for bid-ask bounce
                    asset_b_return_on_the_dollar = (
                        pair_df["asset_b"].iloc[i + 1] / asset_b_long
                    )

                    pair_return = (
                        0.5 * asset_a_return_on_the_dollar
                        + 0.5 * asset_b_return_on_the_dollar
                    )
                    pair_return = pair_return * (
                        1 - transaction_cost
                    )  # return after transaction cost
                    pair_cum_return = pair_cum_return * pair_return
                    pair_return_list.append((pair_return - 1))

            # if position is open, check for zero crossing points
            elif status == -1:
                if pair_df["normalized_spread"].iloc[i] > 0:
                    zero_cross_points.append(i)
                    status = 0
                    asset_a_return_on_the_dollar = (
                        pair_df["asset_a"].iloc[i + 1] / asset_a_long
                    )
                    asset_b_return_on_the_dollar = (
                        asset_b_short / pair_df["asset_b"].iloc[i + 1]
                    )

                    pair_return = (
                        0.5 * asset_a_return_on_the_dollar
                        + 0.5 * asset_b_return_on_the_dollar
                    )
                    pair_return = pair_return * (
                        1 - transaction_cost
                    )  # return after transaction cost
                    pair_cum_return = pair_cum_return * pair_return
                    pair_return_list.append((pair_return - 1))

        # if the last day is reached and the position is still open, close the position
        elif i == len(pair_df) - 1 and status != 0:
            if status == 1:
                zero_cross_points.append(i)
                status = 0
                asset_a_return_on_the_dollar = (
                    asset_a_short / pair_df["asset_a"].iloc[i]
                )  # positions are closed on the next day to account for bid-ask bounce
                asset_b_return_on_the_dollar = pair_df["asset_b"].iloc[i] / asset_b_long

                pair_return = (
                    0.5 * asset_a_return_on_the_dollar
                    + 0.5 * asset_b_return_on_the_dollar
                )
                pair_return = pair_return * (
                    1 - transaction_cost
                )  # return after transaction cost
                pair_cum_return = pair_cum_return * pair_return
                pair_return_list.append((pair_return - 1))

            elif status == -1:
                zero_cross_points.append(i)
                status = 0
                asset_a_return_on_the_dollar = pair_df["asset_a"].iloc[i] / asset_a_long
                asset_b_return_on_the_dollar = (
                    asset_b_short / pair_df["asset_b"].iloc[i]
                )

                pair_return = (
                    0.5 * asset_a_return_on_the_dollar
                    + 0.5 * asset_b_return_on_the_dollar
                )
                pair_return = pair_return * (
                    1 - transaction_cost
                )  # return after transaction cost # TODO: Make this more accurate
                pair_cum_return = pair_cum_return * pair_return
                pair_return_list.append((pair_return - 1))

        pair_df["pair_cum_return"].iloc[
            i
        ] = pair_cum_return  # record the cumulative return for each day

    if plot:
        plt.figure(figsize=figsize)
        pair_df.normalized_spread.plot(x=pair_df.index)
        plt.plot(pair_df["std"], color="purple")
        plt.plot(-pair_df["std"], color="purple")
        plt.scatter(
            pos_cross_points,
            pair_df["std"].iloc[pos_cross_points],
            color="green",
            zorder=3,
            label="Cross Points",
            marker="^",
        )
        plt.scatter(
            zero_cross_points,
            [0] * len(zero_cross_points),
            color="orange",
            zorder=3,
            label="Close Pairs Position",
        )
        plt.scatter(
            neg_cross_points,
            pair_df["neg_std"].iloc[neg_cross_points],
            color="red",
            zorder=3,
            label="Cross Points",
            marker="^",
        )

        plt.axhline(linestyle="--", color="black")
        plt.legend()
        plt.title(f"{asset_a} & {asset_b}")
        plt.ylabel("Spread size")
        plt.xlabel("Date")

        # pair_df.pair_cum_return.plot(x=pair_df.index, secondary_y=True, color='blue', label='Pair Return')
    return pair_df, pair_return_list, pair_cum_return - 1


def pairs_trade_loop(
    data_df,
    asset_a,
    asset_b,
    window_size=252,
    sigma=2,
    transaction_cost=0.005,
    plot=False,
    figsize=(12, 4),
    step_size=20,
):
    """
    Summary:
        This function performs pairs trading on two assets over moving time windows with a chosen step size.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        asset_a (str): The first asset for pairs trading.
        asset_b (str): The second asset for pairs trading.
        window_size (int, optional): The window size for rolling calculations. Defaults to 252.
        sigma (int, optional): The coefficient for standard deviation. Defaults to 2.
        transaction_cost (float, optional): The transaction cost. Defaults to 0.005.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        figsize (tuple, optional): The figure size for the plot. Defaults to (12, 4).
        step_size (int, optional): The step size for the rolling window. Defaults to 20.

    Returns:
        list: List of cumulative returns for each time window.
    """

    returns_list = []
    for i in range(0, len(data_df) - window_size - 126, step_size):
        _, _, cum_return = pairs_trade(
            data_df[i : i + window_size + 126],
            asset_a,
            asset_b,
            window_size=window_size,
            sigma=sigma,
            transaction_cost=transaction_cost,
            plot=plot,
            figsize=figsize,
        )
        returns_list.append(cum_return)
    return returns_list


def find_distance_pairs(data):
    """
    Summary:
        This function finds pairs of assets based on the distance method.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data of multiple assets.

    Returns:
        list: A list of tuples containing the pairs of assets along with their distances.
    """
    
    n = data.shape[1]
    p_value_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            series1 = data.loc[:, keys[i]]
            series2 = data.loc[:, keys[j]]
            normalized_series1 = series1 / series1.iloc[0]
            normalized_series2 = series2 / series2.iloc[0]
            sum_of_squares = np.sum((normalized_series1 - normalized_series2) ** 2)
            distance = np.sqrt(sum_of_squares)
            pairs.append((keys[i], keys[j], distance))

    pairs = sorted(pairs, key=lambda x: x[2])
    return pairs


def pairs_trade_with_selection(
    data_df,
    selection_method,
    window_size=252,
    sigma=2,
    transaction_cost=0.005,
    plot=False,
    figsize=(12, 4),
    step_size=20,
    number_of_top_pairs=5,
):

    assert selection_method in [
        "cointegration",
        "distance",
        "hurst",
        "all",
    ], "Invalid selection method. Choose from 'cointegration', 'distance', or 'hurst'."

    returns_list = []
    for i in tqdm(range(0, len(data_df) - window_size - 126, step_size)):
        top_pairs = []
        window_data = data_df.iloc[i : i + window_size]
        if selection_method == "cointegration":
            _, pairs = find_cointegrated_pairs(window_data)
            if len(pairs) == 0:
                continue
            top_pairs = [(a, b) for a, b, _ in pairs]
            top_pairs = top_pairs[
                :number_of_top_pairs
            ]  # Select top X pairs based on p-value

        elif selection_method == "distance":
            pairs = find_distance_pairs(window_data)
            if len(pairs) == 0:
                continue
            top_pairs = [(a, b) for a, b, _ in pairs]
            top_pairs = top_pairs[:number_of_top_pairs]  # Select top X pairs
            pass
        elif selection_method == "hurst":
            # do that
            pass
        elif selection_method == "all":
            top_pairs = find_pairs_from_list(data_df.columns.tolist())

        if top_pairs:
            for asset_a, asset_b in top_pairs:
                _, _, cum_return = pairs_trade(
                    data_df[i : i + window_size + 126],
                    asset_a,
                    asset_b,
                    window_size=window_size,
                    sigma=sigma,
                    transaction_cost=transaction_cost,
                    plot=plot,
                    figsize=figsize,
                )
                returns_list.append(cum_return)
    return returns_list

def calculate_stats(list, printing=False):
    """
    Summary:
        This function calculates and prints basic statistics for a given list of returns.

    Args:
        list (list): The input list of returns.
        printing (bool, optional): Whether to print the statistics. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """

    assert len(list) > 0, "List must contain at least one element."

    # Calculate basic statistics
    mean_return = np.mean(list)
    median_return = np.median(list)
    std_return = np.std(list)
    min_return = np.min(list)
    max_return = np.max(list)
    q1_return = np.percentile(list, 25)
    q3_return = np.percentile(list, 75)
    sharpe_ratio = mean_return / std_return if std_return != 0 else np.nan
    skew_return = stats.skew(list)
    kurtosis_return = stats.kurtosis(list)

    # Print the statistics
    if printing:
        print(f"Mean Return: {mean_return}")
        # print(f"Median Return: {median_return}")
        print(f"Standard Deviation: {std_return}")
        print(f"Minimum Return: {min_return}")
        print(f"Maximum Return: {max_return}")
        print(f"25th Percentile (Q1): {q1_return}")
        print(f"75th Percentile (Q3): {q3_return}")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Skewness: {skew_return}")
        print(f"Kurtosis: {kurtosis_return}")

    return {
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "min_return": float(min_return),
        "max_return": float(max_return),
        "q1_return": float(q1_return),
        "q3_return": float(q3_return),
        "sharpe_ratio": float(sharpe_ratio),
        "skew_return": float(skew_return),
        "kurtosis_return": float(kurtosis_return),
    }

def find_pairs_from_list(list):
    """
    Summary:
        This function generates all possible pairs from a given list (excluding pairs of itself and (a, b) is considered equal to (b, a)).

    Args:
        list (list): The input list from which pairs are to be generated.

    Returns:
        list: A list of tuples, where each tuple contains a pair of elements from the input list.
    """
    return [
        (list[i], list[j]) for i in range(len(list)) for j in range(i + 1, len(list))
    ]
    

def find_pairs_from_two_lists(list1, list2):
    """
    Generate all possible pairs from two lists.

    This function takes two lists as input and returns a list of tuples,
    where each tuple contains one element from the first list and one
    element from the second list.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.

    Returns:
        list of tuples: A list containing all possible pairs (tuples) 
        formed by taking one element from list1 and one element from list2.
    """
    return [(list1[i], list2[j]) for i in range(len(list1)) for j in range(len(list2))]
