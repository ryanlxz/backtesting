import pandas as pd


def create_month_and_quarter_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create month feature from the datetime index

    Args:
        df (pd.DataFrame): dataframe containing stock prices

    Returns:
        pd.DataFrame: dataframe containing month feature
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["month"] = df.index.month
    df["quarter"] = df["month"].apply(get_quarter)
    return df[["month", "quarter"]]


def get_quarter(month: int) -> int:
    """Create quarter of year feature
    Args:
        month (int): _description_

    Returns:
        int: quarter of year
    """
    return (month - 1) // 3 + 1
