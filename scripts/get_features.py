import pandas as pd
from geopy.distance import great_circle
import numpy as np

def clean_data(df:pd.DataFrame):
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["date_of_birth"] = pd.to_datetime(df["dob"])
    df["merchant"] = df["merchant"].apply(lambda x:x.replace("fraud_", "") )
    
    df.drop(columns=["Unnamed: 0","dob"],inplace=True)
    df = df.sort_values(["cc_num", "trans_date_trans_time"]) # sort values in the start
    return df
    
    
def add_time(df:pd.DataFrame):
    df["week_day"] = df["trans_date_trans_time"].dt.day_name()
    df["hour"] = df["trans_date_trans_time"].dt.hour
    return df

    
def add_age(df:pd.DataFrame):
    df["age"] = df["trans_date_trans_time"].dt.year - df["date_of_birth"].dt.year
    bins = [0, 17, 24, 34, 49, 64, 100]
    labels = ["0-17", "18-24", "25-34","35-49", "50-64","65+"]
    df['age_group'] = pd.cut(df['age'],right=True, bins=bins, labels=labels, include_lowest=True)
    return df


def add_city_type(df: pd.DataFrame):
    bins = [0, 1000, 10000, 100000, 500000, 1000000, 10000000]
    labels = [
        "village",          # <1k
        "small town",       # 1k–10k
        "town",             # 10k–100k
        "small city",       # 100k–500k
        "medium city",      # 500k–1M
        "large city"        # 1M+
    ]
    df['city_type'] = pd.cut(
        df['city_pop'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return df


def calc_distance(a_lat:float, a_long:float, b_lat:float, b_long:float):
    point_a = (a_lat, a_long)
    point_b = (b_lat, b_long)
    return great_circle(point_a, point_b).km
    


def add_distance(df:pd.DataFrame):
    df["distance_from_home"] = df.apply(lambda row: calc_distance(row["lat"], row["long"], row["merch_lat"], row["merch_long"]), axis=1)
    return df


def add_distance_bucket(df: pd.DataFrame):
    bins = [0, 5, 20, 100, 500, float("inf")]

    labels = [
        "Very Close (0–5 km)",
        "Nearby (5–20 km)",
        "City-Level (20–100 km)",
        "Far (100–500 km)",
        "Very Far (>500 km)"
    ]

    df["distance_group"] = pd.cut(
        df["distance_from_home"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Add sort column for Power BI
    df["distance_group_order"] = df["distance_group"].cat.codes

    return df


def add_amt_mean(df:pd.DataFrame):
    df["mean_amt"] = (
        df.groupby("cc_num")["amt"]
        .expanding()
        .mean()
        .shift()
        .reset_index(level=0, drop=True)
    )
    return df


    
def add_amt_std(df:pd.DataFrame):
    df["amt_std"] = (
        df.groupby("cc_num")["amt"]
        .expanding()
        .std()
        .shift()
        .reset_index(level=0, drop=True)
    )
    return df


def add_time_since(df:pd.DataFrame):
    df["time_since_prev_transaction"] = (df.groupby("cc_num")["trans_date_trans_time"].diff())
    return df

def add_prev_transaction_distance(df):
    
    # get previous merchant coordinates per user
    df["prev_merch_lat"] = df.groupby("cc_num")["merch_lat"].shift(1)
    df["prev_merch_long"] = df.groupby("cc_num")["merch_long"].shift(1)

    # compute distance between current and previous merchant location
    df["distance_from_prev"] = df.apply(
        lambda row: calc_distance(
            row["prev_merch_lat"],
            row["prev_merch_long"],
            row["merch_lat"],
            row["merch_long"]
        ) if not np.isnan(row["prev_merch_lat"]) else 0,
        axis=1
    )

    # cleanup
    df.drop(columns=["prev_merch_lat", "prev_merch_long"], inplace=True)

    return df

def add_speed(df:pd.DataFrame):
    df["time_since_prev_transaction"] = df["time_since_prev_transaction"].fillna(pd.Timedelta(seconds=0.001))
    df["time_since_prev_transaction_hours"] = df["time_since_prev_transaction"].dt.total_seconds() / 3600
    df["speed"] = df["distance_from_prev"] / df["time_since_prev_transaction_hours"]
    df["speed"] = df["speed"].replace([np.inf, -np.inf], 0)
    return df 
def add_speed_group(df: pd.DataFrame):
    # Keep inf as very large value instead of killing it
    df["speed"] = df["speed"].replace([np.inf, -np.inf], np.nan)

    # cap extreme values 
    df["speed_capped"] = df["speed"].clip(upper=1000)

    bins = [0, 5, 40, 120, 300, 800, float("inf")]

    labels = [
        "Very Low (0–5 km/h)",
        "Low (5–40 km/h)",
        "Moderate (40–120 km/h)",
        "High (120–300 km/h)",
        "Very High (300–800 km/h)",
        "Extreme (>800 km/h)"
    ]

    df["speed_group"] = pd.cut(
        df["speed_capped"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Sort order for Power BI
    df["speed_group_order"] = df["speed_group"].cat.codes

    return df

def add_amt_vs_mean_group(df: pd.DataFrame):
    # Avoid division issues
    df["amt_vs_avg"] = df["amt"] / df["mean_amt"]
    
    df.loc[df["mean_amt"] == 0, "amt_vs_avg"] = 0
    df["amt_vs_avg"] = df["amt_vs_avg"].fillna(0)
    
    bins = [0, 0.5, 0.9, 1.1, 2, 5, float("inf")]

    labels = [
        "Much Lower (<0.5x)",
        "Lower (0.5x–0.9x)",
        "Typical (0.9x–1.1x)",
        "Above Usual (1.1x–2x)",
        "High Spike (2x–5x)",
        "Extreme Spike (>5x)"
    ]

    df["amt_vs_avg_group"] = pd.cut(
        df["amt_vs_avg"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Create numeric sort order for Power BI
    df["amt_vs_avg_group_order"] = df["amt_vs_avg_group"].cat.codes

    return df
def flag_new_users(df:pd.DataFrame):
    df["is_new_user"] = df["mean_amt"].isna().astype(int)
    return df

def add_trans_date(df:pd.DataFrame):
    df["trans_date"] = df["trans_date_trans_time"].dt.normalize()
    return df
def add_z_score(df:pd.DataFrame):
    df["amt_std"] = df["amt_std"].replace(0, 1e-6)
    df["z_score"] = (df['amt'] - df["mean_amt"])/ df['amt_std']
    df['z_score'] = df['z_score'].fillna(0)
    return df

def fill_mean_std_nas(df):
    df["mean_amt"] = df["mean_amt"].fillna(0)
    df['amt_std'] = df['amt_std'].fillna(0)
    return df


def get_features(df:pd.DataFrame):
    df = clean_data(df)
    df = add_time(df)
    df = add_age(df)
    df = add_city_type(df)
    df = add_distance(df)
    df = add_distance_bucket(df)
    df = add_amt_mean(df)
    df = add_amt_std(df)
    df = flag_new_users(df)
    df = add_time_since(df)
    df = add_prev_transaction_distance(df)
    df = add_speed(df)
    df = add_speed_group(df)
    df = add_amt_vs_mean_group(df)
    df = add_trans_date(df)
    df = add_z_score(df)
    df = fill_mean_std_nas(df)
    return df