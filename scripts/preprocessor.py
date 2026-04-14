import pandas as pd


def remove_unused_categories(df:pd.DataFrame)->pd.DataFrame:
    """ Remove categories which will either lead to overfitting or dont have prediction power"""
    df = df.drop(columns=["trans_date_trans_time",
                          	"cc_num",
                            "merchant", 
                            "first",
                            "last",
                            "job",
                            "street", 
                            "city",
                            "zip",	
                            "lat",	
                            "long",	
                            "city_type",
                            "trans_num",	
                            "unix_time", 
                            "merch_lat",	
                            "merch_long", 
                            "date_of_birth", 
                            "age_group",
                            "distance_group", 
                            "distance_group_order", 
                            "time_since_prev_transaction", 
                            "speed", 
                            "speed_group", 
                            "speed_group_order", 
                            "amt_vs_avg_group", 
                            "amt_vs_avg_group_order", 
                            "trans_date"])
    return df

def get_bool_columns(df):
       dtypes_df = pd.DataFrame(df.dtypes).reset_index()
       bool_cols = list(dtypes_df[dtypes_df[0] == bool]["index"].values)
       
       return bool_cols

def encode_categoricals(df:pd.DataFrame)->pd.DataFrame:
    str_cols = []
    for col_name, col_type in df.dtypes.items():
        if type(col_type) == pd.StringDtype:
            str_cols.append(col_name)

    df_ohe = pd.get_dummies(df, columns=str_cols, prefix=str_cols, drop_first=True)
    bool_cols = get_bool_columns(df_ohe)
    
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)

    return df_ohe

def prepare_data(filepath:str="../data/processed/cleeaned_fraud_train.csv", output_path:str="../data/processed/train_ohe.csv"):
    try:
        df = pd.read_csv(filepath)
        df = remove_unused_categories(df)
        df = encode_categoricals(df)
        df.to_csv(output_path,index=False)
        return "data saved successfully"
    except Exception as e:
        return f"Error: {e}"

