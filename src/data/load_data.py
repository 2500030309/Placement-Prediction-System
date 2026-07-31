import pandas as pd

def load_data():
    df = pd.read_csv(r"C:\Users\LUCKY\OneDrive\Downloads\placement_data.csv")
    return df

def get_summary(df):
    return{
        "rows" : df.shape[0],
        "columns" : df.shape[1],
        "target" : "PlacementStatus"
    }

if __name__ == "__main__":
    df = load_data()
    print(get_summary(df))
    print(df.head())
