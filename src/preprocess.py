import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the raw dataset from disk."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame, amount_scaler: RobustScaler = None, time_scaler: RobustScaler = None):
    
    df = df.drop_duplicates().reset_index(drop=True)

    if amount_scaler is None:
        amount_scaler = RobustScaler()
        df["scaled_amount"] = amount_scaler.fit_transform(df[["Amount"]])
    else:
        df["scaled_amount"] = amount_scaler.transform(df[["Amount"]])

    if time_scaler is None:
        time_scaler = RobustScaler()
        df["scaled_time"] = time_scaler.fit_transform(df[["Time"]])
    else:
        df["scaled_time"] = time_scaler.transform(df[["Time"]])

    df = df.drop(["Amount", "Time"], axis=1)

    # Put scaled columns at the front for readability
    other_cols = [c for c in df.columns if c not in ("scaled_amount", "scaled_time", "Class")]
    cols = ["scaled_amount", "scaled_time"] + other_cols
    if "Class" in df.columns:
        cols = cols + ["Class"]
    df = df[cols]

    return df, amount_scaler, time_scaler


def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """Stratified train/test split so fraud ratio is preserved in both sets."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    df, _, _ = preprocess(df)
    print(df.head())
    print("\nShape after preprocessing:", df.shape)
    print("\nClass balance:\n", df["Class"].value_counts(normalize=True))
