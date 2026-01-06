import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, sep=";")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop(columns=["Date", "Time"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates()

    target = "C6H6(GT)"

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove(target)

    df_clean = remove_outliers_iqr(df, numeric_cols)

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df[target] = y.values
    processed_df.to_csv(output_path, index=False)

    print("Preprocessing selesai")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Saved to:", output_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data(
        input_path="data_raw.csv",
        output_path="preprocessed_data.csv"
    )