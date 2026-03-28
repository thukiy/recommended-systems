import pandas as pd


def leave_last_out_split(df, user_col='user_id', time_col='date_time'):
    """
    Performs a strict temporal split.
    The final interaction of each user is assigned to the test set,
    and all earlier interactions are assigned to the training set.
    """
    print("Sorting data chronologically to prevent time leakage...")
    df_sorted = df.sort_values(by=[user_col, time_col])

    print("Splitting data: Leave-last-out...")
    train_df = df_sorted.groupby(user_col).head(-1).copy()
    test_df = df_sorted.groupby(user_col).tail(1).copy()

    print(f"Train Set: {len(train_df)} interactions")
    print(f"Test Set: {len(test_df)} interactions")

    return train_df, test_df