import pandas as pd


def leave_last_out_split(
    df,
    user_col='user_id',
    time_col='date_time',
    drop_tied_last=True,
):
    """
    Performs a strict temporal leave-last-out split.

    For each user:
    - all earlier interactions are used for training
    - the final interaction is used for testing

    By default, users whose latest timestamp occurs multiple times are removed
    because a strict temporal order cannot be established with timestamp ties
    at the boundary between train and test.
    """
    required_cols = {user_col, time_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns for split: {sorted(missing)}")

    print("Sorting data chronologically to prevent time leakage...")

    df_sorted = df.copy()
    df_sorted["_original_order"] = range(len(df_sorted))

    df_sorted = df_sorted.sort_values(
        by=[user_col, time_col, "_original_order"]
    )

    if drop_tied_last:
        last_timestamp = df_sorted.groupby(user_col)[time_col].transform("max")
        is_last_timestamp = df_sorted[time_col] == last_timestamp

        tied_last_counts = (
            is_last_timestamp
            .groupby(df_sorted[user_col])
            .transform("sum")
        )

        ambiguous_mask = is_last_timestamp & (tied_last_counts > 1)
        ambiguous_users = df_sorted.loc[
            ambiguous_mask, user_col
        ].unique()

        if len(ambiguous_users) > 0:
            print(
                "Dropping users with tied final timestamps because a strict "
                "leave-last-out split is not identifiable: "
                f"{len(ambiguous_users)} users"
            )

            df_sorted = df_sorted[
                ~df_sorted[user_col].isin(ambiguous_users)
            ].copy()

    print("Splitting data: Leave-last-out...")

    train_df = df_sorted.groupby(user_col).head(-1).copy()
    test_df = df_sorted.groupby(user_col).tail(1).copy()

    train_df = train_df.drop(columns="_original_order")
    test_df = test_df.drop(columns="_original_order")

    print(f"Train Set: {len(train_df)} interactions")
    print(f"Test Set: {len(test_df)} interactions")

    return train_df, test_df