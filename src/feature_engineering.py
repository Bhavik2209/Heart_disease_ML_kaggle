def encode_target(df, target_column):
    mapping = {"Absence": 0, "Presence": 1}
    df[target_column] = df[target_column].map(mapping)
    return df


def drop_id(df):
    if "id" in df.columns:
        return df.drop(columns=["id"])
    return df
