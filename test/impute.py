import pandas as pd


def impute_with_regression(df, package):
    if not df.isnull().values.any():
        print("No missing value cell to impute data!")
        return df

    na_columns = df.columns[df.isna().any()].tolist()
    print('   - Columns ' + str(na_columns) + ' will be filled using ' + package.__class__.__name__)

    for c in na_columns:
        temp_df = df.copy(deep=True)  # or tempDF = df.copy(deep=True)
        na_columns_other_than_c = [x for x in na_columns if x != c]
        na_columns_store = temp_df[na_columns_other_than_c].values
        temp_df[na_columns_other_than_c] = temp_df[na_columns_other_than_c].fillna(df.mean())  # or maybe median
        train = temp_df[pd.notnull(temp_df[c])]
        test = temp_df[pd.isnull(temp_df[c])]
        indices = train.index.tolist() + test.index.tolist()
        train_x = train.loc[:, train.columns != c]
        train_y = train[c].astype(int)
        test_x = test.loc[:, test.columns != c]
        package.fit(train_x, train_y)
        y_pred = package.predict(test_x)
        test[c] = y_pred
        filled_column = pd.concat([train[c], test[c]], ignore_index=True)
        filled_column_df = pd.DataFrame({'Indices': indices, 'FilledColumn': filled_column})
        filled_column_df = filled_column_df.sort_values('Indices')
        df[c] = filled_column_df['FilledColumn'].values
        df[na_columns_other_than_c] = na_columns_store

    return df
