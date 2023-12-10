import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


def read_school_holidays(file_path):
    # Read and process the CSV file containing school holidays for Paris
    school_holidays = pd.read_csv(file_path, parse_dates=['start_date', 'end_date'], sep=';')
    school_holidays = school_holidays[school_holidays['location'] == 'Paris']

    school_holidays['start_date'] = pd.to_datetime(school_holidays['start_date'], utc=True)
    school_holidays['end_date'] = pd.to_datetime(school_holidays['end_date'], utc=True)

    school_holidays_ = []
    for start_date, end_date in school_holidays[['start_date', 'end_date']].itertuples(index=False):
        holiday_ = pd.date_range(start_date, end_date)
        for day in holiday_:
            school_holidays_.append(day.to_pydatetime())
    school_holidays_ = [date.strftime('%Y-%m-%d') for date in school_holidays_]
    return school_holidays_


def classify_holiday(row, national_holidays):
    if row['date'].date() in national_holidays:
        return 'National Holiday'
    elif row['is_school_holiday']:
        return 'School Holiday'
    else:
        return 'No Holiday'


def create_holiday_dataframe(national_holidays, school_holidays_):
    # Creating the school holiday dataframe
    df_holidays = pd.DataFrame({'date': pd.date_range(start='2020-01-01', end='2021-12-31')})
    df_holidays['is_school_holiday'] = df_holidays['date'].astype(str).isin(school_holidays_)
    df_holidays['holiday'] = df_holidays.apply(lambda x: classify_holiday(x, national_holidays=national_holidays), axis=1)
    df_holidays = df_holidays.drop('is_school_holiday', axis=1)
    return df_holidays


def create_lockdown_dataframe():
    # Initiating lockdown dataframe
    df_lock = pd.DataFrame(pd.date_range(start='2020-01-01', end='2021-12-31', freq='D'), columns=['date'])
    df_lock['lockdown'] = 'no_lockdown'

    # Defining lockdown periods
    lockdown_periods = [
        {'start': '2020-03-17', 'end': '2020-05-11', 'type': 'lockdown'},
        {'start': '2020-10-30', 'end': '2020-12-15', 'type': 'lockdown'},
        {'start': '2021-04-03', 'end': '2021-05-02', 'type': 'lockdown'}]

    partial_lockdown_periods = [
        {'start': '2020-05-12', 'end': '2020-06-11', 'type': 'partial'},
        {'start': '2020-10-17', 'end': '2020-10-29', 'type': 'partial'},
        {'start': '2020-12-16', 'end': '2020-12-31', 'type': 'partial'},
        {'start': '2021-05-03', 'end': '2021-05-31', 'type': 'partial'}]

    # Updating lockdown column
    for period in lockdown_periods:
        df_lock.loc[(df_lock['date'] >= period['start']) & (
            df_lock['date'] <= period['end']), 'lockdown'] = period['type']

    for period in partial_lockdown_periods:
        df_lock.loc[(df_lock['date'] >= period['start']) & (
            df_lock['date'] <= period['end']), 'lockdown'] = period['type']

    return df_lock


def encode_dates(X):
    X = X.copy()  # modify a copy of X

    # Encode the date information from the date columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    X["counter_year"] = X['counter_installation_date'].dt.year
    X["counter_month"] = X['counter_installation_date'].dt.month
    X["counter_day"] = X['counter_installation_date'].dt.day

    X['is_weekend'] = X['date'].dt.dayofweek // 5 == 1

    X["is_work_morning_peak"] = ((X['holiday'] == False) & (X['is_weekend'] == False) & (X["hour"].isin([7])))
    X["is_work_evening_peak"] = ((X['holiday'] == False) & (X['is_weekend'] == False) & (X["hour"].isin([16])))

    return X.drop(columns=["date","counter_installation_date"]) # date column is dropped in the main function


def encode_and_standardize(df, label_encoder=None, scaler=None):
    df_modified = df.copy()

    # Separate columns into categorical and numerical
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    # Encode categorical columns
    if label_encoder is None:
      label_encoder = []
      for col in categorical_columns :
        encoder = LabelEncoder()
        df_modified[col] = encoder.fit_transform(df[col])
        label_encoder.append(encoder)
    else:
      i = 0
      for col in categorical_columns:
        df_modified[col] = label_encoder[i].fit_transform(df[col])
        i =+ 1

    # Standardize numerical columns
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numerical_columns])
    df_modified[numerical_columns] = scaler.transform(df[numerical_columns])

    return df_modified, label_encoder, scaler


def merge_external_data(X_train_v1, X_test_v1, ext_data_):
    ext_data_['date'] = pd.to_datetime(ext_data_['date']).astype('datetime64[us]')
    X_train = pd.merge_asof(X_train_v1, ext_data_, on='date', direction='backward')
    X_test = pd.merge_asof(X_test_v1, ext_data_, on='date', direction='backward')

    return X_train, X_test


def merge_holiday_data(X_train, X_test, df_holidays):
    df_holidays['date'] = pd.to_datetime(df_holidays['date']).astype('datetime64[us]')
    X_train = pd.merge(X_train, df_holidays, on='date', how='left')
    X_train['holiday'] = X_train['holiday'].astype('category')
    X_test = pd.merge(X_test, df_holidays, on='date', how='left')
    X_test['holiday'] = X_test['holiday'].astype('category')

    return X_train, X_test


def merge_lockdown_data(X_train, X_test, df_lock):
    df_lock['date'] = pd.to_datetime(df_lock['date']).astype('datetime64[us]')
    X_train = pd.merge(X_train, df_lock, on='date', how='left')
    X_train['lockdown'] = X_train['lockdown'].astype('category')
    X_test = pd.merge(X_test, df_lock, on='date', how='left')
    X_test['lockdown'] = X_test['lockdown'].astype('category')

    return X_train, X_test


def preprocess_data(data, test_data, ext_data, df_holidays, df_lock):
    X_train_v1 = data.drop(columns=['log_bike_count'])  # Features
    Y_train = data['log_bike_count']  # Target
    X_test_v1 = test_data

    # Process external data
    selected_columns = ['date', 't', 'ht_neige', 'pres', 'ff', 'vv', 'rr3']
    ext_data_ = ext_data[selected_columns].sort_values('date')
    ext_data_['date'] = pd.to_datetime(ext_data_['date']).astype('datetime64[us]')
    for column in ext_data_:
        ext_data_[column] = ext_data_[column].fillna(ext_data_[column].mean())

    #STORE ORIGINAL ORDER
    X_train_v1['orig_index'] = np.arange(X_train_v1.shape[0]) 
    X_test_v1['orig_index'] = np.arange(X_test_v1.shape[0])
    X_train_v1 = X_train_v1.sort_values('date')
    X_test_v1 = X_test_v1.sort_values('date')

    # Merging External Data
    X_train, X_test = merge_external_data(X_train_v1, X_test_v1, ext_data_)

    # Merging holiday data and lockdown data
    X_train, X_test = merge_holiday_data(X_train, X_test, df_holidays)
    X_train, X_test = merge_lockdown_data(X_train, X_test, df_lock)

    # Sort back to the original order
    X_train = X_train.sort_values("orig_index").drop(columns=['orig_index'])
    X_test = X_test.sort_values("orig_index").drop(columns=['orig_index'])

    # Feature selection
    columns_to_drop = ['coordinates', 'counter_technical_id',  # removed counter name
                       'site_id', 'site_name', 'bike_count']
    X_train = X_train.drop(columns=columns_to_drop)

    # Encoding and scaling
    X_train = encode_dates(X_train)
    X_train, fitted_encoder, fitted_scaler = encode_and_standardize(X_train)

    # Feature selection
    X_test = X_test.drop(columns=columns_to_drop[:-1])

    # Encoding and scaling
    X_test = encode_dates(X_test)
    X_test, _, _ = encode_and_standardize(X_test, fitted_encoder, fitted_scaler)

    return X_train, Y_train, X_test



def train_model(X_train, Y_train):
    model = XGBRegressor(max_depth=7, learning_rate=0.2935871234586509, n_estimators=104)
    model.fit(X_train, Y_train)

    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred


def create_submission_file(y_pred):
    results = pd.DataFrame(
        dict(
            Id=np.arange(y_pred.shape[0]),
            log_bike_count=y_pred,
        )
    )
    results.to_csv("submission.csv", index=False)
    
    return
    


def main():
    national_holidays = {**holidays.France(years=2020), **holidays.France(years=2021)}
    school_holidays_ = read_school_holidays('/kaggle/input/moredata/fr-en-calendrier-scolaire.csv')
    df_holidays = create_holiday_dataframe(national_holidays=national_holidays,school_holidays_=school_holidays_)
    df_lock = create_lockdown_dataframe()

    # Load data
    data = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
    test_data = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")
    ext_data = pd.read_csv("/kaggle/input/mdsb-2023/external_data.csv")

    # Data preprocessing pipeline
    X_train, Y_train, X_test = preprocess_data(data, test_data, ext_data, df_holidays, df_lock)

    # Train the model
    model = train_model(X_train, Y_train)

    # Make predictions
    y_pred = predict(model, X_test)

    # Create submission file
    create_submission_file(y_pred)


if __name__ == "__main__":
    main()