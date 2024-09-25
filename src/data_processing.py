# import necessary libraries
import pandas as pd
import numpy as np
import logging
pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Responsible for processing data
    """
    def __init__(self, prediction_length):
        self.prediction_length = prediction_length

    def get_data_for_prediction(self):
        pass

    def _get_cv_split(self, df, split_num, validation=True):
        """
        Implement train-test split given a cv fold number and return training, val and test data

        Parameters:
        df (pandas.DataFrame): Dataframe of data to split
        split_num (int): Cross-validation fold number
        prediction_length (int): Number of days to predict sales
        validation (bool): Whether to split with validation data or not

        Returns:
        training_df (pandas.DataFrame): Dataframe of training data
        validation_df (pandas.DataFrame): Dataframe of validation data
        test_df (pandas.DataFrame): Dataframe of test data
        """
        if 'series_id' not in df.columns:
            df['series_id'] = df['ticker']
        series_list = df['series_id'].unique()

        test_list = []
        validation_list = []
        training_list = []

        for series in series_list:
            df_series = df.loc[df.series_id==series]
            max_date = df_series['time'].max()
            min_date = df_series['time'].min()
            test_lower_date = max_date - pd.Timedelta(f"{self.prediction_length*((split_num+1)*2-1)} days")
            test_upper_date = max_date - pd.Timedelta(f"{self.prediction_length*(split_num*2)} days")
            val_lower_date = max_date - pd.Timedelta(f"{self.prediction_length*(split_num+1)*2} days")
            if min(test_lower_date, test_upper_date) < min_date:
                raise Exception("Insufficient data for splitting")

            df_series_test = df_series.loc[(df_series['time'] > test_lower_date) & (df_series['time'] <= test_upper_date)]
            if validation:
                df_series_val = df_series.loc[(df_series['time'] > val_lower_date) & (df_series['time'] <= test_lower_date)]
                df_series_train = df_series.loc[df_series['time'] <= val_lower_date]
            else:
                df_series_val = pd.DataFrame()
                df_series_train = df_series.loc[df_series['time'] <= test_lower_date]
            test_list.append(df_series_test)
            validation_list.append(df_series_val)
            training_list.append(df_series_train)

        test_df = pd.concat(test_list)
        validation_df = pd.concat(validation_list)
        training_df = pd.concat(training_list)
        return training_df, validation_df, test_df
    
    @staticmethod
    def get_max_date():
        df = pd.read_csv('C:\Stock-Recommendation\\datasets\lastest_combined_file.csv', parse_dates=['time'])
        return df['time'].max()
    
    @staticmethod
    def get_final_prediction(df, prediction_df):
        lastest_df = df[df['time'] == df['time'].max()][['ticker', 'close']]
        lastest_df = lastest_df.rename(columns = {'ticker': 'series_id'})
        prediction_df_1 = lastest_df.merge(prediction_df, on=['series_id'], how = 'inner')
        prediction_df_1['profit_pct'] = (prediction_df_1['prophet_pred'] - prediction_df_1['close']) / prediction_df_1['close'] * 100
        target_df = prediction_df_1[['series_id', 'profit_pct']].sort_values(by = 'profit_pct', ascending=False)
        target_df.to_csv('C:\Stock-Recommendation\\datasets\prediction_target.csv', index = False)