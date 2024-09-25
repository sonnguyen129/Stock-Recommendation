import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import plotly.express as px
from scipy.signal import savgol_filter
import seaborn as sns
from tqdm import tqdm
np.float_ = np.float64
from prophet import Prophet
import itertools
from datetime import datetime, timedelta

# np.float_ = np.float64

class ProphetModel:
    def __init__(self, monthly_seasonality=True, changepoint_prior_scale=0.05, changepoint_range=0.8):
        pass

    def prophet_predictions(self, training_df, cv, pred_date, monthly_seasonality=True, changepoint_prior_scale=0.05, changepoint_range=0.8):
        """
        Train and predict sales using Prophet
        """
        training_df.rename(columns={'close': 'y', 'time':'ds'}, inplace=True)
        series_list = training_df['series_id'].unique()
        prophet_pred_list = []
        for series in tqdm(series_list, desc=f"Predicting for cv{cv}:"):
            training_df_series = training_df.loc[training_df.series_id==series]
            m = Prophet(yearly_seasonality=False, daily_seasonality=False, \
                        changepoint_prior_scale=  changepoint_prior_scale, changepoint_range= changepoint_range)
            if monthly_seasonality:
                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m.fit(training_df_series)
            future = pd.DataFrame({'ds': [pred_date]})
            print(future)
            forecast = m.predict(future)[['ds', 'yhat']]
            forecast['series_id'] = series
            prophet_pred_list.append(forecast)
        prophet_pred_df = pd.concat(prophet_pred_list)
        prophet_pred_df.rename(columns={'ds':'date', 'yhat':'prophet_pred'}, inplace=True)
        # prophet_test_df = test_df.merge(prophet_pred_df, on=['series_id', 'date'], how='left')
        # training_df.rename(columns={'ds': 'date', 'y': 'close'}, inplace=True)
        return prophet_pred_df