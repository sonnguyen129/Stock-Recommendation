import logging
from src.data_processing import DataProcessor
from src.models import ProphetModel
import pandas as pd
from src.data_fetch import combine_file
from datetime import datetime, timedelta

logFormatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

combine_file('C:\Stock-Recommendation\\datasets\\HOSE_datasets')
df = pd.read_csv('C:\Stock-Recommendation\\datasets\lastest_combined_file.csv', parse_dates=['time'])

dataprocessor = DataProcessor(prediction_length=28)
training_df, _, _ = dataprocessor._get_cv_split(df, split_num = 0, validation=True)
logger.info("Prediction: data for prediction obtained")

pred_date = (dataprocessor.get_max_date() + timedelta(days = 1))

# pass data to model
model = ProphetModel()
predictions_df = model.prophet_predictions(training_df, cv = 0, pred_date = pred_date)
logger.info("Prediction: predictions made")

predictions_df.to_csv('C:\Stock-Recommendation\\datasets\prediction_file.csv', index=False)

dataprocessor.get_final_prediction(df, predictions_df)