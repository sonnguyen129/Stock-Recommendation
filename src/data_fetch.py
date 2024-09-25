import os
import pandas as pd
import random
import time
import numpy as np
from vnstock3 import Vnstock
from datetime import datetime, timedelta

def _load_symbols(path):
    df_hose = pd.read_csv(path, encoding='ISO-8859-1')
    stock_symbols = df_hose['Symbol'].unique().tolist()
    print("Loaded %d stock symbols" % len(stock_symbols))
    return df_hose, stock_symbols

def fetch_prices(symbol, out_name, start_date, end_date):
    print("Fetching {} ...".format(symbol))

    stock = Vnstock().stock(symbol=symbol, source='VCI')
    data = stock.quote.history(start=start_date, end=end_date)
    data['ticker'] = symbol
    data.to_csv(out_name, index = False)

    data = pd.read_csv(out_name)
    if data.empty:
        print("Remove {} because the data set is empty.".format(out_name))
        os.remove(out_name)
    else:
        dates = data.iloc[:, 0].tolist()
        print("# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0]))

    sleep_time = np.round(np.random.uniform(low=1, high=3), 2)
    print("Sleeping ... %.2fs" % sleep_time)
    time.sleep(sleep_time)
    return True

def combine_file(folder_path):
    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]

    df_list = []

    for csv in csv_files:
        file_path = os.path.join(folder_path, csv)
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
                df_list.append(df)
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")

    big_df = pd.concat(df_list, ignore_index=True)
    big_df.to_csv(os.path.join('C:\Stock-Recommendation\\datasets\\lastest_combined_file.csv'), index=False)

if __name__ == '__main__':
    STOCK_DIR = "C:\Stock-Recommendation\\datasets\\HOSE_datasets"
    VN_LIST_PATH = "C:\Stock-Recommendation\\datasets\\VN_HOSE_Companies.csv"

    end_date = datetime.now()
    start_date = end_date - timedelta(days = 500)
    end_date = end_date.strftime("%Y-%m-%d")
    start_date = start_date.strftime("%Y-%m-%d")

    num_failure = 0
    data_failure = 0

    df_hose, symbols = _load_symbols(VN_LIST_PATH)

    print("==================== start fetch data ====================")
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(STOCK_DIR, sym + ".csv")

        succeeded = fetch_prices(sym, out_name, start_date, end_date)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print("# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure))

    # df = combine_file(STOCK_DIR)