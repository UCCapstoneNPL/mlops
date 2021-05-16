# read data from data source
# save it in data/raw for further processing
# preprocess data, derive features and save it in processed folder

import os
import argparse
import datetime as dt
import pandas as pd

from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

from get_data import read_params, get_data

def load_and_save(config_path):
    config = read_params(config_path)
    im_df, ad_df = get_data(config_path)
    raw_data_path_im = config["load_data"]["raw_dataset_im"]
    raw_data_path_ad = config["load_data"]["raw_dataset_ad"]
    im_df.to_pickle(raw_data_path_im)
    ad_df.to_pickle(raw_data_path_ad)

def rolling_count(val):
    """Function for calculating the number of weeks item/product was sold"""
    if val == rolling_count.previous:
        rolling_count.count += 1
    else:
        rolling_count.previous = val
        rolling_count.count = 1
    return rolling_count.count

def build_features(config_path):
    config = read_params(config_path)
    succcess_percentile = config["preprocess_data"]["success_percentile"]
    predictor_weeks = config["preprocess_data"]["predictor_weeks"]
    processed_data = config["preprocess_data"]["processed_data"]

    # read from pickle 
    raw_data_path_im = config["load_data"]["raw_dataset_im"]
    raw_data_path_ad = config["load_data"]["raw_dataset_ad"]
    item_mvmt = pd.read_pickle(raw_data_path_im)
    adj_dtls = pd.read_pickle(raw_data_path_ad)

    # Convert 'Timestamp' to 'datatime' format and create a column with date 
    item_mvmt['Timestamp_fmt'] = pd.to_datetime(item_mvmt['Timestamp'], dayfirst=True)
    item_mvmt['Date'] = [dt.datetime.date(d) for d in item_mvmt['Timestamp_fmt']]

    # Assign Monday's date to each date as 'WeekDate', this is needed for weekly sales plotting
    item_mvmt['WeekDate'] = [(d + dt.timedelta(days =- d.weekday())) for d in item_mvmt['Date']]

    item_mvmt.to_pickle("../data/processed/item_mvmt.pkl")

    # # keep only relevant attributes
    item_trans = item_mvmt.drop(['ID','Timestamp','WeightSold','dayofweek','month','year','hrs'], axis=1)
    item_trans = item_trans.rename({'Timestamp_fmt': 'Timestamp'}, axis=1)

    # identify new products
    # Calculate min and max transaction date for each item
    itemMinMaxDate = item_trans.groupby(['ItemDescription'])['Date'].agg(['min','max'])
    itemMinMaxDate = itemMinMaxDate.rename(columns={'min':'ItemMinDate', 'max': 'ItemMaxDate'})
    item_trans = pd.merge(item_trans, itemMinMaxDate, on='ItemDescription')
    item_trans = item_trans.sort_values(by="Date")

    # First date that shows up in the historical data 
    min_historical_date = min(item_trans['Date'])

    # Creating NP_Date to verify that all new products have atleast 1 month of data. 
    item_trans['NewProductDate'] = item_trans['ItemMinDate'] + relativedelta(months = +1)

    # New product is selected by removing items who's ItemMinDate> Min_historicaldate & item has 4 weeks of sales data available. 
    item_trans_new = item_trans[(item_trans['ItemMinDate'] > min_historical_date) 
                                & (item_trans['NewProductDate'] < item_trans['ItemMaxDate']) 
                                & (item_trans['WeekDate'] > item_trans['ItemMinDate'])]

    # Drop transactions which happened during the week of 4/27/20220 (this logic needs to be generalized later)
    MaxWeekDatetoDrop = dt.date(2020,4,27)
    item_trans_new = item_trans_new[item_trans_new['WeekDate'] != MaxWeekDatetoDrop ]
    # print("Checkpoint reached")

    # Derie target variable - 'SalesSuccess'
    # Keep only relevant attributes
    item_trans_weekly = item_trans_new[['ItemNumber', 'ItemDescription', 'QuantitySold', 'FamilyName', 
                                        'CategoryName','ClassName','WeekDate','OnPromotion', 'OrgUnitNumber']]

    # Aggregate sales at week level for each item and create a weekly flag for prmotion
    item_sales_weekly = item_trans_weekly.groupby(['ItemDescription','FamilyName','CategoryName','ClassName','WeekDate']).agg({'QuantitySold':'sum',
                                                                                                                            'OnPromotion':'sum'})
    item_weekly_sales_df = item_sales_weekly.reset_index()


    # Calculate weekly cumulative sales
    item_weekly_sales_df['CumulativeQuantitySold'] = item_weekly_sales_df.groupby(['ItemDescription'])['QuantitySold'].apply(lambda x: x.cumsum())
    item_total_sales = item_weekly_sales_df.groupby(['FamilyName','CategoryName','ClassName','ItemDescription']).agg({'QuantitySold':'sum',
                                                                                                                  'WeekDate':'count'})
    item_total_sales = item_total_sales.reset_index()
    item_total_sales = item_total_sales.rename({'QuantitySold': 'TotalQuantitySold', 'WeekDate': "WeeksCount"}, axis=1)

    # Derive average sales at item level
    item_total_sales['AverageSales'] = item_total_sales['TotalQuantitySold'] / item_total_sales['WeeksCount']

    # Assign SalesSuccess as '1' for items having average sales above nth percentile and '0' for item having average sales below nth percentile
    # Get a list of categories
    categories = item_total_sales['CategoryName'].value_counts()
    # Assign nth percentile
    n = succcess_percentile
    item_response_var = pd.DataFrame(columns=['ItemDescription', 'SalesSuccess'])

    for category in categories.index:
        item_cat_wkly_sales = item_total_sales[item_total_sales['CategoryName'] == category]
        percentile_n_th = np.percentile(item_cat_wkly_sales['AverageSales'], n)
        item_cat_wkly_sales['SalesSuccess'] = [1 if x >= percentile_n_th else 0 for x in item_cat_wkly_sales['AverageSales']] 
        item_response_var = pd.concat([item_response_var, item_cat_wkly_sales[['ItemDescription', 'SalesSuccess']]], axis=0)

    item_sales_w_response = item_total_sales.merge(item_response_var, on=['ItemDescription'], how='left')

    # Convert 'Timestamp' to 'datatime' format and create a column with date 
    adj_dtls['TimeStamp_fmt'] = pd.to_datetime(adj_dtls['TimeStamp'], dayfirst=True)
    adj_dtls['Date'] = [dt.datetime.date(d) for d in adj_dtls['TimeStamp_fmt']]
  
    # Assign Monday's date to each date as 'WeekDate', this is needed for weekly waste plotting
    adj_dtls['WeekDate'] = [(d + dt.timedelta(days =- d.weekday())) for d in adj_dtls['Date']]

    # Drop records which has AdjustmentAmount = 0
    adj_dtls = adj_dtls[adj_dtls['AdjustmentAmount'] != 0]

    # Aggregate waste at week level for each item
    weekly_adj = adj_dtls.groupby(['ItemDescription', 'WeekDate']).sum()['AdjustmentAmount'].sort_index(ascending=True)
    weekly_adj_df = weekly_adj.to_frame().reset_index()

    # Get cumulative sum
    weekly_adj_df['AdjustmentAmountCum'] = weekly_adj_df.groupby(['ItemDescription'])['AdjustmentAmount'].apply(lambda x: x.cumsum())

    # Add adjustment(waste) weekly stats to master item weekly level data
    item_weekly_master = item_weekly_sales_df.merge(weekly_adj_df, on=['ItemDescription', 'WeekDate'], how='left')

    rolling_count.count = 0 #static variable
    rolling_count.previous = None #static variable

    item_weekly_master['SaleWeek'] = item_weekly_master['ItemDescription'].apply(rolling_count)
    item_weekly_master['SaleWeek'] = 'Week' + item_weekly_master['SaleWeek'].astype(str)

    # Take number of weeks to keep as a parameter and remove data for all subsequent weeks
    num_weeks_4_model = 4
    cols_to_keep = ['ItemDescription','QuantitySold','SaleWeek','OnPromotion', 'AdjustmentAmount']
    item_weekly_master_4_model = item_weekly_master[cols_to_keep].groupby('ItemDescription').head(num_weeks_4_model).reset_index(drop=True)

    # Transpose to get data at item level (unit of analysis) and each weekly measured attribute as columns (one column for each week) 
    item_master_4_model = item_weekly_master_4_model.pivot(index='ItemDescription', columns='SaleWeek', values=['QuantitySold', 'OnPromotion', 'AdjustmentAmount'])

    mi = item_master_4_model.columns
    ind = pd.Index([e[0] + '_' + e[1] for e in mi.tolist()])
    item_master_4_model.columns = ind
    item_master_4_model = item_master_4_model.reset_index()   

    # Add response variable
    item_master_4_model_final = pd.merge(item_master_4_model, item_sales_w_response[['ItemDescription','SalesSuccess']], on='ItemDescription')
    item_master_4_model_final = item_master_4_model_final.fillna(0)       

    # Drop irrelevant columns
    drop_cols = ['ItemDescription']
    drop_cols_pro = ['OnPromotion_Week1', 'OnPromotion_Week2', 'OnPromotion_Week3', 'OnPromotion_Week4']
    item_master_4_model_final = item_master_4_model_final.drop(drop_cols, axis=1)     

    # Write into csv
    item_master_4_model_final.to_csv(processed_data, sep=",", index=False)
                                                                                                             

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
    # build_features(config_path=parsed_args.config)