# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:25:22 2022

@author: books
"""

import pandas as pd
import numpy as np

from datetime import datetime,timedelta
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.fft import rfft, rfftfreq, irfft


#holidays=pd.read_csv('holidays_events.csv')
#oil=pd.read_csv('oil.csv')
#stores=pd.read_csv('stores.csv')
#transactions=pd.read_csv('transactions.csv')
#transactions.head()

def str_date(x):
    """Convenience method for converting string of type YYYY-MM-DD to date"""
    return datetime.strptime(x, '%Y-%m-%d').date()

def linreg_from_dict(values):
    """Converts dict to a LinearRegression object from sklearn by accessing the
    dictionary keys coef and intercept"""
    lm=LinearRegression()
    lm.coef_=np.reshape(np.asarray(values['coef']),(-1,))
    lm.intercept_=values['intercept']
    return lm

def linreg_to_dict(lm):
    """  Converts LinearRegression object to dict by storing intercept_ and coeff_
    attributes in the dict using coef and intercept as keys"""
    model_dict={'coef':lm.coef_, 'intercept':lm.intercept_}
    return model_dict

def load_sales():
    """Tries to load sales_train dataframe from a feather file. Otherwise, it will
    load the data from csv file, and then save to feather file. Returns the
    sales_train dataframe"""
    try:
        sales_train=pd.read_feather('./sales_train.ftr')
        print('Loaded sales_train dataframe.')
    except:
        print('File not found. Loading from CSV.')
        sales_train=pd.read_csv('train.csv')
        sales_train['date']=sales_train['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
        print('Successfully converted dates.')
        sales_train.to_feather('./sales_train.ftr')
        print('Loaded sales_train. Converted date to type Date. Saved to feather.')

    return sales_train

def load_models(sales_train):
    """Requires argument of sales_train dataframe (it uses this to create the
    MultiIndex). Tries to load models dataframe from feather and set multi-index
    on year, family, and store_nbr. Otherwise, creates empty dataframe
    and saves to feather. Returns the models dataframe"""
    try:
        models=pd.read_feather('models.ftr')
        models=models.set_index(['year','family','store_nbr'])
        #models[models['model'].notnull()]=models[models['model'].notnull()].apply(lambda x: linreg_from_dict(x))
        print('Loaded models dataframe.')
    except:
        print('sales_train_models.ftr not found. Creating dataframe: \'models\'')
        
        years=sales_train['date'].apply(lambda x: x.year).unique()
        store_numbers=sales_train['store_nbr'].unique()
        families=sales_train['family'].unique()
        
        rows=pd.MultiIndex.from_product([years,families,store_numbers], names=['year','family','store_nbr'])
        
        models=pd.DataFrame(index=rows,columns=['model'])
        
        models.reset_index().to_feather('models.ftr')
        #models.to_feather('models_backup.ftr')
        
        print('Created empty models dataframe and saved to feather.')
        
    return models




def compute_linear_models(sales_train, models):
    indices=models.index[models['model'].isnull()].values
    counter=0
    start=time.time()
    for index in indices:
        counter+=1
        year=index[0]
        family=index[1]
        store_nbr=index[2]
        
        years_bool=sales_train['date'].apply(lambda x: x.year)==year
        store_bool=sales_train['store_nbr']==store_nbr
        family_bool=sales_train['family']==family
        
        df=sales_train[years_bool & store_bool & family_bool]
        X=df['date']
        X=X.apply(lambda x: x.timetuple().tm_yday)
        X=X.values.reshape(-1,1)
        y=df['sales']
        lm=LinearRegression()
        lm.fit(X,y)
        models.loc[index,'model']=[linreg_to_dict(lm)]
        
        if counter==10:
            end=time.time()
            el_time=end-start
            start=time.time()
            counter=0
            print('Completed 10 more regressions')
            print('  Elapsed time: ', el_time)
            print('  Average time per model: ', round(el_time/10,3))
            models.reset_index().to_feather('models.ftr')
            #models.to_feather('models_backup.ftr')
            
def remove_linear_trend():
    return 0
    
def restore_linear_trend():
    return 0

            
def truncate_fourier(fourier,n):
    """Keeps only the coefficients with magnitude greater
    than or equal to the n-th highest magnitude. I.e. if the
    n-th highest magnitude appears more than once, all coefficients
    with that magnitude are kept. If all magnitudes are unique,
    only n coefficients are kept."""
    for i,coef in enumerate(fourier):
        if abs(coef)<np.flip(np.sort(abs(fourier)))[n-1]:
            fourier[i]=0

def compute_fft(sales_train, models):
    
    if 'fourier' not in models.columns:
        print('models dataframe has no column \'fourier\'.')
        models['fourier']=np.nan
        print('Created column: fourier')
        
    indices=models.index[models['fourier'].isnull()].values
    counter=0
    start=time.time()
    for index in indices:
        counter+=1
        year=index[0]
        family=index[1]
        store_nbr=index[2]
        
        years_bool=sales_train['date'].apply(lambda x: x.year)==year
        store_bool=sales_train['store_nbr']==store_nbr
        family_bool=sales_train['family']==family
        
        df=sales_train[years_bool & store_bool & family_bool]
        #X=df['date']
        #X=X.apply(lambda x: x.timetuple().tm_yday)
        #X=X.values.reshape(-1,1)
        y=df['sales'].values
        
        fourier=rfft(y)
        #Instead of saving only the truncated coefficients,
        #we will save all coefficients. Truncation will
        #take place in the prediction step
        
        real=fourier.real
        imag=fourier.imag
        fourier_dict={'real':real, 'imag':imag}
        
        #Have to wrap dict in list or else assignment doesn't work
        #I.e. pandas tries to iterate the dict instead of assigning 
        # to single cell
        models.at[index,'fourier']=[fourier_dict]
        
        
        if counter==10:
            end=time.time()
            el_time=end-start
            start=time.time()
            counter=0
            print('Completed 10 more rfft')
            print('  Elapsed time: ', el_time)
            print('  Average time per model: ', round(el_time/10,3))
            models.reset_index().to_feather('models.ftr')
            #models.to_feather('models_backup.ftr')

def lm_predict(sales_train, models):
    counter=0
    start=time.time()
    for index in models.index.values:
        counter+=1
        values_dict=models.loc[index]['model'][0] #the [0] is necessary since the dict is wrapped in a list
        lm=linreg_from_dict(values_dict) 
        
        year=index[0]
        family=index[1]
        store_nbr=index[2]
        
        years_bool=sales_train['date'].apply(lambda x: x.year)==year
        store_bool=sales_train['store_nbr']==store_nbr
        family_bool=sales_train['family']==family
        
        df=sales_train[years_bool & store_bool & family_bool]
        X=df['date']
        X=X.apply(lambda x: x.timetuple().tm_yday)
        X=X.values.reshape(-1,1)
        
        y_pred=lm.predict(X)
        
        X=pd.DataFrame(X)
        y_pred=pd.DataFrame(y_pred)
        pred_df=X.join(y_pred,lsuffix='left', rsuffix='right')
        pred_df.rename(columns={'0left':'date', '0right':'lm_pred'}, inplace=True)
        #print(pred_df.head())
        pred_df['date']=pred_df['date'].apply(lambda x: datetime(year,1,1)+timedelta(x-1))
        pred_df['family']=family
        pred_df['store_nbr']=store_nbr
        #print(pred_df['date'].info())
        
        #sales_train['date']=sales_train['date'].astype('datetime64')
        
        #sales_train=pd.merge(sales_train, pred_df,how='left', left_on=['date','family','store_nbr'], right_on=['date','family','store_nbr'])
        #Need to cast to list so that it will assign to whole slice. Leaving as a series causes only one cell to get assigned
        sales_train.loc[years_bool & family_bool & store_bool,'lm_pred']=pred_df['lm_pred'].tolist() 
        print(sales_train.loc[years_bool & family_bool & store_bool,'lm_pred'])
        print(pred_df['lm_pred'])
        if counter==10:
            end=time.time()
            el_time=end-start
            start=time.time()
            counter=0
            print('Completed 10 more rfft')
            print('  Elapsed time: ', el_time)
            print('  Average time per model: ', round(el_time/10,3))
            sales_train.to_feather('sales_train.ftr')
            
    sales_train.to_feather('sales_train.ftr')
    print('Completed all predictions.')
        
        

    

def main():
    sales_train=load_sales()
    models=load_models(sales_train)
    compute_linear_models(sales_train, models)
    
#main()
        
