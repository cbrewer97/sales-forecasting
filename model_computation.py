# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:25:22 2022

@author: books
"""

import pandas as pd
import numpy as np

from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
            
        

def main():
    sales_train=load_sales()
    models=load_models(sales_train)
    compute_linear_models(sales_train, models)
    
main()
        
