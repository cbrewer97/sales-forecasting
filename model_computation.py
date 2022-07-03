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
    return datetime.strptime(x, '%Y-%m-%d').date()

def linreg_from_dict(values):
    lm=LinearRegression()
    lm.coef_=np.reshape(np.asarray(values['coef']),(-1,))
    lm.intercept_=values['intercept']
    return lm

def linreg_to_dict(lm):
    model_dict={'coef':lm.coef_, 'intercept':lm.intercept_}
    return model_dict

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

indices=models.index[models['model'].isnull()].values
    
counter=0
start=time.time()
for index in indices:
    counter+=1
    year=index[0]
    family=index[1]
    store_nbr=index[2]
    df=sales_train[sales_train['date'].apply(lambda x: x.year)==year]
    X=df['date']
    X=X.apply(lambda x: x.timetuple().tm_yday)
    X=X.values.reshape(-1,1)
    y=df['sales']
    lm=LinearRegression()
    lm.fit(X,y)
    models.loc[index,'model']=linreg_to_dict(lm)
    
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
        
    
        
