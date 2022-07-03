#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


#This makes it so that cell output displays all outputs, not just the last command called
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


holidays=pd.read_csv('holidays_events.csv')
oil=pd.read_csv('oil.csv')
stores=pd.read_csv('stores.csv')
sales_train=pd.read_csv('train.csv')
transactions=pd.read_csv('transactions.csv')


# # Holidays Info

# In[4]:


holidays.head()


# In[5]:


holidays.info()


# In[6]:


holidays.describe()


# # Oil Info

# In[7]:


oil.head()


# In[8]:


oil.info()


# In[9]:


oil.describe()


# # Stores Info

# In[10]:


stores.head()


# In[11]:


stores.info()


# In[12]:


stores.describe()


# # Transactions Info

# In[13]:


transactions.head()


# In[14]:


transactions.info()


# In[15]:


transactions.head()


# # Sales Train Info

# In[16]:


sales_train.head()


# In[17]:


sales_train.info()


# In[18]:


sales_train.describe()


# # Exploratory Analysis
# 
# 

# In[19]:


sales_train['family'].unique()
sales_train['family'].nunique()


# In[20]:


#Finding the family which has highest total sales
sales_train.groupby(['family'])['sales'].sum().sort_values(ascending=False)


# In[21]:


#Since the highest sales is Grocery I that is the one I will analyze first
#Make a scatter plot of Grocery I sales over time to get an idea of how the
#trends look

grocery1=sales_train[sales_train['family']=='GROCERY I'].groupby('date')['sales'].sum()
grocery1=grocery1.to_frame().reset_index()
grocery1


# In[22]:


grocery1_2013=grocery1[grocery1['date'].apply(lambda x: x.split('-')[0])=='2013']
grocery1_2013
plot1=sns.scatterplot(data=grocery1_2013)


# In[23]:


lm=LinearRegression()
X=grocery1_2013.index.values.reshape(-1,1)
y_train=grocery1_2013['sales']
lm.fit(X,y_train) 


# In[24]:


y_pred=lm.predict(X)
plot2=sns.lineplot(x=X.reshape(-1),y=y_pred,ax=plot1)
plot2


# In[25]:


sns.lineplot(x=X.reshape(-1),y=y_pred)


# In[26]:


fg = sns.lmplot(x=X.reshape(-1), y=grocery1_2013['sales'] )


# In[ ]:


signal = np.array([0,1,0,1], dtype=float)
fourier = np.fft.fft(signal)
n = signal.size
timestep = 0.1
np.fft.fftfreq(n, d=2)+0.25
np.fft.rfftfreq(n,d=2)


# In[ ]:


fourier


# In[ ]:


fig,ax=plt.subplots()


# In[ ]:


sns.scatterplot(x=X.reshape(-1),y=grocery1_2013['sales'],ax=ax)


# In[ ]:


sns.lineplot(x=X.reshape(-1), y=y_pred, ax=ax, palette='magma')


# In[ ]:


sales=grocery1_2013['sales']
sales2=sales-y_pred


# In[ ]:


fft=np.fft.rfft(sales2)


# In[ ]:


freq=np.fft.rfftfreq(n=len(sales2), d=1/365)
freq


# In[ ]:


fig1,ax1=plt.subplots()
ax1.step(freq, np.abs(fft)**2)


# In[ ]:


plt.step(x=freq, y=np.abs(fft)**2)


# In[ ]:


oil.head()
type(oil['date'].iloc[0])


# In[ ]:


multi=sales_train.set_index(['family'])
multi


# In[ ]:


from datetime import datetime

sales_train['date']=sales_train['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
sales_train['date']=sales_train['date'].apply(lambda x: x.date())


# In[ ]:


sales_train.head()


# In[ ]:


type(sales_train['date'][0])
type(multi['date'][0])
type(sales_train['store_nbr'][0])


# In[ ]:


plot_frame=sales_train[(sales_train['family']=='GROCERY I') & (sales_train['store_nbr']==1)]
plot_frame.head()
sns.scatterplot(x='date',y='sales', data=plot_frame)


# In[ ]:


plot_frame['family'].unique()


# In[ ]:


plots_dict=dict()


# In[ ]:


plots_dict.append('Hello','World')


# In[ ]:


family_plots=dict()
for family in [sales_train['family'].unique()[7]]:
    stores_plot=dict()
    for store_nbr in [sales_train['store_nbr'].unique()[0]]:
        plot_df=sales_train[(sales_train['family']==family) & (sales_train['store_nbr']==store_nbr)]
        plot=sns.lineplot(x='date', y='sales', data=plot_df)
        stores_plot[store_nbr]=plot
        del(plot)
    
    family_plots[family]=stores_plot
        


# In[34]:


years=sales_train['date'].apply(lambda x: x.year).unique()
store_numbers=sales_train['store_nbr'].unique()
families=sales_train['family'].unique()


# In[33]:


rows=pd.MultiIndex.from_product([years,families,store_numbers], names=['year','family','store_nbr'])


# In[32]:


models=pd.DataFrame(index=rows,columns=['model'])


# In[ ]:


models.head()


# In[ ]:


models.loc[2013:2015, :,1 ]


# In[ ]:


len(models.index.values)


# In[ ]:


get_ipython().run_cell_magic('capture', '', "for index in models.index.values:\n    year=index[0]\n    family=index[1]\n    store_nbr=index[2]\n    df=sales_train[sales_train['date'].apply(lambda x: x.year)==year]\n    X=df['date']\n    X=X.apply(lambda x: x.timetuple().tm_yday)\n    X=X.values.reshape(-1,1)\n    y=df['sales']\n    lm=LinearRegression()\n    lm.fit(X,y)\n    models.loc[index,'model']=lm\n")


# In[ ]:


models.loc[2013, 'AUTOMOTIVE', 12]['model'].intercept_


# In[ ]:


models['model'].nunique()


# In[27]:


sales_train['store_nbr'].nunique()


# In[28]:


sales_train.head()


# In[30]:


sales_train.to_pickle('sales_train.pkl')


# In[31]:


models.index.values


# In[ ]:


"

