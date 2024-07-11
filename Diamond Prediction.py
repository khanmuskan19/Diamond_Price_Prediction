#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# import matplotlib as mlt
import plotly.express as px


# In[2]:


data=pd.read_csv(r"C:\Users\Muskan Khan\Downloads\Diamond Prediction\diamonds.csv")
print(data)
data.head()


# In[3]:


data=data.drop("Unnamed: 0" ,axis=1)


# In[4]:


data.head()


# In[5]:


data["size"]= data['x']*data['y']*data['z']


# In[6]:


data.head()


# In[7]:



fig=px.scatter(data_frame=data, x='carat',y='price', color='color',size='depth', trendline='ols', title='Diamond Price vs. Carat Weight')
fig.show()


# In[8]:


# fig=px.bar(data_frame=data, x='carat', y='price', color='color', barmode='group',facet_col='clarity',
#              facet_row='depth', category_orders={ 
#                  'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
#                  'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
#              })
# fig.show()


# In[9]:


fig=px.scatter_matrix(data, dimensions=['carat', 'depth', 'table', 'price'],  # Use numeric columns
                        color='clarity') # or cut 
fig.show() #shows co-relation between two variables 


# In[10]:


fig=px.scatter_matrix(data, dimensions=['carat', 'depth', 'table', 'price'],  # Use numeric columns
                        color='cut') #IDK these two plots were necessary or not!
fig.show()


# In[11]:


fig=px.scatter(data, x='carat', y='price', size='size',color='clarity')
fig.show()


# In[12]:


fig=px.box(data,x='cut',y='price', color='color')
fig.show()


# In[13]:


fig=px.box(data,x='cut',y='price', color='clarity')
fig.show()


# In[14]:


fig=px.box(data,x='cut',y='price', color='color')
fig.show()


# In[15]:


correlation=data.corr()
print(correlation['price'].sort_values(ascending=False))


# In[16]:


data.head()


# # Price Prediction
# 

# In[17]:


data['cut']=data['cut'].map({
    "Ideal":1,
    "Premium":2,
    "Good":3,
    "Very Good":4,
    "Fair":5
})

data['clarity']=data['clarity'].map({
    "IF":1.0,
    "VVS1":2.1,
    "VVS2":2.2,
    "VS1":3.1,
    "VS2":3.2,
    "SI1":4.1,
    "SI2":4.2,
    "I1":5.1,
    "I2":5.2,
    "I3":5.3
})
data.head()
#pd.get_dummies() is used for not doing it manually!


# In[18]:


from sklearn.model_selection import train_test_split

x=np.array(data[['carat','cut',"clarity", "size"]])
y=np.array(data['price'])
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.10, random_state=42)


# In[19]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)


# In[20]:


print("Get Your Diamond Price:")
a=float(input("Carat Size:   "))
b=int(input( '''Cut Type:
Input your cut according to the given table:
    Ideal:1,
    Premium:2,
    Good:3,
    Very Good:4,
    Fair:5
'''))
c=float(input('''Clarity:
(Input your diamond's clarity according to the given table:
    IF:1.0,
    VVS1:2.1,
    VVS2:2.2,
    VS1:3.1,
    VS2:3.2,
    SI1:4.1,
    SI2:4.2,
    I1:5.1,
    I2:5.2,
    I3:5.3)'''))
d=int(input("Size of your Diamond:   "))
features=np.array([[a,b,c,d]])

predicted_price=model.predict(features)
price=predicted_price[0] # to convert it from list or arrays(to remove this[])
print(f"The Price of your Diamond is: {price}$")


# # Checking for Accuracy
# 

# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
y_pred=model.predict(x_test)



# In[25]:


mae=mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# In[24]:


mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:", mse)


# In[26]:


r2=r2_score(y_test,y_pred)
print("R² Score:", r2)


# In[ ]:




