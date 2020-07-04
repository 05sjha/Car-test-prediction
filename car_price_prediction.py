# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:42:36 2020

@author: DELL
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor as xtr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle





#data read and manipulation
df=pd.read_csv('car data.csv')
# print(df.shape)
# print('original data \n',df.head)
data_col=[]
for i in df.columns: 
    data_col.append(i)
# print(data_col)
data_col.remove('Car_Name')
# print(data_col)
df_new=df[data_col]
# print(df_new.head())

df_new['current_year']=2020
df_new['car_age']=df_new['current_year']-df_new['Year']
df_new.drop(['Year','current_year'],axis=1,inplace=True)
print('data column modified \n',df_new.head())
df_new=pd.get_dummies(df_new,drop_first=True)
print('dataset modified with onehot-encoding',df_new.head())

#data visualization
# plt.figure()
# sns.pairplot(df_new)
cmat=df_new.corr()
corr_ft=cmat.index
plt.figure(figsize=(20,20))
#heatmap
fig=sns.heatmap(df_new[corr_ft].corr(),annot=True,cmap="RdYlGn")


#independent feature= all but selling price (x)
#dependant feature= selling price (y)

x=df_new.iloc[:,1:]
y=df_new.iloc[:,0]

#xtr or ****ExtraTreesRegressor*****
# is an algo to find out the importance of the features
model=xtr()
model.fit(x,y)
ft_imp=pd.Series(model.feature_importances_, index=x.columns)
ft_imp.nlargest(5).plot(kind='barh')
plt.show()

#model is random forest regressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#*********HYPER PARAMETER TUNING*************

#no. of trees
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)] 

#no. of features to consider every split
max_features=['auto','sqrt']

#max no. of levels in tree
max_depth=[int(i) for i in np.linspace(5,30,num=6)]

#min number of sampples req to split a node
min_samples_split=[2,5,10,15,100]

#min no. of samples req at each leaf node
min_samples_leaf=[1,2,5,10]
rand_grid={'n_estimators':n_estimators,
           'max_features':max_features,
           'max_depth':max_depth,
           'min_samples_split':min_samples_split,
           'min_samples_leaf':min_samples_leaf}
print(rand_grid)




#*********MODEL DESIGN*************

rf=RandomForestRegressor()
rf_rand=RandomizedSearchCV(estimator=rf, param_distributions=rand_grid,
                           scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,
                           random_state=42,n_jobs=1)
rf_rand.fit(x_train,y_train)


#PREDICTION
pred=rf_rand.predict(x_test)
print(pred)
sns.distplot(y_test-pred)
plt.figure()
plt.scatter(y_test,pred)

trained_model=open('random_forest_regression.pkl','wb') # create a file to dump the model
pickle.dump(rf_rand,trained_model) # dumping the model in created file







 































