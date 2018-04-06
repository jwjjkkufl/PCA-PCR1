# PCA-PCR
'''
from __future__ import division
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.kde import gaussian_kde
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LassoCV

x=os.getcwd()
os.chdir('C:\Users\Wei Jiang\Desktop\Selvius')
os.chdir('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl')
xvar=pd.read_csv('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl\Xvar-full.csv')
yvar=pd.read_csv('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl\Wheat-full.csv')

xvar=xvar.dropna(axis=0)## remove all nas in dataframe
xvar=xvar.drop(xvar.index[0])
xvar=xvar.drop(xvar.columns[0],axis=1)
yvar=yvar.drop(yvar.index[0]) ## remove year 2006 to make it consistent with xvar
yvar=yvar.drop(yvar.columns[0],axis=1)
yvar=yvar.dropna(axis=0)


#Corn--2007 Lasso Regression
xvar=pd.read_csv('Xvars_price.csv')
xvar=xvar.dropna(axis=0)## remove all nas in dataframe
xvar=xvar.drop(xvar.columns[0],axis=1) #remove year
yvar=pd.read_csv("harvest_prices_main.csv")
yvar=yvar.dropna(axis=0)
yvar=yvar.drop(yvar.index[0])
yvar=yvar.drop(yvar.columns[0],axis=1)
y=yvar.iloc[:,2]


##Soybean---2006 Lasso Regression
xvar=pd.read_csv('Xvars_price - Copy.csv')
xvar=xvar.dropna(axis=0)## remove all nas in dataframe
xvar=xvar.drop(xvar.index[0])
xvar=xvar.drop(xvar.columns[0],axis=1) #remove year
yvar=pd.read_csv("harvest_prices_main.csv")
yvar=yvar.dropna(axis=0)
yvar=yvar.drop(yvar.columns[0],axis=1)
y=yvar.iloc[:,2]


## PCA analysis

X_std=StandardScaler().fit_transform(xvar)
sklearn_pca = sklearnPCA(n_components= 12)
xvar_reduced = sklearn_pca.fit_transform(X_std)

## if we want to perform PC-lasso--First PCA and then use lasso regression instead of linear regression
def lasso_regression(X, y, alpha_max):

    '''X: independent variables, y: dependent variable, alpha_max: upper bound of parsimony parameters to investigate (50 suggested)
    Cross validation is leave-one-out'''

    cv = cross_validation.KFold(len(y), n_folds=len(y), shuffle=True, random_state=0)
    
    #alpha-space to search
    alphas = np.concatenate((np.logspace(-4, -0.0001, 5000), np.linspace(1, alpha_max, 1000)))
    model = LassoCV(alphas = alphas, cv=cv, normalize = True).fit(X, y)
            
    #print 'alpha:'
    #print model.alpha_
    #recreate with optimal alpha level and test predictive accuracy
    alpha_model = sk.linear_model.Lasso(alpha = model.alpha_, normalize = True).fit(X, y)
    prediction=alpha_model.predict(X)
    
    #returns, for each element in the input, the prediction that was obtained for that element when it was in the test set
    predicted = cross_val_predict(alpha_model, X, y, cv=cv)
    r22_score = r2_score(y, predicted, multioutput = 'uniform_average')
    #This is the actual R2 between the observation and the prediction when it was left out of sample
    #print 'predictive r2:'
    #print r22_score
    
    return model.alpha_, predicted


a=[]
for i in np.arange(1,12):
      lasso=lasso_regression(xvar_reduced[:,:i], y.ravel(),50) 
      a.append(lasso)         

alpha, predicted=zip(*a) ## for lasso regression, three components has the highest R2    
predicted=pd.DataFrame(list(predicted)).T
print predicted
                      
n_components = len(predicted.columns)
y=np.array(y)
predicted['Corn']=pd.Series(y)
mae, mse = [],[]
r=[]
for i in range(0,n_components):
    resid = mean_absolute_error(predicted['Corn'], predicted[i])
    sq_resid = mean_squared_error(predicted['Corn'], predicted[i])
    r2=r2_score(predicted['Corn'], predicted[i])
    mae.append(resid)
    mse.append(sq_resid)
    r.append(r2)

print mae ## decide how many components need to be 
print r
print(np.argmin(mae))
predicted['error'] = predicted[np.argmin(mae)] - predicted['Corn']
predicted['pct_error'] = ((abs((predicted[np.argmin(mae)] - predicted['Corn']))/predicted['Corn']))*100
predicted["year"]=np.arange(2006,2018)


print predicted ## then turn to excel to peroform AE, select the model to have the samllest AE. the coeff is just regress the y over the chosen number of components
print('n_components in PCA = %d' % np.argmin(mae))
predicted.to_csv("Corn_PCA_Lasso.csv", index=False)

nxvar=pd.read_csv('fill - repeat-copy.csv')
nxvar=pd.read_csv('fill-repeat.csv')
nxvar=nxvar.dropna(axis=0)## remove all nas in dataframe
nxvar=nxvar.drop(["year"],axis=1)
X_std=StandardScaler().fit_transform(nxvar)
sklearn_pca = sklearnPCA(n_components= np.argmin(mae)+1)
xvar_reduced1 = sklearn_pca.fit_transform(X_std)





## lasso-prediction
def lasso_regression(X, y, alpha_max):

    '''X: independent variables, y: dependent variable, alpha_max: upper bound of parsimony parameters to investigate (50 suggested)
    Cross validation is leave-one-out'''

    cv = cross_validation.KFold(len(y), n_folds=len(y), shuffle=True, random_state=0)
    
    #alpha-space to search
    alphas = np.concatenate((np.logspace(-4, -0.0001, 5000), np.linspace(1, alpha_max, 1000)))
    model = LassoCV(alphas = alphas, cv=cv, normalize = True).fit(X, y)
            
    #print 'alpha:'
    #print model.alpha_
    #recreate with optimal alpha level and test predictive accuracy
    alpha_model = sk.linear_model.Lasso(alpha = model.alpha_, normalize = True).fit(X, y)
    prediction=alpha_model.predict(X)
    predicted = cross_val_predict(alpha_model, X, y, cv=cv)
    
    return alpha_model


lasso=lasso_regression(xvar_reduced[:,:np.argmin(mae)+1], y.ravel(),50) 
newdata=xvar_reduced1[12]
newdata=newdata.reshape(1,-1)
corn=lasso.predict(newdata) 
import statsmodels.stats.api as sms
sms.DescrStatsW(corn).tconfint_mean()
corn=pd.DataFrame(corn)
print corn
corn["year"]=np.arange(2006,2019)
corn.to_csv('Soybean-PCR-2006-LASSO.csv',index=False)



##wheat PCA+Linear Regression (1999)

from __future__ import division
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.kde import gaussian_kde
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LassoCV

x=os.getcwd()
os.chdir('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl')
xvar=pd.read_csv('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl\Xvar-full.csv')
yvar=pd.read_csv('C:\Users\Wei Jiang\Desktop\Selvius\Pulled more data from quandl\KW-full.csv')


xvar=xvar.drop(xvar.columns[0],axis=1)## remove year
yvar=yvar.drop(yvar.columns[0],axis=1)


y=yvar.iloc[:,0]
## PCA analysis

X_std=StandardScaler().fit_transform(xvar)
sklearn_pca = sklearnPCA(n_components= 15)
xvar_reduced = sklearn_pca.fit_transform(X_std)
# Leave one out validataion
n = len(y)  
kf_10 = cross_validation.KFold(n, n_folds=n, random_state=2) ##LeaveOneOut() is equivalent to KFold(n_splits=n) and LeavePOut(p=1) where n is the number of samples.
                        
#r-square
regr = LinearRegression()
b=[]
for i in np.arange(1,15):
      predictions = cross_val_predict(regr, xvar_reduced[:,:i], y.ravel(), cv=kf_10)
      b.append(predictions)
b=pd.DataFrame(b).T 
print b
#print b_custom ##  column is the number of component while the row is the predicted value
n_components = len(b.columns)
y=np.array(y)
b['Corn']=pd.Series(y)
mae, mse = [],[]
r=[]
for i in range(0,n_components):
    resid = mean_absolute_error(b['Corn'], b[i])
    sq_resid = mean_squared_error(b['Corn'], b[i])
    r2=r2_score(b['Corn'], b[i])
    mae.append(resid)
    mse.append(sq_resid)
    r.append(r2)

print mae ## decide how many components need to be 
print r
print(np.argmin(mae))
b['error'] = b[np.argmin(mae)] - b['Corn']
b['pct_error'] = ((abs((b[np.argmin(mae)] - b['Corn']))/b['Corn']))*100
b["year"]=np.arange(2006,2018)


print b ## then turn to excel to peroform AE, select the model to have the samllest AE. the coeff is just regress the y over the chosen number of components
print('n_components in PCA = %d' % np.argmin(mae))
b.to_csv('MW_expand.csv',index=False)



####Prediction

##Peform prediction with new xvar
nxvar=pd.read_csv('Xvar-full-copy.csv')
date=nxvar.iloc[:,0]
nxvar=nxvar.dropna(axis=0)## remove all nas in dataframe
nxvar=nxvar.drop(["year"],axis=1)
X_std=StandardScaler().fit_transform(nxvar)
sklearn_pca = sklearnPCA(n_components= np.argmin(mae)+1)
xvar_reduced1 = sklearn_pca.fit_transform(X_std)


regr = LinearRegression()
regr.fit(xvar_reduced[:,:np.argmin(mae)+1], y)
print('Coefficients: \n', regr.coef_)
predict=regr.predict(xvar_reduced1)
predict=pd.DataFrame(predict)


## check confidence interval

import statsmodels.api as sm
X2 = sm.add_constant(xvar_reduced[:,:np.argmin(mae)+1])
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
X3=sm.add_constant(xvar_reduced1)
predictions=est2.predict(X3)

predictions1 = est2.get_prediction(X3)
ci=predictions1.summary_frame(alpha=0.05)
ci1=ci.iloc[:,0:4]
ci1.columns = ['Price Prediction', 'Price Se','Lower CI','Updder CI']
ci1["Date"]=date
ci1.to_csv('KW-Price-Predict-Selvius.csv',index=False)
'''
