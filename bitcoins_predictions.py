import pandas as pd
#import the dataset
df=pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1081/challenge-2-bitcoin.csv")
df.head()
#focus the three useful columns and use them for a new dataset
data=df[['btc_market_price', 'btc_total_bitcoins', 'btc_transaction_fees']]
data.head()

from matplotlib import pyplot as plt
#plot the data from the three columns 
fig, axes= plt.subplots(1, 3, figsize=(16,5))
axes[0].plot(data['btc_market_price'])
axes[0].set_xlabel('time')
axes[0].set_ylabel('btc_market_price')
axes[1].plot(data['btc_total_bitcoins'])
axes[1].set_xlabel('time')
axes[1].set_ylabel('btc_total_bitcoins')
axes[2].plot(data['btc_transaction_fees'])
axes[2].set_xlabel('time')
axes[2].set_ylabel('btc_transaction_fees')

#split the data for training and testing
split_num=int(len(data)*0.7)
train_data=data[:split_num]
test_data=data[split_num:]
train_x=train_data[['btc_total_bitcoins', 'btc_transaction_fees']]
train_y=train_data[['btc_market_price']]
test_x=test_data[['btc_total_bitcoins', 'btc_transaction_fees']]
test_y=test_data[['btc_market_price']]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#First we use three degree polynomial to build the model
def poly3():
    poly_model=PolynomialFeatures(degree=3, include_bias=False)
    poly_train_x=poly_model.fit_transform(train_x)
    poly_test_x=poly_model.fit_transform(test_x)
    model=LinearRegression()
    model.fit(poly_train_x,train_y)
    pred_y=model.predict(poly_test_x)
    mae=mean_absolute_error(test_y, pred_y)
    return mae
mae_result=poly3()
print(mae_result)

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
#Use a pipeline to build a model with N degree polynomial
def poly_plot(N):
    m=1
    mse=[]
    while m<=N:
        model=make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
        model.fit(train_x,train_y)
        pred_y=model.predict(test_x)
        mse.append(mean_squared_error(test_y, pred_y))
        m+=1
    return mse
mse=poly_plot(10)

#make a plot to compare the mse for model with 1 to 10 degree polynomial
plt.plot(mse, '.-')
plt.title('MSE')
plt.xlabel('N')
plt.ylabel('MSE')

