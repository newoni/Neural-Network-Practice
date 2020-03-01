# <20.2.28> by KH

'''
23 page
Supervised learning algorithms
'''

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Using a standard dataset that we ca find in scikit-learn
cal_house = fetch_california_housing()

# Split the input data into training/testing sets
cal_house_X_train = cal_house.data[:-20]
cal_house_X_test = cal_house.data[-20:]

# Split the targets into training/testing sets
cal_house_Y_train = cal_house.target[:-20]
cal_house_Y_test = cal_house.target[-20:]

# Created linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(cal_house_X_train, cal_house_Y_train)

# Calculating he predictions
predictions = regr.predict(cal_house_X_test)

#Calculating the loss
print('MSE: {:.2f}'.format(mean_squared_error(cal_house_Y_test,predictions)))
