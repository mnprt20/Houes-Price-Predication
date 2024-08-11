#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


# In[4]:


data= pd.read_csv("C:\\project\\House Price India.csv")


# In[5]:


print(data)


# In[6]:


data.info()


# In[7]:


data.head()


# In[8]:


#checking the missing values 
data.isnull().sum()


# In[9]:


#statistical measure of dataset
data.describe()


# In[10]:


correaltion= data.corr()


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(correaltion , cbar = True, square=True, fmt='.1f',annot=True ,annot_kws ={'size':8},cmap='plasma')


# In[12]:


#splitting the data
X= data.drop(['Price'],axis=1)
Y= data['Price']


# In[13]:


print(X)
print(Y)


# In[14]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=1/3, random_state=0)


# In[15]:


print(X.shape,X_train.shape,X_test.shape)


# In[16]:


model= LinearRegression()
model.fit(X_train,Y_train)


# In[17]:


# prediction on training data
training_data_prediction = model.predict(X_train)


# In[18]:


print(training_data_prediction)


# In[19]:


# Accuray 
# R-squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R Square error:", score_1)
print("Mean Absolute error:", score_2)


# In[20]:


test_data_prediction = model.predict(X_test)


# In[21]:


# Accuray 
# R-squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R Square error:", score_1)
print("Mean Absolute error:", score_2)


# In[22]:


#visualizing actual price and predicted price
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[23]:


#Random forest not working 


# In[24]:


#decision tress


# In[25]:


X = data[['number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 'number of floors', 'waterfront present', 'number of views', 'condition of the house', 'Built Year', 'Renovation Year', 'Postal Code', 'Lattitude', 'Longitude', 'living_area_renov', 'lot_area_renov', 'Number of schools nearby', 'Distance from the airport']]
y = data['Price']


# In[26]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Postal Code'] = le.fit_transform(X['Postal Code'])
X = pd.get_dummies(X, columns=['condition of the house'])


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[28]:


from sklearn import tree


# In[29]:


clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, Y_train)


# In[30]:


print(clf.predict(X_test))


# In[31]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class HousePricePredictor:
    def __init__(self, data_file="C:\\project\\House Price India.csv"):
        self.data_file = data_file
        self.data = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data.dropna(inplace=True)

    def prep_data(self):
        X = self.data[['number of bedrooms', 'living area', 'number of floors']]
        y = self.data['Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_price(self, num_bedrooms, living_area, num_floors):
        data = pd.DataFrame({'number of bedrooms': [num_bedrooms], 'living area': [living_area], 'number of floors': [num_floors]})
        return self.model.predict(data)[0]

if __name__ == '__main__':
    hpp = HousePricePredictor()
    hpp.load_data()
    hpp.prep_data()
    hpp.train_model()

    num_bedrooms = int(input("Enter the number of bedrooms: "))
    living_area = float(input("Enter the living area in sqft: "))
    num_floors = int(input("Enter the number of floors: "))

    price = hpp.predict_price(num_bedrooms, living_area, num_floors)
    print(f'Predicted price for a house with {num_bedrooms} bedrooms, living area of {living_area} sqft, and {num_floors} floors is: {price:.2f}')


# In[ ]:




