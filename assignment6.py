# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

"""
1. Add degree-2 polynomial features. The added features should contain degree-2 age,
degree-2 salary, and age times salary.
"""
from sklearn.preprocessing import PolynomialFeatures  
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
poly_feature.fit(X)
X_poly = poly_feature.transform(X)

"""
2. 25% of the data should go to the test set. In addition, random_state must be set to 0
in Python and seed must be set to 123 in R
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 
                                                    test_size = 1/4, random_state = 0)

"""
3. Feature scaling is required for both Python and R.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
4. Train your model based on the training set. Then, print out the confusion matrix and
accuracy based on the test set.
"""
# Training the model based on training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# print out confusion matrix accuracy
y_pred = classifier.predict(X_test)

# Show the confusion matrix and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

"""
5. A training set plot and a test set plot must be generated in each programming language.
The style of the plots should be identical to the one used in class. In each plot:
• The horizontal axis should be scaled age and the vertical axis should be scaled
salary.
• For background, use light red to represent the predicted region of “not purchased”
and use light green to represent the predicted region of “purchased”.
• Use red dots to represent “not purchased” observations and use green dots to
represent “purchased” observations.
• Have proper title and axis labels.
"""

# Visualizing the training set results using scaled features
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))

# Shape X1 and X2 into a two-column matrix
X_mat = np.column_stack((X1.ravel(), X2.ravel()))        

# Use existing poly feature to transform the resulting matrix
X_mat = poly_feature.transform(X_mat)

# Use existing standard scaler to inverse transform the resulting matrix
X_mat = sc.inverse_transform(X_mat)

# Extract first 2 columns
extracted_X_mat = X_mat[:, :2]

# Use existing polynomial features to again transform matrix
poly_extract_X_mat = poly_feature.transform(extracted_X_mat)

# Use existing standard scaler again to transform the resulting matrix
X_grid = sc.transform(poly_extract_X_mat)

# Printing the training set plot
plt.contourf(X1, X2, 
             classifier.predict(X_grid).reshape(X1.shape),
             alpha = 0.75, стар = ListedColormap(['red', 'green']))

for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(['red', 'green'])(i),
                edgecolors = 'black')
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.show()

# Printing the test set plot
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))

# Shape X1 and X2 into a two-column matrix
X_mat = np.column_stack((X1.ravel(), X2.ravel()))        

# Use existing poly feature to transform the resulting matrix
X_mat = poly_feature.transform(X_mat)

# Use existing standard scaler to inverse transform the resulting matrix
X_mat = sc.inverse_transform(X_mat)

# Extract first 2 columns
extracted_X_mat = X_mat[:, :2]

# Use existing polynomial features to again transform matrix
poly_extract_X_mat = poly_feature.transform(extracted_X_mat)

# Use existing standard scaler again to transform the resulting matrix
X_grid = sc.transform(poly_extract_X_mat)

# Printing the training set plot
plt.contourf(X1, X2, 
             classifier.predict(X_grid).reshape(X1.shape),
             alpha = 0.75, стар = ListedColormap(['red', 'green']))

for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(['red', 'green'])(i),
                edgecolors = 'black')
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.show()
