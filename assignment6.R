#-------------------------------------------------------------------------------
# 1. Add degree-2 polynomial features. The added features should contain degree-2 age,
# degree-2 salary, and age times salary
#-------------------------------------------------------------------------------

#Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')

# add age degree-2
dataset$age2 = dataset$Age^2

# add salary degree-2
dataset$salary2 = dataset$EstimatedSalary^2

# add age times salary
dataset$age_salary = dataset$Age * dataset$EstimatedSalary

# Encoding categorical data
dataset$Purchased = as.factor(dataset$Purchased)

#-------------------------------------------------------------------------------
# 2. 25% of the data should go to the test set. In addition, random_state must be set to 0
# in Python and seed must be set to 123 in R.
#-------------------------------------------------------------------------------

# Splitting the dataset into Training and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#-------------------------------------------------------------------------------
# 3. Feature scaling is required for both Python and R.
#-------------------------------------------------------------------------------

# Feature Scaling
training_scaled_cols = scale(training_set[, c(1:2, 4:6)])
training_set[, c(1:2, 4:6)] = training_scaled_cols
test_set[, c(1:2, 4:6)] = scale(test_set[, c(1:2, 4:6)],
                        center = attr(training_scaled_cols, 'scaled:center'),
                        scale = attr(training_scaled_cols, 'scaled:scale'))

#-------------------------------------------------------------------------------
# 4. Train your model based on the training set. Then, print out the confusion matrix and
# accuracy based on the test set.
#-------------------------------------------------------------------------------

classifier = glm(formula = Purchased ~ .,
                        family = binomial,
                        data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set)
y_pred = as.factor(ifelse(prob_pred > 0.5, 1, 0))

# Showing the confusion matrix and accuracy
library(caret)
cm = confusionMatrix(y_pred, test_set$Purchased)
print(cm$table)
print(cm$overall['Accuracy'])

#-------------------------------------------------------------------------------
# 5. A training set plot and a test set plot must be generated in each programming language.
# The style of the plots should be identical to the one used in class. In each plot:
# • The horizontal axis should be scaled age and the vertical axis should be scaled
# salary.
# • For background, use light red to represent the predicted region of “not purchased”
# and use light green to represent the predicted region of “purchased”.
# • Use red dots to represent “not purchased” observations and use green dots to
# represent “purchased” observations.
# • Have proper title and axis labels
#-------------------------------------------------------------------------------

# Visualizing the Training set results
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(Age = X1, EstimatedSalary = X2)

# add features
grid_set$age2 = grid_set$Age^2
grid_set$salary2 = grid_set$EstimatedSalary^2
grid_set$age_salary = grid_set$Age * grid_set$EstimatedSalary

# scale the features only
grid_set[, c(1:5)] = scale(grid_set[, c(1:5)],
                        center = attr(training_scaled_cols, 'scaled:center'),
                        scale = attr(training_scaled_cols, 'scaled:scale'))

prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = as.factor(ifelse(prob_set > 0.5, 1, 0))
plot(NULL,
     main = 'Logistic Regression (Test set)',
     xlab = 'Age (Scaled)', ylab = 'Estimated Salary (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = 20, col = c('tomato', 'springgreen3')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4')[set$Purchased])