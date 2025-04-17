# logistic_regression_polynomial.R
#
# This script:
# 1. Loads Social_Network_Ads.csv
# 2. Adds degree‑2 polynomial features: age², salary², age×salary
# 3. Splits into 75% train / 25% test (seed = 123)
# 4. Scales all predictors
# 5. Trains a logistic regression
# 6. Prints confusion matrix & accuracy on test set
# 7. Plots decision regions & points for both training and test sets

#-------------------------------------------------------------------------------
# 0. Libraries
#-------------------------------------------------------------------------------
library(caTools)
library(caret)

#-------------------------------------------------------------------------------
# 1. Load data & feature engineering
#-------------------------------------------------------------------------------
dataset <- read.csv("Social_Network_Ads.csv")
dataset$age2        <- dataset$Age^2
dataset$salary2     <- dataset$EstimatedSalary^2
dataset$age_salary  <- dataset$Age * dataset$EstimatedSalary
dataset$Purchased   <- as.factor(dataset$Purchased)

#-------------------------------------------------------------------------------
# 2. Split into train/test
#-------------------------------------------------------------------------------
set.seed(123)
split      <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set     <- subset(dataset, split == FALSE)

#-------------------------------------------------------------------------------
# 3. Feature scaling (keep centers & scales for later)
#-------------------------------------------------------------------------------
# columns: 1=Age, 2=EstimatedSalary, 4=age2, 5=salary2, 6=age_salary
training_scaled_cols <- scale(training_set[, c(1,2,4,5,6)])
centers <- attr(training_scaled_cols, "scaled:center")
scales  <- attr(training_scaled_cols, "scaled:scale")

training_set[, c(1,2,4,5,6)] <- training_scaled_cols
test_set[, c(1,2,4,5,6)]     <- scale(
  test_set[, c(1,2,4,5,6)],
  center = centers,
  scale  = scales
)

#-------------------------------------------------------------------------------
# 4. Train logistic regression
#-------------------------------------------------------------------------------
classifier <- glm(
  formula = Purchased ~ .,
  family  = binomial,
  data    = training_set
)

#-------------------------------------------------------------------------------
# 5. Evaluate on test set
#-------------------------------------------------------------------------------
prob_pred <- predict(classifier,
                     type    = "response",
                     newdata = test_set)
y_pred <- as.factor(ifelse(prob_pred > 0.5, 1, 0))

cm <- confusionMatrix(y_pred, test_set$Purchased)
print(cm$table)
print(cm$overall["Accuracy"])

#-------------------------------------------------------------------------------
# 6. Helper function to plot decision regions
#-------------------------------------------------------------------------------
plot_decision <- function(dataset_scaled, title_text) {
  
  # Extract scaled coords
  X1 <- seq(min(dataset_scaled[,1]) - 1,
            max(dataset_scaled[,1]) + 1,
            by = 0.01)
  X2 <- seq(min(dataset_scaled[,2]) - 1,
            max(dataset_scaled[,2]) + 1,
            by = 0.01)
  
  # Build a grid of scaled points
  grid_coords <- expand.grid(X1, X2)
  names(grid_coords) <- c("x1","x2")
  
  # Un‑scale back to raw Age & Salary
  raw_age <- grid_coords$x1 * scales["Age"] +
    centers["Age"]
  raw_sal <- grid_coords$x2 * scales["EstimatedSalary"] +
    centers["EstimatedSalary"]
  
  # Re‑compute raw polynomial features
  raw_grid <- data.frame(
    Age             = raw_age,
    EstimatedSalary = raw_sal,
    age2            = raw_age^2,
    salary2         = raw_sal^2,
    age_salary      = raw_age * raw_sal
  )
  
  # Re‑scale all five columns
  grid_for_pred <- as.data.frame(
    scale(raw_grid,
          center = centers,
          scale  = scales)
  )
  
  # Predict on the grid
  prob_set <- predict(classifier,
                      type    = "response",
                      newdata = grid_for_pred)
  y_grid <- as.factor(ifelse(prob_set > 0.5, 1, 0))
  
  # Plot background
  plot(NULL,
       main = title_text,
       xlab = "Age (Scaled)",
       ylab = "Estimated Salary (Scaled)",
       xlim = range(X1),
       ylim = range(X2))
  points(grid_coords$x1,
         grid_coords$x2,
         pch = 20,
         col = c("tomato","springgreen3")[y_grid])
  
  # Overlay the actual datapoints
  points(dataset_scaled[,1],
         dataset_scaled[,2],
         pch = 21,
         bg  = c("red3","green4")[dataset_scaled$Purchased])
}

#-------------------------------------------------------------------------------
# 7. Visualize Training set
#-------------------------------------------------------------------------------
plot_decision(training_set, "Logistic Regression (Training set)")

#-------------------------------------------------------------------------------
# 8. Visualize Test set
#-------------------------------------------------------------------------------
plot_decision(test_set, "Logistic Regression (Test set)")

