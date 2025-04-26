install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("dplyr")

# Load libraries
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)

data <- read.csv("online_shoppers_intention.csv")
head(data)

# Convert 'Revenue' (target variable) to a factor for classification
data$Revenue <- as.factor(data$Revenue)

# Convert categorical features to factors
data$Month <- as.factor(data$Month)
data$VisitorType <- as.factor(data$VisitorType)
data$Weekend <- as.factor(data$Weekend)

# Split the data into training (70%) and testing (30%) sets
set.seed(123)  # Ensure reproducibility
trainIndex <- createDataPartition(data$Revenue, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train a decision tree model using relevant features
model <- rpart(Revenue ~ Administrative + Administrative_Duration + 
                 Informational + Informational_Duration + ProductRelated + 
                 ProductRelated_Duration + BounceRates + ExitRates + PageValues + 
                 SpecialDay + Month + OperatingSystems + Browser + Region + 
                 TrafficType + VisitorType + Weekend, 
               data = trainData, method = "class")

# Visualize the Decision Tree
rpart.plot(model, main = "Decision Tree for Online Shopping Purchase Prediction", type = 3, extra = 101)

# Make predictions on the test set
predictions <- predict(model, testData, type = "class")

# Evaluate model accuracy using a confusion matrix
confMatrix <- confusionMatrix(predictions, testData$Revenue)
print(confMatrix)

# Display feature importance
print(model$variable.importance)

##########

# Convert to data frame
importance_df <- data.frame(Feature = names(model$variable.importance),
                            Importance = model$variable.importance)

# Sort by importance (optional)
importance_df <- importance_df[order(-importance_df$Importance), ]

# Print as table
print(importance_df, row.names = FALSE)

# If not installed, install knitr
install.packages("knitr")
library(knitr)

kable(importance_df, caption = "Feature Importance")


