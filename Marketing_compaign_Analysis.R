# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
install.packages("rpart.plot")
library(rpart.plot)
# Load necessary libraries
library(tidyverse)
library(randomForest)
library(caret)
library(ROSE)
library(rpart)
library(pROC)

# Read CSV file
df <- read.csv("ifood_df.csv")

# Display first few rows of the dataset
head(df, 10)

# Get a concise summary of the dataframe
str(df)
summary(df)

# Calculate mean
colMeans(df, na.rm = TRUE)

# Drop columns
# We are dropping these columns since they have a unique value throughout, and hence will not be useful in our analysis
df <- subset(df, select = -c(Z_CostContact, Z_Revenue))

# Display updated information about the dataframe
str(df)

columns_to_keep <- c("Recency", "Customer_Days", "AcceptedCmpOverall", "Income", 
                     "MntRegularProds", "MntTotal", "MntWines", "MntMeatProducts", 
                     "MntGoldProds", "Age","Response")
df <- df[, columns_to_keep]

# Splitting data into training and testing sets
set.seed(123)  # For reproducibility
index <- createDataPartition(df$Response, p = 0.8, list = FALSE)
train_data <- df[index,]
test_data <- df[-index,]

# Oversampling the training data to balance the class distribution
train_data_balanced <- ovun.sample(Response ~ ., data = train_data, method = "both", p = 0.5, seed = 123)$data
# Verify the new class distribution
train_data_balanced$Response <- as.factor(train_data_balanced$Response)

#####
# Train Logistic Regression
logistic_model <- glm(Response ~ Recency + Customer_Days + AcceptedCmpOverall + Income + MntRegularProds + MntTotal + MntWines + MntMeatProducts + MntGoldProds + Age
                      , family = "binomial", data = train_data_balanced)

# Train Decision Tree
#tree_model <- rpart(Response ~ Recency + Customer_Days + AcceptedCmpOverall + Income + MntRegularProds + MntTotal + MntWines + MntMeatProducts + MntGoldProds + Age
#                   , data = train_data_balanced)
tree_model <- rpart(Response ~ Recency + Customer_Days + AcceptedCmpOverall + Income + MntRegularProds + MntTotal + MntWines + MntMeatProducts + MntGoldProds + Age
                    , data = train_data_balanced, method = "class", cp = 0.01, minsplit = 2, minbucket = 1) 



# Train the Random Forest model
rf_model <- randomForest(Response ~ ., data = train_data_balanced, ntree = 100)



# Predict probabilities on test data
logistic_pred <- predict(logistic_model, test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)
tree_pred <- predict(tree_model, test_data, type = "class")
rf_probabilities <- predict(rf_model, test_data, type = "prob")[,2]  # Assuming the second column is for the positive class

# Confusion matrices for each model
logistic_conf <- confusionMatrix(factor(logistic_pred_class), factor(test_data$Response))
tree_conf <- confusionMatrix(factor(tree_pred), factor(test_data$Response))
rf_conf <- confusionMatrix(as.factor(predict(rf_model, test_data, type = "class")), as.factor(test_data$Response))
#Confusion matrices for :
print(logistic_conf$table)
print(tree_conf$table)
print(rf_conf$table)


# Extracting and displaying metrics
metrics <- function(conf_mat) {
  data.frame(
    Accuracy = conf_mat$overall['Accuracy'],
    Sensitivity = conf_mat$byClass['Sensitivity'],
    Specificity = conf_mat$byClass['Specificity']
  )
}

logistic_metrics <- metrics(logistic_conf)
tree_metrics <- metrics(tree_conf)
rf_metrics <- metrics(rf_conf)

# Print the performance metrics
cat("Logistic Regression Metrics:\n")
print(logistic_metrics)
cat("\nDecision Tree Metrics:\n")
print(tree_metrics)
cat("\nRandom Forest Metrics:\n")
print(rf_metrics)


# Load required libraries
library(dplyr)      # For data manipulation
library(ggplot2)    # For data visualization
library(factoextra) # For clustering visualization
library(cluster)    # For clustering analysis

# Assuming your dataset is named "df"
# Select relevant features for clustering
cluster_data <- df %>%
  select(-Response) # Remove the response variable for clustering

# Standardize numerical variables
scaled_data <- scale(cluster_data)



# Determine the optimal number of clusters using the Elbow method
wss <- numeric(10) # Initialize within-cluster sum of squares vector
for (i in 1:10) {
  kmeans_model <- kmeans(scaled_data, centers = i)
  wss[i] <- kmeans_model$tot.withinss
}
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters",
     ylab = "Within-cluster Sum of Squares (WSS)",
     main = "Elbow Method for Optimal Number of Clusters")

# From the plot, determine the optimal number of clusters (e.g., based on the elbow point)

# Perform K-means clustering with the optimal number of clusters
k <- 4 # Update with the optimal number of clusters
kmeans_model <- kmeans(scaled_data, centers = k, nstart = 25) # Adjust nstart for multiple initializations


print(kmeans_model$centers)

# Add cluster labels to the original dataset
df$Cluster <- as.factor(kmeans_model$cluster)

ggplot(df, aes(x = Income, y = Age, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "Customer Segmentation by Income and Age",
       x = "Income", y = "Age") +
  scale_color_manual(values = c("red","blue","pink","green")) +  # Adjust colors if needed
  theme_minimal()

cluster_1 = df[df$Cluster==1,]
cluster_2 = df[df$Cluster==2,]
cluster_3 = df[df$Cluster==3,]
cluster_4 = df[df$Cluster==4,]

conversion_cluster_1 = 
  
  
  positive_response_rates <- numeric(k)
for (cluster in 1:k) {
  cluster_data <- df[df$Cluster == cluster, ]
  positive_responses <- sum(cluster_data$Response == 1)
  total_responses <- nrow(cluster_data)
  positive_response_rate <- (positive_responses / total_responses) * 100
  positive_response_rates[cluster] <- positive_response_rate
  cat("Cluster", cluster, ": Positive Response Rate =", positive_response_rate, "%\n")
}  


# Plotting the positive response rates for each cluster
barplot(positive_response_rates, 
        main = "Positive Response Rates by Cluster",
        xlab = "Cluster",
        ylab = "Positive Response Rate (%)",
        names.arg = 1:length(positive_response_rates),
        col = "skyblue",
        ylim = c(0, max(positive_response_rates) * 1.2),
        beside = TRUE)

text(x = 1:length(positive_response_rates), 
     y = positive_response_rates, 
     label = sprintf("%.2f%%", positive_response_rates), 
     pos = 3, 
     cex = 0.8, 
     col = "black", 
     offset = 0.5)

# Analyze cluster characteristics
cluster_summary <- df %>%
  group_by(Cluster) %>%
  summarize_if(is.numeric, list(mean = mean, sd = sd))  # Apply summary functions only to numeric variables

# Print summary statistics for each cluster
print(cluster_summary)





