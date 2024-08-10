# Marketing Campaign Analysis: Decoding Customer Response for Profitable Campaigns
## Introduction:

Have you ever wondered which customers are most likely to respond to your food company's marketing efforts? We embarked on a data-driven quest to answer just that!

## The Challenge:

Predicting customer response to marketing campaigns is crucial for maximizing profitability.  Our goal was to unlock the key customer characteristics that influence their reaction to food company campaigns.
![](https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/08/customerbehavior-scaled.jpg)
Image source: [How to Analyze and Predict the Behavior of Consumers](https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/08/customerbehavior-scaled.jpg)

## The Data:

We delved into a rich dataset of over 2,200 customers, encompassing 39 variables and revealing their socio-economic and firmographic profiles. However, duplicate entries required attention, and the response variable itself was imbalanced, demanding a strategic approach.  (Full dataset available: https://github.com/nailson/ifood-data-business-analyst-test/tree/master)

## Unveiling the Secrets:

Our analysis focused on the 10 most impactful customer features, providing deep insights into their demographics and behaviors.  We employed various techniques to extract these secrets:

* **Descriptive Analysis:** We cleaned the data, analyzed response rates across age and income groups, and meticulously explored customer profiles.
* **Machine Learning Models:** We unleashed the power of machine learning to predict customer responses:
  * **Random Forest Classifier:** This champion model emerged with the highest accuracy, sensitivity, and AUC (Area Under the Curve), making it the MVP for overall customer response prediction.
  * **Logistic Regression:** While not the top scorer, this model offered valuable insights into how individual features influence customer behavior.
  * **Decision Tree Classifier:** This transparent model provided a clear picture of the decision-making process behind customer responses.
* **K-Means Clustering:** To identify prime targets for our marketing campaigns, we utilized K-means clustering. Here's what we discovered:
  * High-Income Responders: This cluster, boasting an average income of $78,800, displayed the highest response rate.
  * Big Spenders: This group also exhibited the highest spending across diverse product categories.
  * Campaign Enthusiasts: These customers demonstrated a strong positive response to various marketing initiatives.

## The Golden Ticket:

This analysis empowered us to achieve significant results:

* Predictive Power: We built an accurate customer response forecasting system using the Random Forest model, achieving an impressive 87% accuracy.
* Key Customer Insights: We identified high-income customers with strong purchasing power, high brand engagement, and a positive attitude towards marketing campaigns as the most receptive group.
* Profitable Targeting: By tailoring marketing strategies and offers to these customer segments, we can maximize response rates and ultimately, profitability.

## The Final Bite:

This journey through customer data has yielded invaluable insights. By understanding customer behavior and characteristics, we can craft more effective marketing campaigns that resonate with the right audience, driving success for the food company.


