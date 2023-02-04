# Credit-Card-Fraud-Analysis-Using-Machine-Learning

Credit Card Fraud Detection
Group 9:
Abhishek Sand (002752069)
Krupa Patel (002789566)

Introduction
As technology advances, the entire globe is moving toward digitalization. Credit card use has skyrocketed as a result of cashless purchases. The number of credit card accounts increased by 2.6% over the previous year. With the increased use of credit cards comes a rise in credit card fraud. Credit card fraud is a form of identity theft that causes massive personal financial deficits, corporate shortfalls, and national economic losses.
With the increased use of digital payment methods, the number of fraudulent transactions is expanding in novel and unexpected ways. Credit card fraud detection utilizing machine learning is not only a trend in the banking sector, but it is also a need for them to have proactive monitoring and fraud protection measures. Machine learning is assisting these institutions in reducing time-consuming manual checks, expensive chargebacks and fees, and valid transaction denials.
It becomes important to analyze credit card fraud as new data emerges. Understanding such trends using existing machine learning algorithms and models can aid in the creation of Automated Credit Card Fraud Detection Systems. The primary objective of this project is to utilize Machine learning-based techniques such as Artificial Neural Networks, Logistic Regression, and Random Forest, to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications and to provide a comparative study for performance analysis of the algorithms. This project also touches down upon Explainable AI in order to acquire a deeper understanding of the algorithms deployed, their expected impact, and any biases.

Methodology
This section examines the proposed method for detecting fraud using Machine Learning algorithms. We have approached this problem by dividing it into five fundamental steps:
•	Dataset: We gathered information from around 300,000 transactions. This dataset comprises 492 frauds out of 284,807 transactions that occurred during two days.
•	Exploratory Data Analysis and Pre-Prepossessing: In this step, we attempt to examine the data in order to identify significant aspects that will impact the output of our model and eliminate those that are superfluous. We will also standardize all the model's characteristics here.
•	Training and Testing Data: The whole amount of data obtained will be separated in 3:1 part: train data and test data. It indicates that 75% of the total data will be utilized to train the ML model, while the remaining 25% will be used to assess the model's accuracy.
•	Training Machine Learning Models: We used four different Machine Learning Algorithms and trained them all with the same data. These three algorithms are as follows:
o	Logistic Regression: Logistic regression is a widely used Machine Learning method that belongs to the Supervised Learning approach. It is used to predict the categorical dependent variable from a group of independent factors. As a result, the conclusion must be categorical or discrete. It can be No or Yes, 0 or 1, True or False, etc. but instead of giving the precise value like 0 and 1, it delivers the probability values that fall between 0 and 1, allowing it to classify whether the transaction is fraudulent or not.

o	Random Forest: Random Forest is a well-known machine learning algorithm from the supervised learning approach. It may be applied to both classification and regression issues in machine learning. Random Forest is a supervised machine learning technique that uses ensemble learning to mix many algorithms of the same type, i.e., multiple decision trees, resulting in a forest of trees, thus the name "Random Forest." When splitting a node, it looks for the best feature from a random subset of characteristics rather than the most essential feature. As a result, there is a greater variety, which leads to a better model. We utilize the algorithm for credit card fraud detection since it is not biased and offers advantages such as increased dimensionality and accuracy.

o	Artificial Neural Networks (ANNs): Artificial Neural Networks (ANNs) are machine learning algorithms that mimic the human nervous system. They process records one at a time and learn by comparing their (mostly arbitrary) categorization of the record to the known real classification of the record. The faults from the first record's categorization are sent back into the network and utilized to adjust the network's algorithm for subsequent rounds.

The neural network is trained to detect credit card fraud using previously collected data. This data includes the cardholder's occupation, income, credit card number, larger-amount transactions, frequency of transactions, location of where purchases are made, and information about past credit card purchases. Neural network uses this information to assess whether or not the cardholder is carrying out a specific transaction and whether the particular transaction is fraudulent or not.

Since the Neural network model is heavily reliant on hyperparameters, optimizing hyperparameters is crucial when building a neural network learning model.
•	Evaluating Results:
In this stage, we will test all four models with the remaining testing data to see which model will offer us with the most accurate results.




The project workflow is depicted in the figure below:
 
Figure 1: Methodology
Dataset
A number of datasets were reviewed and analyzed for the purpose of fraud detection and the credit card dataset, which is accessible on Kaggle. This dataset, made by European cardholders in September 2013, comprises 492 frauds out of 284,807 transactions that occurred during a two-day period. The Feature 'Class' is the response variable that takes the value of 1 in case of fraud and a value of 0 for non-fraud cases. The dataset is significantly skewed, with the positive class (frauds) accounting for 0.172% of all transactions.

Data Observation:
There is a noticeable statistical imbalance between the data points identified as fraudulent and those labeled as non-fraudulent. The use of data may result in the problem of overfitting. Undersampling, oversampling, and cross-validation can be used to resolve such issues.

Dataset Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

References:
[1] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[2]https://www.solver.com/xlminer/help/neural-networks-classification-intro#:~:text=Artificial%20neural%20networks%20are%20relatively,actual%20classification%20of%20the%20record.
[3] https://en.wikipedia.org/wiki/Logistic_regression
[4] https://en.wikipedia.org/wiki/Random_forest

