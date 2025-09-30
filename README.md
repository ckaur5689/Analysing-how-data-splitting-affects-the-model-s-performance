# Analysing-how-data-splitting-affects-the-model-s-performance

# Introduction:

Understanding the balance between training, validation and test sets is key to building ML models that generalise well beyond a single data set. In this activity, you will analyse how different data split strategies affect the model's performance.

# Instructions:

This activity has two parts. You’ll first set up and run experiments in a Google Colab, and then you’ll reflect on the results.

# Part 1: running experiments in Google Colab.
Open a Jupyter notebook in Google Colab. For the steps required to open and set up your Colab notebook, please see the Colab guide Download Colab guidefrom the orientation module.
Copy and paste the code blocks below into your Jupyter notebook. This code creates an initial split of 70% training, 15% validation and 15% test.
Adjust the hyperparameter ratio to 60:20:20 (training:validation:test).
Note: To create a final split of 60% training, 20% validation and 20% test, the data is divided in two steps. First, set aside 20% for testing. Then, from the remaining 80%, take 25% for validation (since 25% of 80% = 20% of the total). This ensures the final split is 60:20:20.
Run the code, and observe the model's performance.
Input the following code blocks:

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the default wine dataset
data = load_wine()
X = data.data
y = data.target

# Split data into train (70%), validation (15%) and test (15%)sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)
# 0.1765 * 0.85 ≈ 0.15, so validation is ~15% of total

# Fit the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the validation set
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate the test set
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report on test set
print("\nClassification Report on Test Set:")
print(classification_report(y_test, test_preds, target_names=data.target_names))

# Part 2: reflect on the results.

Now that you've experimented with different data splits and observed the model's performance, reflect on your findings by answering the following questions:

# a) What impact does the 60:20:20 split have on model accuracy? There is less data for the model to train on (reduced from 70% to 60%).
Although the Validation Accuracy has reduced from 1 to 0.9444 (implying there were data points in the training data set that could not be validated), 
the Test Accuracy increased from 0.9630 to 0.9722. The increase in Test Accuracy implies that the 60:20:20 is more robust to new data (test data) 
and can learn (therefore it is a generalisable model). 

# b) How does the model's performance change if you use a 70:15:15 split?
Changing from 60:20:20 to 70:15:15 split; overall the Validation Accuracy increases from 0.9444 to 1.000 and produced a lower Test Accuracy metric
(reducing from 0.9722 to 0.9630). This illustrates that the 70:15:15 split produced a large degree of overfitting in the training to validation phase]
and therefore makes it vulnerable to unseen (test) data so the model produced would not generalise well to real/new data. 
We in particular see that when predicting class_2, the f1-score has reduced from 0.95 to 0.92 with a 70:15:15 split, so the models’ ability to classify
class_2 accurately has reduced by 3%.

# c) What might happen if you omitted the validation set and only used training and testing data?
Without a validation set, the model is tested on the same data used for training and tuning. This creates a highly favourable, but inaccurate, assessment of its performance,
as it doesn't reflect how well the model will work on truly unseen data. 

It is also impossible to reliably identify overfitting, a situation where the model has memorised the training data rather than learning generalisable patterns. 
A validation set is crucial for estimating a model's ability to generalise to new, independent data. Its exclusion means you lose this vital insight into the model's 
true predictive power. 

Interesting publication: https://www.sciencedirect.com/science/article/pii/S0003267023007535
The importance of choosing a proper validation strategy in predictive models. A tutorial with real examples - ScienceDirect

# d) How can you apply what you've learned from experimenting with different data splits and model types to improve your capstone project's model performance and reliability?

To improve my capstone project, I would firstly evaluate whether Machine Learning can be applied to the problem; does it meet the 3 requirements
(probabilistic setting, stationary and a priori knowledge). If these are met (may require some data transformations), I would experiment with different
data splits (train, validation and test) to ensure I am creating an optimal model which is generalisable. 
For classification models, I would evaluate the model performance using Validation Accuracy, Test Accuracy and Classification Reports. 
Also understand which of these metrics (Recall or Sensitivity, Specificity, Precision, Negative Prediction Value and Accuracy) matter to my classification problem.

For regression models, I would use measures such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE) and Adjusted R2  to evaluate the different splits.

For both, I would keep in mind the Bias_Variance Tradeoff, and plot this chart, opting for optimal model complexity.

