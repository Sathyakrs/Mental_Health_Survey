 Data Preprocessing & EDA:

- Insert local data, ensure they are inserted properly
- Define target variable "Depression" 
- Find the null cells in this dataset
- Fill the values using median, mode, and mean functions
- Identify the numeric columns and normalise using the MinMaxScaler function
- Identify the categorical columns and use one-hot coding for them
- Split the data as two train and validation sets 
- Check if it's successful and for any overlap
- Fitting and preprocessing the numeric and categorical columns for deep learning
- convert it into tensors and performing EDA to find the imbalance in the data


Logistic regression for making a benchmark: 

- Applied logistic regression with valuation metrics as accuracy, precision, recall and F1 score and confusion matrix
- Interpretation of the baseline evaluation has good accuracy, precision and recall, an F1 score of .8 and low false positives and negatives. We can still aim for a better recall by making we tunings so as to make it more precise, so that the model does not miss the depressed minority individual.
- performing class weight balance to minimise the wrong classification of the minority
- Tuning the threshold to increase the recall so that more depressed individuals will be identified
- performing ROC curve and selecting the optimal classification threshold using Youden’s J statistic
- This has provided us with a better recall and F1 score with relatively good precision, which can be used as a baseline. Since it's a medical case, it's better to have more false positives than fewer recalls


MLP Model, evaluation, training loop, early stopping







