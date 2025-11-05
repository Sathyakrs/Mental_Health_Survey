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
- convert it into tensors and perform EDA to find the imbalance in the data


Logistic regression for making a benchmark: 

- Applied logistic regression with valuation metrics as accuracy, precision, recall and F1 score and confusion matrix
- Interpretation of the baseline evaluation has good accuracy, precision and recall, an F1 score of .8 and low false positives and negatives. We can still aim for a better recall by making we tunings so as to make it more precise, so that the model does not miss the depressed minority individual.
- performing class weight balance to minimise the wrong classification of the minority
- Tuning the threshold to increase the recall so that more depressed individuals will be identified
- performing ROC curve and selecting the optimal classification threshold using Youden’s J statistic
- This has provided us with a better recall and F1 score with relatively good precision, which can be used as a baseline. Since it's a medical case, it's better to have more false positives than fewer recalls


MLP Model, evaluation, training loop, early stopping:

- ensuring all data are in the same device (CPU or GPU)
- key configuration for training neural network model (input size, hidden layers, dropout point, learning rate, batch size, epochs, patience, model path)
- Perform data loaders
- defining the model with a sigmoid 
- criterion: loss function to measure the predictions 
- optimiser: optimising model weights to reduce loss
- scheduler: decreases lr rate if there is no improvement after n epochs 
- creating the evaluation model
- Print the value metrics (data was not great due to the imbalance very less true positives)
- Process training model to get the best model 
- printing final validation metrics of the best model 
- making a plot of training curves
- Save model weights
- Save preprocessing components
- Save the target feature order for safety

Streamlit application creation

- Create an app.py file
- Import the required libraries
- Set up the Streamlit page name and icon 
- Loading the model and preprocessor saved earlier
- Building the same model architecture as the MLP model 
- Creating a dummy data row with correct data types so that he columns are the same in both Streamlit and preprocessor
- Compute input dimension 
- Creating a Streamlit UI and setting up the user inputs for depression prediction
- creating the dataframe for the user input 
- creating the prediction option with the flow

Uploading the project un github repo


Deployment of the App in AWS

Creating EC2

- Go to your AWS Console → EC2 Dashboard.
- Click Launch Instance.
- Choose Amazon Machine Image (AMI): Ubuntu Server 22.04 LTS 
- Instance Type: t3.medium or t2.medium (2 vCPUs, 4 GB RAM minimum for PyTorch).
- Key Pair: Create a new key pair (e.g., aws-key.pem)- Download it and keep it safe — you’ll use it to SSH into the instance.
- Network Settings: Allow HTTP (port 80) and Custom TCP port 8501 (for Streamlit), also ensure SSH (port 22) is open to your IP.

Connecting to EC2 Instance

- Open the project folder in the terminal 
- run the AWS key with the instance IPv4
- install the dependencies (Python, git)
- Clone GitHub
- Open the mental health survey folder
- Install the requirements
- run the Streamlit application for depression
- Use nohup to keep the site running and you can see the output logs later 




