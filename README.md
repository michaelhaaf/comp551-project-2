# comp551-project-2

# Usage instructions:
cd ./source

python3 ./source/runner.py --selected_features both_gram --selected_classifier mnb

# Options
-- selected_features: convenience for choosing between defined features at command line. Multiple features can be combined (separated by spaces)

-- selected_classifer: choose which classifier to run on this test run

-- perform_cv: run cross-validation on this test run

# Results
results are printed to timestamped directories within the ./results directory. ROC curve, accuracy/FI, and the submission file are automatically generated here.

# Custom Naive Bayes Instructions:

Before Running the code you need to have you train test split ready
The X and Y values are converted to list for convenience (from numpy)
Naive Bayes Run Uses CountVectorizer to create an input and the creates a NB classifier object
that object is the fitted into the training data which is directly followed by a prediction
for the parameters in these files an accuracy of 87.74 was obtained

