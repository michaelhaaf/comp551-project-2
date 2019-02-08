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
