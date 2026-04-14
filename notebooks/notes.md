roc_auc: 0.9237840498493634, pr_auc: 0.23064750122616814, best_recall: 0.0 for logistic regression model 
model = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=4)


random forest was much better
  precision    recall  f1-score   support

           0       1.00      1.00      1.00    257834
           1       0.98      0.73      0.83      1501

    accuracy                           1.00    259335
   macro avg       0.99      0.86      0.92    259335
weighted avg       1.00      1.00      1.00    259335

roc_auc: 0.993237942470326, pr_auc: 0.9246769274524205, best_recall: 0.8481012658227848


but xgboost is safer for big datasets

