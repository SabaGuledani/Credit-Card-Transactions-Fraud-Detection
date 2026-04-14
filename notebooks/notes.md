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
	model_name	eta	max_depth	n_estimators	gamma	subsample	min_child_weight	colsample_bytree	reg_alpha	roc_auc	pr_auc	best_recall
1254	xgboost_85	0.10	9	700	0.1	0.9	5	0.9	0.3	0.999299	0.957614	0.852099

best set of results
for test set
roc_auc: 0.9983474779521921, precision recall score: 0.9239965370385764, best recall at 0.95 precision: 0.7883449883449883