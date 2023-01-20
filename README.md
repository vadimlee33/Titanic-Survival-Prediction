# Titanic Survival Prediction
This is one of my first Machine Learning team projects.
In this project using ML we are trying to predict whether a person survived or not.
## Machine Learning Classification Algorithms we use during this project:
- LogisticRegression
- KNN
- Gradient Boosting
- Random Forest
- SVC
All algorithms were used from the scikit-learn open source library.
## Data Preprocessing:
- Data Imputation: MeanMedianImputer, ArbitraryNumberImputer, CategoricalImputer
- Feature Scaling: MinMaxScaler, StandardScaler, RobustScaler
- Encoding: RareLabelEncoder, OrdinalEncoder
## Data Visualization:
- Seaborn
- Matplotlib
## Metrics:
- Accuracy
- Recall
- Roc-auc
- F1
- Precision
- Confusion Matrix
## Other tools:
- Pipeline
- Pandas
- ColumnTransformer
- GridSearch

## Conclusion
Gradient Boosting Classifier shows a little bit better performance than other algorithms, but it has a huge overfit problem.

So we decided to take Logistic Regression, because this algorithm shows good performance without overfit problem.

Also, we obtain feature importances from coefficients and found that 'sex' and 'pclass' had the greatest impact on prediction .



