import pandas as pd
import statsmodels.api as sm

# load the preprocessed dataset

file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data = pd.read_csv(file_path)

# prepare data for regression analysis

X = data[['age', 'income', 'hours_worked_per_week', 'job_satisfaction', 'mental_health_score']]
y = data['physical_health_score']

# add a constant term for the intercept

X = sm.add_constant(X)

# fit the regression model

model = sm.OLS(y, X).fit()

# summarize the regression results

regression_summary = model.summary()
print(regression_summary)

# save the regression summary to a file

output_path = "C:/puredata/statistical_methodology_regression_summary.txt"
with open(output_path, "w") as f:
    f.write(str(regression_summary))

print(f"Regression analysis results saved to: {output_path}")
