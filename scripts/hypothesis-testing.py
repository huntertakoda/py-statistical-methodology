import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# load the preprocessed dataset

file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data = pd.read_csv(file_path)

# hypothesis testing: t-test

def perform_t_test(df, group_column, numeric_column, group1, group2):
    """Perform an independent t-test between two groups."""
    group1_data = df[df[group_column] == group1][numeric_column]
    group2_data = df[df[group_column] == group2][numeric_column]

    t_stat, p_value = ttest_ind(group1_data, group2_data)
    print(f"T-Test Results for {numeric_column} by {group_column} ({group1} vs {group2}):")
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
    return t_stat, p_value

# example: t-test for income between regions

perform_t_test(data, 'region', 'income', 0, 1)

# hypothesis testing: chi-square test

def perform_chi_square_test(df, column1, column2):
    """Perform a chi-square test of independence between two categorical variables."""
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Test Results for {column1} and {column2}:")
    print(f"Chi2 Statistic: {chi2_stat}, P-Value: {p_value}, Degrees of Freedom: {dof}")
    return chi2_stat, p_value, dof, expected

# example: chi-square test for education level and region

perform_chi_square_test(data, 'education_level', 'region')

# save t-test and chi-square results to a file

output_path = "C:/puredata/statistical_methodology_hypothesis_testing_results.txt"
with open(output_path, "w") as f:
    f.write("T-Test Results:\n")
    t_stat, p_value = perform_t_test(data, 'region', 'income', 0, 1)
    f.write(f"T-Statistic: {t_stat}, P-Value: {p_value}\n\n")

    f.write("Chi-Square Test Results:\n")
    chi2_stat, p_value, dof, _ = perform_chi_square_test(data, 'education_level', 'region')
    f.write(f"Chi2 Statistic: {chi2_stat}, P-Value: {p_value}, Degrees of Freedom: {dof}\n")

print(f"Hypothesis testing results saved to: {output_path}")
