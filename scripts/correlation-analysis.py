import pandas as pd

# load the preprocessed dataset

file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data = pd.read_csv(file_path)

# calculate the correlation matrix

correlation_matrix = data.corr()

# display the correlation matrix

print("Correlation Matrix:")
print(correlation_matrix)

# save the correlation matrix to a file

output_path = "C:/puredata/statistical_methodology_correlation_matrix.csv"
correlation_matrix.to_csv(output_path, index=True)

print(f"Correlation matrix saved to: {output_path}")

