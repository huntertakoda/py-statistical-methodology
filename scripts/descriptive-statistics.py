import pandas as pd

# load the preprocessed dataset

file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data = pd.read_csv(file_path)

# descriptive statistics summary

def descriptive_statistics(df):
    """Generate descriptive statistics for numeric columns."""
    stats = df.describe()
    print("Descriptive Statistics:")
    print(stats)
    return stats

# generate descriptive statistics

descriptive_stats = descriptive_statistics(data)

# save descriptive statistics to a CSV file

output_path = "C:/puredata/statistical_methodology_descriptive_statistics.csv"
descriptive_stats.to_csv(output_path)
print(f"Descriptive statistics saved to: {output_path}")
