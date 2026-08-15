import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import load_data, get_project_root

def get_charts_dir():
    project_root = get_project_root()
    charts_dir = os.path.join(project_root, "app", "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir

def basic_eda(df):
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nRows 25 to 35:")
    print(df.iloc[25:36])
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nComplete Information:")
    print(df.info())
    print("\nNumber of null values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("\nNumber of duplicate records:")
    print(df.duplicated().sum())
    print("\nTarget variable status:")
    count = df["PlacementStatus"].value_counts()
    print(count)
    
    plt.figure(figsize=(6, 5))
    plt.bar(count.index.astype(str), count.values)
    plt.title("Placement Prediction Distribution")
    plt.xlabel("Placement Status")
    plt.ylabel("Number of Records")
    
    charts_dir = get_charts_dir()
    plt.savefig(os.path.join(charts_dir, "Placement Prediction.png"))
    plt.close()

# Alias for backwards compatibility
basic_ede = basic_eda

def univariate(df):
    charts_dir = get_charts_dir()

    plt.figure(figsize=(6, 5))
    plt.hist(df["CGPA"], bins=10, edgecolor='black')
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(charts_dir, "Histogram of CGPA.png"))
    plt.close()

    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.pie(gendercount, labels=gendercount.index, autopct="%1.1f%%", startangle=90)
    plt.title("Gender Distribution Pie Chart")
    plt.savefig(os.path.join(charts_dir, "Gender Piechart.png"))
    plt.close()

# Alias for backwards compatibility
univariant = univariate

def bivariate(df):
    charts_dir = get_charts_dir()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"], alpha=0.5)
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(os.path.join(charts_dir, "CGPA vs APTITUDE.png"))
    plt.close()

    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
    plt.figure(figsize=(6, 5))
    plt.boxplot([placed, not_placed], tick_labels=["Placed", "Not Placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(os.path.join(charts_dir, "CGPA vs PLACEMENT.png"))
    plt.close()

def multivariate(df):
    charts_dir = get_charts_dir()

    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
    correlation = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(charts_dir, "HEATMAP.png"))
    plt.close()

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("All EDA charts generated and saved successfully!")