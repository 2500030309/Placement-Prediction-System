import seaborn as sns
from src.data.load_data import load_data
import matplotlib.pyplot as plt

def basic_ede(df):
    print(df.head())
    print("first 5 rows")
    print(df.head())
    print("last 5 rows")
    print(df.tail())
    print("print 25 to 35 rows")
    print(df.iloc[25:36])
    print(" print coloumn names")
    print(df.columns)
    print("datatypes")
    print(df.dtypes)
    print("Complete Information")
    print(df.info())
    print("no of null values")
    missing=df.isnull().sum()
    print(missing[missing > 0])
    print("no of duplicate records")
    print(df.duplicated().sum())
    print("Target variable status")
    count=df["PlacementStatus"].value_counts()
    plt.figure(figsize = (6,5))
    plt.bar(count.index,count.values)
    plt.title("Placment prediction")

    plt.xlabel("Placement Status")
    plt.ylabel("Number of Records")
    plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\Placement Prediction")
    plt.show()


def univariant(df):
    plt.figure(figsize = (6,5))
    plt.hist(df["CGPA"],bins=10)
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\Histogram of CGPA")

    plt.show()

    gendercount=df["Gender"].value_counts()
    plt.figure(figsize = (6,5))
    plt.pie(gendercount,labels=gendercount.index,autopct="%1.1f%%",startangle=90)
    plt.title("Gender distribution piechart")
    plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\Gender Piechart")
    plt.show()


def bivariate(df):
    plt.figure(figsize = (6,5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"])
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\CGPA vs APTITUDE")
    plt.show()
    plt.close()

    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
    plt.boxplot([placed, not_placed], label=["placed", "not_placed"])
    plt.title("CGPA vs PlacementStatus")
    plt.xlabel("PlacementStatus")
    plt.ylabel("CGPA")
    plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\CGPA vs PLACEMENT")
    plt.show()

def multivariate(df):
        data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
        correlation = data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("correlation heatmap")
        plt.savefig(r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\app\static\charts\HEATMAP.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    df=load_data()
    basic_ede(df)
    univariant(df)
    bivariate(df)
    multivariate(df)