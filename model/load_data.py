import seaborn as sns
df = sns.load_dataset("iris")

if __name__ == "__main__":
    df.to_csv("model/iris.csv", index=False)
    print(df)