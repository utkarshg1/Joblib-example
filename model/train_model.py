import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_dataframe():
    df = pd.read_csv("model/iris.csv")
    return df 

def split_data(df, test_size=0.33):
    X = df.drop(columns=["species"])
    Y = df["species"]
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )
    return xtrain, xtest, ytrain, ytest

def get_pipeline():
    pipe = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression()
    )
    return pipe 

def save_model(pipe, xtrain, ytrain, xtest, ytest):
    pipe.fit(xtrain, ytrain)
    train_score = pipe.score(xtrain, ytrain)
    test_score = pipe.score(xtest, ytest)
    cv_scores = cross_val_score(pipe, xtrain, ytrain, cv=5, scoring="f1_macro")
    print(f"Train Score : {train_score:.4f}")
    print(f"Test Score : {test_score:.4f}")
    print(f"5 fold cross validated on Train F1 Macro score : {cv_scores.mean():.4f}")
    ypred_test = pipe.predict(xtest)
    print("\nClassification Report for test data :\n")
    print(classification_report(ytest, ypred_test))
    path = "model/iris_model.joblib"
    joblib.dump(pipe, path)
    return path

def load_model(path="model/iris_model.joblib"):
    model = joblib.load(path)
    return model

def predict_results(pipe, sep_len, sep_wid, pet_len, pet_wid):
    xnew = pd.DataFrame(
        {
            "sepal_length" : [sep_len],
            "sepal_width" : [sep_wid],
            "petal_length" : [pet_len],
            "petal_width" : [pet_wid] 
        }
    )
    preds = pipe.predict(xnew)[0]
    probs = pipe.predict_proba(xnew)[0]
    classes = pipe.classes_
    probs_dict = {
        classes[i]:round(float(prob), 4) for i, prob in enumerate(probs)
    } 
    return preds, probs_dict

if __name__ == "__main__":
    # Load data and fit model
    df = load_dataframe()
    xtrain, xtest, ytrain, ytest = split_data(df)
    pipe = get_pipeline()
    path = save_model(pipe, xtrain, ytrain, xtest, ytest)
    model = load_model()

    # Take inputs from user
    sep_len = float(input("Sepal Length in cm : "))
    sep_wid = float(input("Sepal Width in cm : "))
    pet_len = float(input("Petal Length in cm : "))
    pet_wid = float(input("Petal Width in cm : "))
    preds, probs = predict_results(pipe, sep_len, sep_wid, pet_len, pet_wid)
    print(f"Prediction is : {preds}")
    print(f"Probabilities : {probs}")

