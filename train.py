from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train RandomForest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model as model.pkl
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()
