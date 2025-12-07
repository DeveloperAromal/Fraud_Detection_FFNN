from model.train import train_model
from utils.load_dataset import load_data
from nn.ffnn import FeedForwardNN
from model.predict import predict
from utils.dataset_preprocessor import preprocess_csv


import uvicorn


def main():
    # X_train, X_test, y_train, y_test = load_data()

    # model = FeedForwardNN()
    # train_model(model, X_train, y_train, X_test, y_test)

    # if __name__ == "__main__":
    #     uvicorn.run(
    #                     "api.server:app",
    #                     host="127.0.0.1",
    #                     port=8000,
    #                     reload=True
    #             )
    
    preprocess_csv("data/raw/dataset.csv", "data/processed/new_dataset.csv")
main()