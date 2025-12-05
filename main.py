from model.train import train_model
from utils.load_dataset import load_data
from nn.ffnn import FeedForwardNN

from model.predict import predict



def main():
    # X_train, X_test, y_train, y_test = load_data()

    # model = FeedForwardNN()
    # train_model(model, X_train, y_train, X_test, y_test)

    predict()
    
main()