from model.train import train_model 
from nn.ffnn import FeedForwardNN
from utils.get_data_loader import load_data



def pipeline():
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = FeedForwardNN()
    train_model(model, X_train, y_train, X_val, y_val, X_test, y_test)