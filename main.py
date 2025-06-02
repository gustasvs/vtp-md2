from data_fetch import fetch_data
from data_preprocess import preprocess_data
from model_naive_bayes import train
from model_roberta import train_roberta

if __name__ == "__main__":

    # Datu kopas sagatavošana, analīze, priekšapstrāde
    fetch_data()

    # Datu priekšapstrāde
    preprocess_data()

    train()

    train_roberta()
