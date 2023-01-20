import os
from model_train import model_train, train_kws_model
from model_test import model_test

HOTWORD = "stop"

def main():
    trained = os.path.isfile(f"../models/{HOTWORD}_kws_svm.pickle") and os.path.isfile(f"../models/{HOTWORD}_kws_pca.pickle")

    if not trained:
        print("Training model")
        model_train(HOTWORD)
        train_kws_model(HOTWORD)
    else:
        print("Testing model")
        model_test()


if __name__ == "__main__":
    main()
