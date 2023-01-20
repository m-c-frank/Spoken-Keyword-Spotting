import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

from parameters import *
from utils import OC_Statistics
from utils import getDataset
from get_data import downloadData, getDataDict, getDataframe


def model_test(hotword="stop"):
    """
    Tests the SVM hotword detector
    :return: None
    """

    # Download data
    downloadData(data_path="/input/speech_commands/")

    # Get dictionary with files and labels
    dataDict = getDataDict(data_path="/input/speech_commands/")

    # Obtain dataframe by merging dev and test dataset
    devDF = getDataframe(dataDict["dev"], include_unknown=True)
    testDF = getDataframe(dataDict["test"], include_unknown=True)

    evalDF = pd.concat([devDF, testDF], ignore_index=True)

    print("Test files: {}".format(evalDF.shape[0]))

    # Obtain hotword - Other separated data
    evalDF["class"] = evalDF.apply(lambda row: 1 if row["category"] == hotword else -1, axis=1)
    evalDF.drop("category", axis=1)
    test_true_labels = evalDF["class"].tolist()

    eval_data, _ = getDataset(df=evalDF, batch_size=BATCH_SIZE, cache_file="kws_val_cache", shuffle=False)

    # Load trained model
    model = load_model(f"../models/{hotword}_kws.h5")

    layer_name = "features256"
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Load trained PCA object
    with open(f"../models/{hotword}_kws_pca.pickle", "rb") as file:
        pca = pickle.load(file)

    # Load trained SVM
    with open(f"../models/{hotword}_kws_svm.pickle", "rb") as file:
        hotword_svm = pickle.load(file)

    # Extract the feature embeddings and evaluate using SVM
    X_test = feature_extractor.predict(eval_data, use_multiprocessing=True)

    X_test_scaled = pca.transform(X_test)
    test_pred_labels = hotword_svm.predict(X_test_scaled)

    OC_Statistics(test_pred_labels, test_true_labels, f"{hotword}_cm_without_noise")
