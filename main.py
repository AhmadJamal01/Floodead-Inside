import pickle
import argparse
import pandas as pd
from utils import load_model, prepare_dataX

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", help="Model pkl file path you wish to use", type=str)
    parser.add_argument(
        "--test_path", help="Data file path you wish to use for testing", type=str)

    args = parser.parse_args()

    # load the model
    model_path = args.model_path
    model = load_model(model_path)

    # load the data file
    test_path = args.test_path
    testdata = pd.read_csv(test_path)

    # Preprocess the data
    testdata = prepare_dataX(testdata)

    # predict and save them to preds.txt
    predictions = model.predict(testdata)
    Body_Level_str_to_int = {
    "Body Level 1": 0,
    "Body Level 2": 1,
    "Body Level 3": 2,
    "Body Level 4": 3,
    }
    # int to str dict
    Body_Level_int_to_str = {
    0: "Body Level 1",
    1: "Body Level 2",
    2: "Body Level 3",
    3: "Body Level 4"
    }

    # print(predictions)
    # for i in range(len(predictions)):
    #     print(predictions[i])
    with open("preds.txt", "w") as fp:
        fp.writelines('%s\n' % Body_Level_int_to_str[class_out] for class_out in predictions)
        # fp.writelines('%s\n' % '' for class_out in predictions)

