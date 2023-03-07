import os.path as osp
BASE_DIR = osp.dirname(osp.abspath(__file__))
BATCH_SIZE = 2
EPOCH_NUMBER = 100
DATASET = ['Shrimp_TOP', 7]

crop_size = (480, 1600)

class_dict_path = PATH + 'class_dict.csv'
TRAIN_ROOT =    "Configuring the path to the train set data."
TRAIN_LABEL =  "Configuring the path to the train label set data"
VAL_ROOT =      "Configuring the path to the validation set data"
VAL_LABEL =     "Configuring the path to the validation label set data"

TEST_ROOT = "Configuring the path to the test set data"
TEST_LABEL = "Configuring the path to the test label set data"

Weight_Path = "Set the path where you want to save the model file."