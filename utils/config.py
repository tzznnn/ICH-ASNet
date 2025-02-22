# This file is used to configure the training or testing parameters for each task
class Config_BCIHM:
    data_path = "./dataset/BCIHM/"
    save_path = "./checkpoints/BCIHM/"
    load_path = ''
    save_path_code = "_"

    workers = 4                         # data loading workers (default: 8)
    epochs = 100                        # total training epochs (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-5                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 512                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    pre_trained = False
    mode = "test"
    visual = False

class Config_INSTANCE:
    data_path = "./dataset/INSTANCE/"
    save_path = "./checkpoints/INSTANCE/"
    load_path = './checkpoints/INSTANCE/autoSAM_01110311_85_0.7319.pth'
    save_path_code  = "_"

    workers = 2                         # data loading workers (default: 8)
    epochs = 100                        # total epochs to run (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 512                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    pre_trained = False
    mode = "test"
    visual = False

# ==================================================================================================
def get_config(task="BCIHM"):
    if task == "BCIHM":
        return Config_BCIHM()
    elif task == "INSTANCE":
        return Config_INSTANCE()
    else:
        assert("We do not have the related dataset, please choose another task.")
