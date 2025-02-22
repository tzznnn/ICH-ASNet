import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from utils.config import get_config
from utils.evaluation import eval_mask_slice
from models.model_dict import get_model
from utils.data_ihs import BCIHM, Transform2D_BCIHM, INSTANCE, Transform2D_INSTANCE
from utils.loss_functions.sam_loss import get_criterion
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.prompt_generator import PromptGenerator

def main():
    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-task', required=False, default='BCIHM')
    parser.add_argument('-fold', required=False, type=int, default=0)
    parser.add_argument('-encoder_input_size', type=int, default=1024)
    parser.add_argument('-low_image_size', type=int, default=256)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.0001)


    args = parser.parse_args()
    opt = get_config(args.task)
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "test"
    opt.visual = False
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    if args.task == 'BCIHM':
        tf_val = Transform2D_BCIHM(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                   ori_size=opt.img_size)
        val_dataset = BCIHM(opt.data_path, opt.test_split, tf_val, fold=args.fold, img_size=args.encoder_input_size,
                            class_id=1)
    elif args.task == 'INSTANCE':
        tf_val = Transform2D_INSTANCE(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                      ori_size=opt.img_size)
        val_dataset = INSTANCE(opt.data_path, opt.val_split, tf_val, fold=args.fold, img_size=args.encoder_input_size)
    else:
        assert ("We do not have the related dataset, please choose another task.")

    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model = get_model()

    prompt_generator = PromptGenerator("./pretrained/coarse_mask_generator.pth")

    prompt_generator.to(device)
    prompt_generator.eval()


    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    criterion = get_criterion(opt=opt)

    #  ========== begin to evaluate the model ==========
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total_params: {}".format(pytorch_total_params))

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = eval_mask_slice(valloader, model, prompt_generator, criterion, opt)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = eval_mask_slice(valloader, model, prompt_generator, criterion, opt)
        print("dataset:" + args.task + " -----------model name: " + args.modelname)
        print("task", args.task, "checkpoints:", opt.load_path)
        print(mean_dice, mean_hdis, mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        print(std_dice, std_hdis, std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])

if __name__ == '__main__':
    main()
