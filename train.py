import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import random
from utils.config import get_config
from utils.evaluation import eval_mask_slice

from models.model_dict import get_model
from utils.data_ihs import BCIHM, Transform2D_BCIHM, INSTANCE, Transform2D_INSTANCE
from utils.loss_functions.sam_loss import get_criterion    

from tqdm import tqdm
from utils.prompt_generator import PromptGenerator
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def main():

    # ========== parameters setting ==========

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-task', required=False, default='BCIHM')
    parser.add_argument('-fold', required=False, type=int, default=0)
    parser.add_argument('--encoder_input_size', type=int, default=1024)
    parser.add_argument('--low_image_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--warmup_period', type=int, default=250)

    args = parser.parse_args()
    opt = get_config(args.task)
    opt.mode = 'train'

    device = torch.device(opt.device)

    #  ========== add the seed to make sure the results are reproducible ==========
    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ========== model and data preparation ==========
    model = get_model()
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    for param in model.image_encoder.crossAttentionModule.parameters():
        param.requires_grad = True
    for param in model.image_encoder.adapter_down.parameters():
        param.requires_grad = True
    for param in model.image_encoder.adapter_up.parameters():
        param.requires_grad = True
    for param in model.image_encoder.layer_norm1.parameters():
        param.requires_grad = True
    prompt_generator = PromptGenerator("./pretrained/coarse_mask_generator.pth")
    prompt_generator.to(device)
    prompt_generator.eval()
    if args.task == 'BCIHM':
        tf_train = Transform2D_BCIHM(mode=opt.mode, img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size)
        tf_val = Transform2D_BCIHM(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size)
        train_dataset = BCIHM(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
        val_dataset = BCIHM(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    elif args.task == 'INSTANCE':
        tf_train = Transform2D_INSTANCE(mode=opt.mode, img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size)
        tf_val = Transform2D_INSTANCE(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size)
        train_dataset = INSTANCE(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
        val_dataset = INSTANCE(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    else:
        assert("We do not have the related dataset, please choose another task.")
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    b_lr = args.base_lr / args.warmup_period
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)

    criterion = get_criterion(opt=opt)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========== begin to train the model ==========
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):

        # ---------- Train ----------
        model.train()
        optimizer.zero_grad()
        train_losses = 0
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for batch_idx, (datapack) in enumerate(trainloader):
                imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
                masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
                with torch.no_grad():
                    pt = prompt_generator.generator_points(imgs)
                    text_feat = prompt_generator.generator_text(imgs)
                pred = model(imgs, pt=pt, bbox=None, text=text_feat)
                train_loss = criterion(pred, masks)
                # ---------- backward ----------
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_postfix(**{'loss (batch)': train_loss.item()})
                train_losses += train_loss.item()
                # ---------- Adjust learning rate ----------
                if args.warmup and iter_num < args.warmup_period:
                    lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    if args.warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                        lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_
                iter_num = iter_num + 1

                pbar.update()
        # ---------- Write log ----------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        print('lr: ', optimizer.param_groups[0]['lr'])

        # ---------- Validation ----------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = eval_mask_slice(valloader, model, prompt_generator, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + "ICH-ASNet" + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(round(best_dice, 4))
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)

            save_path = opt.save_path + "ICH-ASNet" + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
if __name__ == '__main__':
    main()
