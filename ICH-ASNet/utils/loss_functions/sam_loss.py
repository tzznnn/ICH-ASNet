from pyexpat import model
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F

class Match_loss(nn.Module):
    def __init__(self, lambda_b):
        super(Match_loss, self).__init__()
        self.lambda_b = lambda_b

    def match_loss(self, v_f, v_b, u_f, lambda_b):
        """
        计算匹配损失（L_match）

        参数：
        - v_f: 前景特征向量 (batch_size, feature_dim)
        - v_b: 背景特征向量 (batch_size, feature_dim)
        - u_f: 目标特征向量 (batch_size, feature_dim)
        - lambda_b: 平衡系数 (标量)

        返回：
        - 损失值 (标量)
        """
        # 计算前景与目标特征之间的相似度（余弦相似度）
        sim_f = F.cosine_similarity(v_f, u_f, dim=-1)  # (batch_size,)
        sim_f[sim_f < 0] = 0.0001
        # sim_f = (sim_f + 1) / 2
        # 计算背景与目标特征之间的相似度（余弦相似度）
        sim_b = F.cosine_similarity(v_b, u_f, dim=-1)  # (batch_size,)
        sim_b[sim_b < 0] = 0.0001
        # print("sim_f"+str(sim_f))
        # print("sim_b"+str(sim_b))
        # sim_b = (sim_b + 1) / 2
        # 计算第一个损失项：log(sim(v_f, u_f))
        loss_f = torch.log(sim_f)
        # 计算第二个损失项：λ_b * log(1 - sim(v_b, u_f))
        loss_b = lambda_b * torch.log(1 - sim_b)
        # 总损失
        loss = torch.mean(-(loss_f + loss_b))  # 对 batch 求平均
        return loss

    def forward(self, output):
        # 计算损失
        v_f = output["v_f"]
        v_b = output["v_b"]
        u_f = output["u_f"]
        loss = self.match_loss(v_f, v_b, u_f, self.lambda_b)
        return loss

class Prompt_loss(nn.Module):
    def __init__(self, lambda_t):
        super(Prompt_loss, self).__init__()
        self.lambda_t = lambda_t

    def prompt_loss(self, v_b, u_f, u_b, lambda_t):
        """
        计算匹配损失（L_match）

        参数：
        - v_f: 前景特征向量 (batch_size, feature_dim)
        - v_b: 背景特征向量 (batch_size, feature_dim)
        - u_f: 目标特征向量 (batch_size, feature_dim)
        - lambda_b: 平衡系数 (标量)

        返回：
        - 损失值 (标量)
        """
        # 计算背景和可训练提示之间的相似度（余弦相似度）
        sim_f = F.cosine_similarity(u_b, v_b, dim=-1)  # (batch_size,)
        sim_f[sim_f < 0] = 0.0001
        # sim_f = (sim_f + 1) / 2
        # 计算前景文本和可训练提示的相似度（余弦相似度）
        sim_b = F.cosine_similarity(u_b, u_f, dim=-1)  # (batch_size,)
        sim_b[sim_b < 0] = 0.0001
        # print(sim_f)
        # print(sim_b)
        # sim_b = (sim_b + 1) / 2
        # 计算第一个损失项：log(sim(v_f, u_f))
        loss_f = torch.log(sim_f)
        # 计算第二个损失项：λ_b * log(1 - sim(v_b, u_f))
        loss_b = lambda_t * torch.log(sim_b)
        # 总损失
        loss = torch.mean(-loss_f + loss_b)  # 对 batch 求平均
        return loss

    def forward(self, output):
        # 计算损失
        v_b = output["v_b"]
        u_f = output["u_f"]
        u_b = output["u_b"]
        loss = self.prompt_loss(v_b, u_f, u_b, self.lambda_t)
        return loss

class Total_loss(nn.Module):
    def __init__(self, lambda_, lambda_b):
        super(Total_loss, self).__init__()
        self.lambda_ = lambda_
        self.lambda_b = lambda_b

    def match_loss(self, v_f, v_b, u_f, lambda_b):
        """
        计算匹配损失（L_match）

        参数：
        - v_f: 前景特征向量 (batch_size, feature_dim)
        - v_b: 背景特征向量 (batch_size, feature_dim)
        - u_f: 目标特征向量 (batch_size, feature_dim)
        - lambda_b: 平衡系数 (标量)

        返回：
        - 损失值 (标量)
        """
        # 计算前景与目标特征之间的相似度（余弦相似度）
        sim_f = F.cosine_similarity(v_f, u_f, dim=-1)  # (batch_size,)
        sim_f[sim_f < 0] = 0.0001
        # sim_f = (sim_f + 1) / 2
        # 计算背景与目标特征之间的相似度（余弦相似度）
        sim_b = F.cosine_similarity(v_b, u_f, dim=-1)  # (batch_size,)
        sim_b[sim_b < 0] = 0.0001
        # sim_b = (sim_b + 1) / 2
        # 计算第一个损失项：log(sim(v_f, u_f))
        loss_f = torch.log(sim_f)
        # 计算第二个损失项：λ_b * log(1 - sim(v_b, u_f))
        loss_b = lambda_b * torch.log(1 - sim_b)
        # 总损失
        loss = torch.mean(-(loss_f + loss_b))  # 对 batch 求平均
        return loss

    def refine_loss(self, v_f, u_b):
        """
        计算匹配损失（L_match）

        参数：
        - v_f: 前景特征向量 (batch_size, feature_dim)
        - v_b: 背景特征向量 (batch_size, feature_dim)
        - u_f: 目标特征向量 (batch_size, feature_dim)
        - lambda_b: 平衡系数 (标量)

        返回：
        - 损失值 (标量)
        """
        # 计算背景和可训练提示之间的相似度（余弦相似度）
        sim_f = F.cosine_similarity(v_f, u_b, dim=-1)  # (batch_size,)
        sim_f[sim_f < 0] = 0.0001
        # 计算第一个损失项：log(sim(v_f, u_f))
        loss_f = torch.log(1 - sim_f)
        # 总损失
        loss = torch.mean(-loss_f)  # 对 batch 求平均
        return loss

    def forward(self, output):
        # 计算损失
        v_f = output["v_f"]
        v_b = output["v_b"]
        u_f = output["u_f"]
        u_b = output["u_b"]
        match_loss = self.match_loss(v_f, v_b, u_f, self.lambda_b)
        refine_loss = self.refine_loss(v_f, u_b)
        loss = match_loss + self.lambda_ * refine_loss
        return loss

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DC_and_BCE_loss(nn.Module):
    def __init__(self, classes=2, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.ce = CrossEntropyLoss()
        self.dc = DiceLoss(classes)
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        # low_res_logits = net_output
        if len(target.shape) == 4:
            target = target[:, 0, :, :]
        loss_ce = self.ce(low_res_logits, target[:].long())
        loss_dice = self.dc(low_res_logits, target, softmax=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        # loss = loss_ce + loss_dice
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=True):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.5):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss
    
class Mask_DC_and_BCE_loss_typical(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.5):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss_typical, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=False)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss


class Mask_BCE_loss(nn.Module):
    def __init__(self, pos_weight):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BCE_loss, self).__init__()

        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, net_output, target):
        # low_res_logits = net_output
        low_res_logits = net_output['low_res_logits']
        loss = self.ce(low_res_logits, target)
        # loss = self.ce(net_output, target)
        return loss


class Match_and_Mask_BD_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, bd_weight, lambda_b, lambda_t):
        super(Match_and_Mask_BD_and_BCE_loss, self).__init__()
        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lambda_b = lambda_b
        self.lambda_t = lambda_t
        self.bd = BoundarySensitiveLoss()
        self.bd_weight = bd_weight
    def match_loss(self, v_f, v_b, u_f, lambda_b):
        # 计算前景与目标特征之间的相似度（余弦相似度）
        sim_f = F.cosine_similarity(v_f, u_f, dim=-1)  # (batch_size,)
        sim_f = torch.clamp(sim_f, min=1e-4, max=1)  # 确保值在 [1e-6, 1] 范围内
        # 计算背景与目标特征之间的相似度（余弦相似度）
        sim_b = F.cosine_similarity(v_b, u_f, dim=-1)  # (batch_size,)
        sim_b = torch.clamp(sim_b, min=1e-4, max=1 - 1e-4)  # 确保值在 [1e-6, 1) 范围内

        # 计算第一个损失项：log(sim(v_f, u_f))
        loss_f = torch.log(sim_f)

        # 计算第二个损失项：λ_b * log(1 - sim(v_b, u_f))
        loss_b = lambda_b * torch.log(1 - sim_b)

        # 总损失
        mat_loss = torch.mean(-(loss_f + loss_b))  # 对 batch 求平均
        return mat_loss
    def forward(self, output, target):
        v_f = output["train"]["v_f"]
        v_b = output["train"]["v_b"]
        u_f = output["train"]["u_f"]

        lambda_b = torch.tensor(self.lambda_b, device=v_f.device)
        lambda_t = torch.tensor(self.lambda_t, device=v_f.device)
        bd_weight = torch.tensor(self.bd_weight, device=v_f.device)

        low_res_logits = output['low_res_logits']
        mat_loss = self.match_loss(v_f, v_b, u_f, lambda_b)
        bd_loss = self.bd(low_res_logits, target, sigmoid=True)
        bec_loss = self.ce(low_res_logits, target)
        pic_loss = (1 - bd_weight) * bec_loss + bd_weight * bd_loss
        loss = pic_loss + lambda_t * mat_loss
        return loss
class BoundarySensitiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0],[1,1,1],[0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0 
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss
    
    def forward(self, inputs, target, weight=None, sigmoid=False):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        BD_loss = self._adaptive_size(inputs[:, 0], target[:, 0])
        return BD_loss
    
class Mask_BD_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, bd_weight=0.5):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BD_and_BCE_loss, self).__init__()

        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bd = BoundarySensitiveLoss()
        self.bd_weight = bd_weight
    def forward(self, net_output, target):
        # low_res_logits = net_output['low_res_logits']
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.bd(low_res_logits, target, sigmoid=True)
        loss = (1 - self.bd_weight) * loss_ce + self.bd_weight * loss_dice
        return loss

def get_criterion(opt=None):
    device = torch.device(opt.device)
    pos_weight = torch.ones([1]).cuda(device=device)*2
    criterion = Mask_BD_and_BCE_loss(pos_weight=pos_weight)
    return criterion
