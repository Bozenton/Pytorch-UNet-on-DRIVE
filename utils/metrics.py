import torch


def BinaryConfusionMatrix(pred, gt):
    """Computes scores:
    TP = True Positives    真正例
    FP = False Positives   假正例
    FN = False Negatives   假负例
    TN = True Negatives    真负例
    return: TP, FP, FN, TN"""
    assert torch.max(gt) <= 1, "The input should be binary image"
    assert torch.max(pred) <= 1, "The input should be binary image"

    gt = gt.bool()
    pred = pred.bool()

    TP = torch.sum(gt & pred)
    FN = torch.sum(gt & ~pred)
    FP = torch.sum(~gt & pred)
    TN = torch.sum(~gt & ~pred)

    return TN, FP, FN, TP


# 精准率和 或 查准率的计算方法
def get_precision(prediction, groundtruth):
    _, FP, _, TP = BinaryConfusionMatrix(prediction, groundtruth)
    precision = float(TP) / (float(TP + FP) + 1e-6)
    return precision


# 召回率和 或 查全率的计算方法
def get_recall(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    recall = float(TP) / (float(TP + FN) + 1e-6)
    return recall


# 准确率的计算方法
def get_accuracy(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    accuracy = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
    return accuracy


def get_sensitivity(prediction, groundtruth):
    return get_recall(prediction, groundtruth)


def get_specificity(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    specificity = float(TN) / (float(TN + FP) + 1e-6)
    return specificity


def get_f1_score(prediction, groundtruth):
    precision = get_precision(prediction, groundtruth)
    recall = get_recall(prediction, groundtruth)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


# Dice相似度系数，计算两个样本的相似度，取值范围为[0, 1], 分割结果最好为1，最坏为0
def get_dice(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    dice = 2 * float(TP) / (float(FP + 2 * TP + FN) + 1e-6)
    return dice


# 交并比 一般都是基于类进行计算, 值为1这一类的iou
def get_iou1(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TP) / (float(FP + TP + FN) + 1e-6)
    return iou


# 交并比 一般都是基于类进行计算, 值为0这一类的iou
def get_iou0(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TN) / (float(FP + TN + FN) + 1e-6)
    return iou


# 基于类进行计算的IoU就是将每一类的IoU计算之后累加，再进行平均，得到的就是基于全局的评价
# 平均交并比
def get_mean_iou(prediction, groundtruth):
    iou0 = get_iou0(prediction, groundtruth)
    iou1 = get_iou1(prediction, groundtruth)
    mean_iou = (iou1 + iou0) / 2
    return mean_iou


if __name__ == '__main__':
    pass
