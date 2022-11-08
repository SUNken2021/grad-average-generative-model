import matplotlib.pyplot as plt
import numpy as np
import torch
import time


def score_matching_estimator(scorenet, samples, alpha, sigmas, hook=None):
    labels = torch.ones((samples.shape[0],))
    labels = labels.long()
    used_sigmas = sigmas[labels].view(
        samples.shape[0],
        *(
                [1] * len(samples.shape[1:])
        )
    )
    
    # 计算scores以及其平方的二分之一scores_sq
    scores = scorenet(samples, labels)
    scores_sq = 1 / 2 * (scores.view(scores.shape[0], -1) ** 2)
    
    # 用最小二乘法估计导数
    grad_trace = esitimate_grad_trace(samples, scores, alpha)
    # print("score_sq:{}\ngrad_trace:{}".format(scores_sq.sum(dim=-1), grad_trace.sum(dim=-1)))

    loss = torch.abs(scores_sq.sum(dim=-1) + grad_trace.sum(dim=-1))

    # print("loss:{}".format(loss))

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)

# 根据alpha来进行分划
def esitimate_grad_trace(samples, scores, alpha):
    samples = samples.view(samples.shape[0], -1)
    scores = scores.view(samples.shape[0], -1)

    grad_trace = torch.zeros_like(samples).float()

    if alpha == 0:
        return grad_trace
    
    n_dot, n_dimension = samples.shape
    per_dot = np.int(2 ** alpha)
    
    # 对数据进行排序,并记录下排序前的序号,根据排序的顺序进行分块
    x, index = torch.sort(samples, dim=0)
    fx = torch.zeros_like(scores).float()
    
    for i in range(n_dot):
        for j in range(n_dimension):
            fx[i, j] = scores[index[i, j], j]

    grad_trace_order = torch.zeros_like(samples).float()
    for i in range(n_dot // per_dot):
        grad_trace_order[i * per_dot:(i + 1) * per_dot] = \
            LSM(x[i * per_dot:(i + 1) * per_dot], \
                fx[i * per_dot:(i + 1) * per_dot])

    # 根据编号还原顺序
    for i in range(n_dot):
        for j in range(n_dimension):
            grad_trace[index[i, j], j] = grad_trace_order[i, j]

    return grad_trace

# 最小二乘法
def LSM(x, fx):
    n = x.shape[0]
    denominator = n * (x * fx).sum(dim=0) - x.sum(dim=0) * fx.sum(dim=0)
    numerator = n * (x * x).sum(dim=0) - (x.sum(dim=0)) ** 2

    k = denominator / numerator
    for i in range(len(numerator)):
        if numerator[i] == 0:
            k[i] = 0

    # print("k:{}".format(k))
    return k * (torch.ones_like(x).float())


def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(
            0,
            len(sigmas),
            (samples.shape[0],),  # 图片数*1的二维tensor
            device=samples.device
        )  # 0到图片数的随机整数
    used_sigmas = sigmas[labels].view(
        samples.shape[0],
        *(
                [1] * len(samples.shape[1:])
        )
    )  # 图片数*1*1*1的tensor
    noise = torch.randn_like(samples) * used_sigmas  # 0~1随机数*0~图片数随机整数，每张图片加上统一的噪声
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise

    scores = scorenet(perturbed_samples, labels)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
