import numpy as np
import pulp
from torch import Tensor
from torchvision.ops.giou_loss import generalized_box_iou_loss


def oc_cost(true_boxes, pred_boxes, true_labels, pred_labels, scores, lam=0.5, beta=0.6):
    costs = get_costs(true_boxes, pred_boxes, true_labels, pred_labels, scores, lam)
    n, m = costs.shape
    pi, costs = optimize_transportation_problem(costs, beta)
    pi[n, m] = 0
    total = np.sum(pi)
    pi_tilda = pi / total
    return np.sum(costs * pi_tilda)


def calc_cloc(true_boxes, pred_boxes):
    giou = generalized_box_iou_loss(pred_boxes, true_boxes)
    if isinstance(giou, Tensor):
        giou = giou.cpu().numpy()
    return (1 - giou) / 2


def calc_ccls(true_labels, pred_labels, scores):
    true_labels = true_labels.reshape(1, -1)
    pred_labels = pred_labels.reshape(-1, 1)
    scores = scores.reshape(-1, 1)
    res = pred_labels == true_labels
    costs = (1 - scores) / 2 * res + (1 + scores) / 2 * (~res)
    if isinstance(costs, Tensor):
        costs = costs.cpu().numpy()
    return costs


def get_costs(true_boxes, pred_boxes, true_labels, pred_labels, scores, lam):
    return lam * calc_cloc(true_boxes, pred_boxes) + (1 - lam) * calc_ccls(true_labels, pred_labels, scores)


def optimize_transportation_problem(costs, beta):
    pi = {}
    n, m = costs.shape
    costs = np.insert(costs, n, beta, axis=0)
    costs = np.insert(costs, m, beta, axis=1)
    d = [1] * m + [n]
    s = [1] * n + [m]
    for i in range(n + 1):
        for j in range(m + 1):
            pi[i, j] = pulp.LpVariable(f"pi({i}, {j})", lowBound=0)
    transportation_problem = pulp.LpProblem('TP', pulp.LpMinimize)
    objective = pulp.lpSum(costs[i, j] * pi[i, j] for i in range(n + 1) for j in range(m + 1))
    transportation_problem += objective
    for j in range(m + 1):
        transportation_problem += pulp.lpSum(pi[i, j] for i in range(n + 1)) == d[j]
    for i in range(n + 1):
        transportation_problem += pulp.lpSum(pi[i, j] for j in range(m + 1)) == s[i]
    solver = pulp.PULP_CBC_CMD(msg=False)

    transportation_problem.solve(solver)
    pi_val = np.zeros((costs.shape))
    for i in range(n + 1):
        for j in range(m + 1):
            pi_val[i, j] = pi[i, j].value()
    return pi_val, costs
