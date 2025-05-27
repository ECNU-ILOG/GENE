import random
import time
import numpy as np
import torch
from torch import nn
from scipy.stats import chi2_contingency
import copy

# torch.manual_seed(40)
# np.random.seed(40)

def GENE(train_data, test_data):
    node_num = train_data.shape[1]
    hidden_size = 20
    alpha = 1
    eval_dict = {}
    eval_dict_iden = {}
    model_dict = {}
    for i in range(1, node_num):
        model = nn.Sequential(
            nn.Linear(i, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
        model_dict[i] =model

    def chisquare_test(X, Y, bin_num=10):
        c_XY = np.histogram2d(X, Y, bins=[bin_num, bin_num])[0] + 1e-8
        chi2, p, dof, ex = chi2_contingency(c_XY)
        return p

    def mlp_regress(train_data, test_data, columns, label):
        input_size = len(columns)
        num_epochs = 500
        learning_rate = 0.1
        input = train_data[:, columns]
        input_test = test_data[:, columns]
        X_train = torch.from_numpy(input).reshape(-1, input_size).to(torch.float32)
        X_test = torch.from_numpy(input_test).reshape(-1, input_size).to(torch.float32)
        Y_train = torch.from_numpy(train_data[:, [label]]).reshape(-1, 1).to(torch.float32)
        Y_test = torch.from_numpy(test_data[:, [label]]).reshape(-1, 1).to(torch.float32)
        model = copy.deepcopy(model_dict[input_size])
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        for _ in range(num_epochs):
            y_pred = model.forward(X_train)
            loss = loss_func(y_pred, Y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        r2_loss = 1 - loss_func(model.forward(X_test).data, Y_test) / np.var(np.array(Y_test.data))
        residual = np.array(model.forward(X_test).data).reshape(1, -1)[0] - np.array(Y_test.data).reshape(1, -1)[0]
        return r2_loss, residual

    def fitness(order, train_data, test_data):
        n = len(order)
        fitness = []
        fitness.append(0)
        for i in range(1, n):
            node = order[i]
            columns = order[:i]
            if str(node) + str(set(columns)) in eval_dict.keys():
                metric = eval_dict[str(node) + str(set(columns))]
            else:
                r2_loss, residual = mlp_regress(train_data=train_data, test_data=test_data,
                                                columns=columns, label=node)
                metric = r2_loss
                eval_dict[str(node) + str(set(columns))] = metric
            if metric >= 0:
                fitness.append(metric)
            else:
                fitness.append(0)
        return fitness

    def fitness_iden_origin(order, train_data, test_data, significance_level=0.05):
        n = len(order)
        fitness = []
        fitness.append(0)
        for i in range(1, n):
            node = order[i]
            columns = order[:i]
            if str(node) + str(set(columns)) in eval_dict_iden.keys():
                metric = eval_dict_iden[str(node) + str(set(columns))]
            else:
                r2_loss, residual = mlp_regress(train_data=train_data, test_data=test_data, columns=columns, label=node)
                metric = r2_loss
                for c in columns:
                    shuru = test_data[:, c].reshape(1, -1)[0]
                    p_value = chisquare_test(shuru, residual, bin_num=10)
                    if p_value < significance_level:
                        metric = metric - (alpha * r2_loss / i)
                eval_dict_iden[str(node) + str(set(columns))] = metric
            fitness.append(metric)
        return fitness


    def move(order, point1, point2):
        new_order = order.copy()
        if point1 < point2:
            temp = order[point1+1:point2+1]
            new_order[point1: point2] = temp
            new_order[point2] = order[point1]
        else:
            temp = order[point2: point1]
            new_order[point2+1: point1+1] = temp
            new_order[point2] = order[point1]
        return new_order

    def cal_fit_val(order_fitness):
        return np.sum(order_fitness)

    def order_search(train_data, test_data):
        n = train_data.shape[1]
        best_order = np.random.permutation(n)
        order_fit = fitness_iden_origin(best_order, train_data, test_data)
        best_value = cal_fit_val(order_fit)
        contin = True
        while(contin):
            contin = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    new_order = move(best_order, i, j)
                    new_fit = fitness_iden_origin(new_order, train_data, test_data)
                    if cal_fit_val(new_fit) > best_value:
                        best_value = cal_fit_val(new_fit)
                        best_order = new_order.copy()
                        contin = True
        return best_order
    def parent_search(order, train_data, test_data, threshold=0.02):
        n = len(order)
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            parent_set = []
            node = order[i]
            fit = 0
            while(True):
                candidate = {}
                for j in range(i):
                    if order[j] not in parent_set:
                        input_size = 1 + len(parent_set)
                        columns = parent_set.copy()
                        columns.append(order[j])
                        if str(node) + str(set(columns)) in eval_dict.keys():
                            r2_loss = eval_dict[str(node) + str(set(columns))]
                        else:
                            r2_loss, residual = mlp_regress(train_data=train_data, test_data=test_data,
                                                            columns=columns, label=node)
                            eval_dict[str(node) + str(set(columns))] = r2_loss
                        candidate[order[j]] = r2_loss
                if len(candidate) > 0:
                    thekey = max(candidate, key=candidate.get)
                else:
                    break
                if candidate[thekey] - fit >= threshold:
                    parent_set.append(thekey)
                    fit = candidate[thekey]
                else:
                    break
            for pa in parent_set:
                adj_matrix[pa][node] = 1

        return adj_matrix



    def parent_search_pruning(order, train_data, test_data, threshold=0.1, decay=1.1):
        n = len(order)
        adj_matrix = np.zeros((n, n), dtype=int)
        est_edge = 0
        for i in range(1, n):
            parent_set = list(order[:i])
            node = order[i]
            if str(node) + str(set(parent_set)) in eval_dict.keys():
                fit = eval_dict[str(node) + str(set(parent_set))]
            else:
                fit, residual = mlp_regress(train_data=train_data, test_data=test_data,
                                            columns=parent_set, label=node)
                eval_dict[str(node) + str(set(parent_set))] = fit
            while (True):
                if fit <= 0.1:
                    parent_set = []
                    break
                tobeprune = {}
                for j in range(i):
                    if order[j] in parent_set:
                        input_size = len(parent_set) - 1
                        if input_size > 0:
                            columns = parent_set.copy()
                            columns.remove(order[j])
                            if str(node) + str(set(columns)) in eval_dict.keys():
                                r2_loss = eval_dict[str(node) + str(set(columns))]
                            else:
                                r2_loss, residual = mlp_regress(train_data=train_data, test_data=test_data,
                                                                columns=columns, label=node)
                                eval_dict[str(node) + str(set(columns))] = r2_loss
                            tobeprune[order[j]] = r2_loss
                        else:
                            tobeprune[order[j]] = 0
                if len(tobeprune) > 0:
                    thekey = max(tobeprune, key=tobeprune.get)
                    minkey = min(tobeprune, key=tobeprune.get)
                else:
                    break
                if (fit - tobeprune[thekey]) < fit / min(6, len(parent_set)):
                    parent_set.remove(thekey)
                    fit = tobeprune[thekey]
                    threshold = threshold * decay
                else:
                    break
            for pa in parent_set:
                adj_matrix[pa][node] = 1
                est_edge += 1

        return adj_matrix

    t = time.time()
    best_order = order_search(train_data=train_data, test_data=test_data)
    est_matr = parent_search(best_order, train_data=train_data, test_data=test_data)
    # est_matr = parent_search_pruning(best_order, train_data=train_data, test_data=test_data)
    ex_time = time.time() - t
    return best_order, est_matr, ex_time


