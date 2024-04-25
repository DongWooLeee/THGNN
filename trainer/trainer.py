import torch
import torch.nn as nn

def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels, mask, loss_func=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos, adj_neg)

    loss = loss_func(logits,labels)
    return loss, logits


def extract_data(data_dict, device):
    '''
    data_dict: {'pos_adj': pos_adj, 'neg_adj': neg_adj, 'features': features, 'labels': labels, 'mask': mask}
    '''
    pos_adj = data_dict['pos_adj'].to(device).squeeze() #  따로 차원을 설정하지 않으면 1인 차원을 모두 제거, 주의할 점은 생각치도 못하게 batch가 1일 때 batch차원도 없애버리는 불상사가 발생할 수있다. 
    neg_adj = data_dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask'] # 문제는 Semi-Supervised로서 정의되었음. 즉, label이 없는 것들에 대해서는 mask를 통해서 구분해줘야함
    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx == 0:
                loss_return += loss.data
    return loss_return/len(dataset_train)


def eval_epoch(args, model, dataset_eval, loss_fcn):
    loss = 0.
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
        loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        break
    return loss, logits