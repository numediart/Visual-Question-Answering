import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, lr, early_stop, logger):

    optim = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    best_eval_score = 0
    es = 0

    for epoch in range(num_epochs):
        model_loss = 0
        train_score = 0

        t = time.time()

        for i, (v, q, a) in enumerate(train_loader):
            v = Variable(v).cuda().to(torch.float32)
            q = Variable(q).cuda()
            a = Variable(a).cuda()


            logits = model(v, q, a)
            loss = instance_bce_with_logits(logits, a)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(logits, a.data).sum()
            train_score += batch_score.data
            model_loss += loss.data * v.size(0)

            model_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (model_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        es += 1
        if eval_score > best_eval_score:
            old = os.path.join(output, 'model_%.4f.pth' % best_eval_score)
            if os.path.exists(old):
                os.remove(old)
            model_path = os.path.join(output, 'model_%.4f.pth' % eval_score)
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            es = 0
        if es == early_stop:
            print("Early stop reached")
            sys.exit()


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, q, a in iter(dataloader):
        with torch.no_grad():
            v = Variable(v).cuda().to(torch.float32)
            q = Variable(q).cuda()
        pred = model(v, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
