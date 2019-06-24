import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils
import os
from torch.autograd import Variable
import torchvision
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    #train
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--v_dim', type=int, default=1024, help='size of visual features')
    parser.add_argument('--output', type=str, default='ckpt')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--layer', type=str, default='layer3')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--early_stop', type=int, default=5)
    #eval
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default='ckpt/best.pth')
    #inference
    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--image', type=str, default="img.jpg")
    parser.add_argument('--question', type=str, default="What is life ?")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write("Output dir is %s"%args.output)

    batch_size = args.batch_size
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.data_root, layer=args.layer, inference=args.inference)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    model = base_model.build_model(eval_dset,
                                   args.v_dim,
                                   args.num_hid,
                                   logger=logger).cuda()

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    if args.eval:
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()
        score, upper_bound = evaluate(model,eval_loader)
        print('\teval score: %.2f (%.2f)' % (100 * score, 100 * upper_bound))

    elif args.inference:
        model.load_state_dict(torch.load(args.ckpt))
        # transform image for resnet input
        img = eval_dset._read_image(args.image)
        # get features
        cnn = torchvision.models.resnet50(pretrained=True).cuda().eval()
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        getattr(cnn, args.layer).register_forward_hook(get_activation(args.layer))
        img = torch.unsqueeze(img, 0).cuda()  # 1,dim,x,x
        cnn(img)
        v = activation[args.layer]
        v = v.view(v.size(0), v.size(1), -1)
        v = v.permute(0, 2, 1)
        #process question
        q = eval_dset.dictionary.tokenize(args.question, False)
        q = torch.from_numpy(np.array(q)).unsqueeze(0)

        v = Variable(v).cuda()
        q = Variable(q).cuda()

        pred = model(v, q, None)
        print("Question:",args.question)
        print("Answer:",eval_dset.label2ans[torch.max(pred,1)[1]])

    else:
        train_dset = VQAFeatureDataset('train', dictionary, dataroot=args.data_root, layer=args.layer)
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
        train(model, train_loader, eval_loader,
              args.epochs,
              args.output,
              args.lr,
              args.early_stop,
              logger)


