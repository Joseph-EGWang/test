import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from models.resnet import ResNet, BasicBlock 
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from tqdm import tqdm
import numpy as np
import os



# # 可插拔
from utils.mix_dummy_aug import topKFrequent,generate_label,generate_reduced_mix_label,generate_reduced_mix_label_intra_clusters,generate_reduced_mix_label_inter_clusters
from utils.mix_dummy_aug import CIFAR100_50_50_10_reduced_classifier_para,CIFAR100_50_50_20_reduced_classifier_para,CIFAR100_80_20_10_reduced_classifier_para,CIFAR100_80_20_20_reduced_classifier_para
from utils.mix_dummy_aug import CIFAR100_50_50_random_sampled_mix_classes,CIFAR100_80_20_random_sampled_mix_classes
from utils.nets import MultiHeadResNet
# #






def train(model, train_loader, labeled_eval_loader, args, reduced_classifier_num, lebel2cluster, cluster2lebel):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    # print(reduced_classifier_num, lebel2cluster, cluster2lebel)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            images, label = x.to(device), label.to(device)


            # # 可插拔
            batch_size = 0
            if args.aug_type != "None":
                mix_times = args.mix_times
                batch_size = images.size()[0]
                mix_data = []
                mix_target = []
                # print('mix_times:',mix_times)

                # if args.aug_type == "random_sample_unknown_num_classes":
                #     try:
                #         inputs, targets = next(self.inverse_iter)
                #     except:
                #         self.inverse_iter = iter(self.random_classes_lab_train_loader)
                #         inputs, targets = next(self.inverse_iter)
                #     inputs = torch.cat((inputs[0], inputs[1]), 0).cuda()
                #     targets = targets.repeat(2).cuda()
                #     # print('inputs:',inputs.size(),targets.size(),targets)
                #     # print('aaa')

                if not args.reduce_time:
                    for _ in range(mix_times):
                        index = torch.randperm(batch_size)
                        # 遍历batch_size
                        for n in range(batch_size):
                            # 如果两个标签的值不同
                            if label[n] != label[index][n]:
                                # 融合两个标签，得到新的标签
                                if args.aug_type == "mix_dummy_reduced_classifiers":
                                    new_label = generate_reduced_mix_label(lebel2cluster, cluster2lebel,
                                                                           label[n].item(), label[index][n].item(),
                                                                           args.num_labeled_classes)
                                elif args.aug_type == "mix_dummy_reduced_classifiers_intra_clusters":
                                    if lebel2cluster[label[n].item()] == lebel2cluster[label[index][n].item()]:
                                        new_label = generate_reduced_mix_label_intra_clusters(lebel2cluster,
                                                                                              cluster2lebel,
                                                                                              label[n].item(),
                                                                                              label[index][n].item(),
                                                                                              args.num_labeled_classes)
                                    else:
                                        continue
                                elif args.aug_type == "mix_dummy_reduced_classifiers_inter_clusters":
                                    if lebel2cluster[label[n].item()] != lebel2cluster[label[index][n].item()]:
                                        new_label = generate_reduced_mix_label_inter_clusters(lebel2cluster,
                                                                                              cluster2lebel,
                                                                                              label[n].item(),
                                                                                              label[index][n].item(),
                                                                                              args.num_labeled_classes)
                                    else:
                                        continue
                                    # print('new_label:',new_label)
                                    # print('aaa')
                                # elif args.aug_type == "random_sample_unknown_num_classes":
                                #
                                #     if targets[n] < targets[index][n]:
                                #         label_pair = (targets[n].item(), targets[index][n].item())
                                #     else:
                                #         label_pair = (targets[index][n].item(), targets[n].item())
                                #     # print('label_pair:',label_pair)
                                #     if label_pair in self.random_sampled_mix_classes.keys():
                                #         new_label = self.random_sampled_mix_classes[
                                #                         label_pair] + args.num_labeled_classes
                                #         # print('new_label:',new_label)
                                #     else:
                                #         # print('aaa')
                                #         continue
                                else:
                                    new_label = generate_label(label[n].item(), label[index][n].item(), args)
                                # print('aaa')
                                # print('new_label:',new_label)
                                # 从beta分布中采样参数
                                lam = np.random.beta(20.0, 20.0)
                                # 如果参数的数量大于0.4或者小于0.6
                                if lam < 0.4 or lam > 0.6:
                                    # 将参数值设置为0.5
                                    lam = 0.5
                                # 对数据进行mixup
                                if args.aug_type == "random_sample_unknown_num_classes":
                                    pass
                                    # mix_data.append(lam * inputs[n] + (1 - lam) * inputs[index, :][n])
                                else:
                                    mix_data.append(lam * images[n] + (1 - lam) * images[index, :][n])
                                # 存储新的标签
                                mix_target.append(new_label)
                    # print('mix_target:',len(mix_target),mix_target)
                    # print(aaa)
                    # 将新标签转换为张量
                    new_target = torch.Tensor(mix_target).cuda()
                    # print('new_target:',new_target.device,labels.device)
                    # 将原始数据和增广后的数据拼接
                    labels = torch.cat((label, new_target.long()), 0)
                    for item in mix_data:
                        images = torch.cat((images, item.unsqueeze(0)), 0)
                else:
                    for _ in range(mix_times):
                        index = torch.randperm(batch_size)
                        # 遍历batch_size
                        # 融合两个标签，得到新的标签
                        lam = np.random.beta(20.0, 20.0)

                        # 如果参数的数量大于0.4或者小于0.6
                        if lam < 0.4 or lam > 0.6:
                            # 将参数值设置为0.5
                            lam = 0.5

                        # 从beta分布中采样参数
                        for n in range(batch_size):
                            if args.hparams.aug_type == "mix_dummy_reduced_classifiers":
                                new_label = generate_reduced_mix_label(lebel2cluster, cluster2lebel,
                                                                       label[n].item(), label[index][n].item(),
                                                                       args.num_labeled_classes)

                            else:
                                new_label = generate_label(label[n].item(), label[index][n].item(), args)
                            mix_target.append(new_label)

                        # 对数据进行mixup
                        mix_images = lam * images + (1 - lam) * images[index, :]
                        # print('mix_images:',mix_images.size())
                        mix_data.append(mix_images)
                        # 存储新的标签

                    # 将新标签转换为张量
                    new_target = torch.Tensor(mix_target).cuda()
                    # print('new_target:',new_target.size())
                    # 将原始数据和增广后的数据拼接
                    labels = torch.cat((label, new_target.long()), 0)
                    # print('images:',images.size())
                    mix_datas = torch.stack(mix_data).view(-1, images.size(1), images.size(2), images.size(3))
                    # print('mix_datas:',mix_datas.size())
                    images = torch.cat((images, mix_datas), 0)

            model.normalize_prototypes()
            print(images.shape,labels.shape)
            # print(aaa)

            ####

            # print(type(images))
            # output1, _, _ = model(images)
            # 进行前向传播
            is_train = True
            outputs = model(images, batch_size, is_train)
            output1 = outputs["logits_lab"]
            # print(type(output1), output1.shape)
            # print(aaa)
            loss= criterion1(output1, labels)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        # args.head = 'head1'
        test(model, labeled_eval_loader, args)

def test(model, test_loader, args):
    model.eval() 
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        images, labels = x.to(device), label.to(device)
        batch_size_val = images.size()[0]
        is_train = False
        logits = model(images, batch_size_val, is_train)["logits_lab"]
        if args.hparams.aug_type != "None":
            logits = logits[:, :args.num_labeled_classes]
        # print('logits:',logits.size())
        # 获取预测
        _, pred = logits.max(dim=-1)

        # output1, output2, _ = model(x)
        # if args.head=='head1':
        #     output = output1
        # else:
        #     output = output2
        # _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return preds 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--rotnet_dir', type=str, default='./data/experiments/selfsupervised_learning/rotnet_cifar100.pth')
    parser.add_argument('--model_name', type=str, default='resnet_rotnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--mode', type=str, default='train')

    # # 可插拔
    parser.add_argument('--aug_type', default="None", type=str, help='data augmentation type',
                        choices=("None", 'mixup', 'mix_dummy', 'mix_dummy_reduced_classifiers',
                                 'mix_dummy_reduced_classifiers_intra_clusters',
                                 'mix_dummy_reduced_classifiers_inter_clusters', 'random_sample_unknown_num_classes'))
    parser.add_argument("--mix_times", default=4, type=int, help="number of mix_times")
    parser.add_argument("--dummy_nums", default=50, type=int, help="number of dummy_classes")
    parser.add_argument('--add_bias', default=False, action="store_true", help='add_bias')
    parser.add_argument("--reduced_classifier_type", type=str, default='None', choices=(
    'None', 'CIFAR100_50_50_10_clusters', 'CIFAR100_50_50_20_clusters', 'CIFAR100_80_20_10_clusters',
    'CIFAR100_80_20_20_clusters'), help="pretrained checkpoint path")

    parser.add_argument("--random_sampled_unknown_dataset", type=str, default='None',
                        choices=('None', 'CIFAR100_50_50', 'CIFAR100_80_20'), help="pretrained checkpoint path")
    parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
    parser.add_argument("--reduce_time", default=False, action="store_true", help="disable reduce time")
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    # #


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name)

    # print("1")

    # # 可插拔

    reduced_classifier_num = None
    lebel2cluster = None
    cluster2lebel = None

    if args.reduced_classifier_type == "CIFAR100_50_50_10_clusters":
        # 获取单个类别标签到超类簇的映射
        lebel2cluster = CIFAR100_50_50_10_reduced_classifier_para[0]
        # 获取超类簇到单个类别标签的映射
        cluster2lebel = CIFAR100_50_50_10_reduced_classifier_para[1]
        # 获取减少后分类器的数量
        if args.aug_type == "mix_dummy_reduced_classifiers":
            reduced_classifier_num = CIFAR100_50_50_10_reduced_classifier_para[2]
        if args.aug_type == "mix_dummy_reduced_classifiers_intra_clusters":
            reduced_classifier_num = CIFAR100_50_50_10_reduced_classifier_para[3]
        if args.aug_type == "mix_dummy_reduced_classifiers_inter_clusters":
            reduced_classifier_num = CIFAR100_50_50_10_reduced_classifier_para[4]
        # print('reduced_classifier_num:',self.reduced_classifier_num)
        # print('aaa')
    if args.reduced_classifier_type == "CIFAR100_50_50_20_clusters":
        lebel2cluster = CIFAR100_50_50_20_reduced_classifier_para[0]
        cluster2lebel = CIFAR100_50_50_20_reduced_classifier_para[1]
        # 获取减少后分类器的数量
        if args.aug_type == "mix_dummy_reduced_classifiers":
            reduced_classifier_num = CIFAR100_50_50_20_reduced_classifier_para[2]
        if args.aug_type == "mix_dummy_reduced_classifiers_intra_clusters":
            reduced_classifier_num = CIFAR100_50_50_20_reduced_classifier_para[3]
        if args.aug_type == "mix_dummy_reduced_classifiers_inter_clusters":
            reduced_classifier_num = CIFAR100_50_50_20_reduced_classifier_para[4]

    if args.reduced_classifier_type == "CIFAR100_80_20_10_clusters":
        lebel2cluster = CIFAR100_80_20_10_reduced_classifier_para[0]
        cluster2lebel = CIFAR100_80_20_10_reduced_classifier_para[1]
        # 获取减少后分类器的数量
        if args.aug_type == "mix_dummy_reduced_classifiers":
            reduced_classifier_num = CIFAR100_80_20_10_reduced_classifier_para[2]
        if args.aug_type == "mix_dummy_reduced_classifiers_intra_clusters":
            reduced_classifier_num = CIFAR100_80_20_10_reduced_classifier_para[3]
        if args.aug_type == "mix_dummy_reduced_classifiers_inter_clusters":
            reduced_classifier_num = CIFAR100_80_20_10_reduced_classifier_para[4]

    if args.reduced_classifier_type == "CIFAR100_80_20_20_clusters":
        lebel2cluster = CIFAR100_80_20_20_reduced_classifier_para[0]
        cluster2lebel = CIFAR100_80_20_20_reduced_classifier_para[1]
        # 获取减少后分类器的数量
        if args.aug_type == "mix_dummy_reduced_classifiers":
            reduced_classifier_num = CIFAR100_80_20_20_reduced_classifier_para[2]
        if args.aug_type == "mix_dummy_reduced_classifiers_intra_clusters":
            reduced_classifier_num = CIFAR100_80_20_20_reduced_classifier_para[3]
        if args.aug_type == "mix_dummy_reduced_classifiers_inter_clusters":
            reduced_classifier_num = CIFAR100_80_20_20_reduced_classifier_para[4]

    # if args.aug_type == "random_sample_unknown_num_classes":
    #     if args.random_sampled_unknown_dataset == "CIFAR100_50_50":
    #         random_sampled_mix_classes = CIFAR100_50_50_random_sampled_mix_classes
    #     if args.random_sampled_unknown_dataset == "CIFAR100_80_20":
    #         random_sampled_mix_classes = CIFAR100_80_20_random_sampled_mix_classes
    #
    #     random_classes_lab_train = extra_data.random_classes_lab_train
    #     random_classes_lab_train_loader = torch.utils.data.DataLoader(
    #         random_classes_lab_train,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=True,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=True,
    #         drop_last=True,
    #     )
    #     self.inverse_iter = iter(random_classes_lab_train_loader)

    #

    print(args.num_labeled_classes,args.num_unlabeled_classes,args.aug_type,args.dummy_nums,args.add_bias,reduced_classifier_num)
    model = MultiHeadResNet(
        arch=args.arch,  # 获取网络名称
        low_res="CIFAR",  # 获取数据集名称是否为CIFAr
        num_labeled=args.num_labeled_classes,  # 获取已知类别的数量
        num_unlabeled=args.num_unlabeled_classes,  # 获取未知类别的数量
        num_heads=None,  # 获取分类头的数量,是否使用无标签分类头
        aug_type=args.aug_type,
        dummy_nums=args.dummy_nums,
        proto_bias=args.add_bias,
        reduced_classifier_num=reduced_classifier_num
    )


    # model1 = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    state_dict = torch.load(args.rotnet_dir)
    del state_dict['linear.weight']
    del state_dict['linear.bias']
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        # print(name)
        # 只更新两个分类头和最后一层
        if 'head' not in name and 'layer4' not in name and 'novel' not in name:
            print(name)
            param.requires_grad = False
    # model1.load_state_dict(state_dict, strict=False)
    # for name, param in model1.named_parameters():
    #     if 'head' not in name and 'layer4' not in name:
    #         print(name)
    print(model)
    # print(model1)
    # print(aaa)
    model = model.to(device)

    if args.dataset_name == 'cifar10':
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'cifar100':
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'svhn':
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))

    if args.mode == 'train':
        train(model, labeled_train_loader, labeled_eval_loader, args, reduced_classifier_num, lebel2cluster, cluster2lebel)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    # print('test on labeled classes')
    # args.head = 'head1'
    # test(model, labeled_eval_loader, args)
