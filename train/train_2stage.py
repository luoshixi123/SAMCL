
import collections
import os
import os.path as osp

from datetime import timedelta
import time
import sys
import random
from sklearn.metrics import adjusted_rand_score

import easydict
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from util.loss.make_loss import DCLLoss
# from tensorboardX import SummaryWriter
from torch.cuda import amp

from util.eval_metrics import extract_features_clip
from util.faiss_rerank import compute_jaccard_distance
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from ClusterContrast.cm import ClusterMemory,ClusterMemory_2
from data.data_manager import process_query_sysu, process_gallery_sysu
from data.data_manager import process_test_regdb
from data.dataloader import SYSUData_Stage2, RegDBData_Stage2, IterLoader, TestData
from util.eval import tester
from util.utils import IdentitySampler_nosk, GenIdx

from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR

import collections
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
# torch.backends.cuda.max_split_size_mb = 2750

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader


def do_train_stage2(args,
                    unlabel_dataset,
                    model,
                    optimizer,
                    scheduler,
                    loss_fn_rgb,
                    loss_fn_ir,
                    ):
    best_acc = 0
    device = 'cuda'
    epochs = args.stage2_maxepochs
    start_time = time.monotonic()

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset == 'sysu':
        transform_train_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5)
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    else:
        transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5),
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])


    batch = args.stage2_ims_per_batch
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir
    i_ter_rgb = num_classes_rgb // batch
    i_ter_ir = num_classes_ir // batch
    left_rgb = num_classes_rgb-batch* (num_classes_rgb//batch)
    left_ir = num_classes_ir-batch* (num_classes_ir//batch)
    if left_rgb != 0 :
        i_ter_rgb = i_ter_rgb+1
    if left_ir != 0 :
        i_ter_ir = i_ter_ir+1
    text_features_rgb = []
    text_features_ir = []
    if args.dataset == 'regdb':
        with torch.no_grad():
            for i in range(i_ter_rgb):
                if i+1 != i_ter_rgb:
                    l_list_rgb = torch.arange(i*batch, (i+1)* batch)
                else:
                    l_list_rgb = torch.arange(i*batch, num_classes_rgb)
                #  with amp.autocast(enabled=True):
                text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
                text_features_rgb.append(text_feature_rgb.cpu())
            text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
        with torch.no_grad():
            for i in range(i_ter_ir):
                if i+1 != i_ter_ir:
                    l_list_ir = torch.arange(i*batch, (i+1)* batch)
                else:
                    l_list_ir = torch.arange(i*batch, num_classes_ir)
                # with amp.autocast(enabled=True):
                text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
                text_features_ir.append(text_feature_ir.cpu())
            text_features_ir = torch.cat(text_features_ir, 0).cuda()
    if args.dataset == 'sysu':
        with torch.no_grad():
            for i in range(i_ter_rgb):
                if i+1 != i_ter_rgb:
                    l_list_rgb = torch.arange(i*batch, (i+1)* batch)
                else:
                    l_list_rgb = torch.arange(i*batch, num_classes_rgb)
                #  with amp.autocast(enabled=True):
                text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
                text_features_rgb.append(text_feature_rgb.cpu())
            text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
        with torch.no_grad():
            for i in range(i_ter_ir):
                if i+1!= i_ter_ir:
                    l_list_ir = torch.arange(i*batch, (i+1)* batch)
                else  :
                    l_list_ir = torch.arange(i*batch, num_classes_ir)

                text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
                text_features_ir.append(text_feature_ir.cpu())
            text_features_ir = torch.cat(text_features_ir, 0).cuda()


    scaler = amp.GradScaler()
    losses = AverageMeter()
    losses_rgb = AverageMeter()
    losses_ir = AverageMeter()
    losses_i2t = AverageMeter()
    losses_i2t_rgb = AverageMeter()
    losses_i2t_ir = AverageMeter()
    losses_DCL_rgb = AverageMeter()
    losses_DCL_ir =AverageMeter()
    losses_DCL = AverageMeter()


    # torch.cuda.empty_cache()

    for epoch in range(1, epochs+1):
        with torch.no_grad():
            if epoch == 1:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled data')
            print("==> Extract RGB features")
            unlabel_dataset.rgb_cluster = True
            unlabel_dataset.ir_cluster = False
            if args.dataset == 'sysu':
                cluster_loader_rgb = get_cluster_loader(unlabel_dataset, 64, args.workers)
            if args.dataset == 'regdb':
                cluster_loader_rgb = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
            features_rgb, _ = extract_features_clip(model, cluster_loader_rgb, modal=1, get_image=False)
            features_rgb = torch.cat([features_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0)

            print("==> Extract IR features")
            unlabel_dataset.ir_cluster = True
            unlabel_dataset.rgb_cluster = False
            if args.dataset == 'sysu':
                cluster_loader_ir = get_cluster_loader(unlabel_dataset, 64, args.workers)
            if args.dataset == 'regdb':
                cluster_loader_ir = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
            features_ir, _ = extract_features_clip(model, cluster_loader_ir, modal=2, get_image=False)
            features_ir = torch.cat([features_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0)

            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2)
            pseudo_labels_rgb = cluster.fit_predict(rerank_dist_rgb)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2)
            pseudo_labels_ir = cluster.fit_predict(rerank_dist_ir)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
      ###ARI

            # == 过滤噪声点 ==
        # 转换为 NumPy 数组
            pseudo_labels_rgbs = np.array(pseudo_labels_rgb)
            pseudo_labels_irs = np.array(pseudo_labels_ir)
            true_labels_rgb = np.array(unlabel_dataset.train_color_label)
            true_labels_ir = np.array(unlabel_dataset.train_thermal_label)

            # 过滤噪声点
            valid_indices_rgb = pseudo_labels_rgbs != -1
            valid_indices_ir = pseudo_labels_irs != -1

            pseudo_labels_rgbs = pseudo_labels_rgbs[valid_indices_rgb]
            true_labels_rgb = true_labels_rgb[valid_indices_rgb]

            pseudo_labels_irs = pseudo_labels_irs[valid_indices_ir]
            true_labels_ir = true_labels_ir[valid_indices_ir]

            # 计算 ARI
            ari_rgb = adjusted_rand_score(true_labels_rgb, pseudo_labels_rgbs)
            ari_ir = adjusted_rand_score(true_labels_ir, pseudo_labels_irs)

            print(f"Epoch {epoch} - RGB ARI: {ari_rgb:.4f}")
            print(f"Epoch {epoch} - IR ARI: {ari_ir:.4f}")
        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        del  cluster_loader_rgb, cluster_loader_ir, rerank_dist_rgb, rerank_dist_ir

        if args.dataset == 'regdb' and epoch == args.base_epoch:
            optimizer = make_optimizer_2stage_later(args, model)
            scheduler = WarmupMultiStepLR(optimizer, args.stage2_steps, args.stage2_gamma, args.stage2_warmup_factor,
                                                args.stage2_warmup_iters, args.stage2_warmup_method)
        if epoch >= args.base_epoch:
            change_scale = args.change_scale
            print('----------Start Memory(rgb and ir) Change!----------')
            
        else:
            change_scale = 1.


        memory = ClusterMemory(2048, num_cluster_rgb, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, change_scale=change_scale).cuda()
        memory.features_rgb = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory.features_ir = F.normalize(cluster_features_ir, dim=1).cuda()
        # generate new dataset
        end = time.time()
        if args.dataset == 'sysu':
            trainset = SYSUData_Stage2(args.data_path, pseudo_labels_rgb, pseudo_labels_ir, transform_train_rgb,
                                 transform_train_ir)
        else:
            trainset = RegDBData_Stage2(args.data_path, args.trial, pseudo_labels_rgb, pseudo_labels_ir, transform_train_rgb,
                                 transform_train_ir)
        print("New Dataset Information---- ")
        print("  ----------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------")
        print("  visible  | {:5d} | {:8d}".format(len(np.unique(trainset.train_color_pseudo_label)),
                                                  len(trainset.train_color_image)))
        print("  thermal  | {:5d} | {:8d}".format(len(np.unique(trainset.train_thermal_pseudo_label)),
                                                  len(trainset.train_thermal_image)))
        print("  ----------------------------")
        print("Data loading time:\t {:.3f}".format(time.time() - end))

        color_pos, thermal_pos = GenIdx(trainset.train_color_pseudo_label, trainset.train_thermal_pseudo_label)

        sampler = IdentitySampler_nosk(trainset.train_color_pseudo_label, trainset.train_thermal_pseudo_label, color_pos, thermal_pos,
                                       args.num_instances, args.batch_size)

        trainset.cIndex = sampler.index1
        trainset.tIndex = sampler.index2

        trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                      num_workers=args.workers,
                                      drop_last=True)

        losses.reset()
        losses_rgb.reset()
        losses_ir.reset()
        losses_i2t.reset()
        losses_i2t_rgb.reset()
        losses_i2t_ir.reset()
        losses_DCL_rgb.reset()
        losses_DCL_ir.reset()
        losses_DCL.reset()


        
        model.train()
        for n_iter, (img_rgb, img_ir, label_rgb, label_ir, vid_rgb, vid_ir) in enumerate(trainloader):

            optimizer.zero_grad()
            img_rgb = img_rgb.to(device)
            label_rgb = label_rgb.to(device)
            vid_rgb = vid_rgb.to(device)

            img_ir = img_ir.to(device)
            label_ir = label_ir.to(device)
            vid_ir = vid_ir.to(device)
            image_features, image_features_proj = model(x1=img_rgb, x2=img_ir, modal=0)
            mix_features = model(get_fusion_text=True,l=text_features_rgb,s=text_features_ir)
            mix_features_text_rgb = text_features_rgb +0.8* mix_features
            mix_features_text_ir = text_features_ir + 0.8*mix_features
            logits_rgb = image_features_proj[:img_rgb.size(0)] @ mix_features_text_rgb.t()
            logits_ir = image_features_proj[img_rgb.size(0):] @ mix_features_text_ir.t()
            loss_i2t_rgb = loss_fn_rgb(logits_rgb, vid_rgb)
            loss_i2t_ir = loss_fn_ir(logits_ir, vid_ir)
            loss_i2t = loss_i2t_rgb + loss_i2t_ir
            out_rgb = image_features[:img_rgb.size(0)]
            out_ir = image_features[img_rgb.size(0):]
            loss_rgb, loss_ir = memory(out_rgb, out_ir, label_rgb, label_ir)
            loss = loss_rgb + loss_ir + loss_i2t
            if epoch < args.base_epoch:
                out_rgb_de = out_rgb.detach()
                out_ir_de = out_ir.detach()
                DCLLoss_ = DCLLoss(sigma=1, delta=1)
                loss_DCL_rgb = DCLLoss_(out_rgb, out_rgb_de)
                loss_DCL_ir = DCLLoss_(out_ir, out_ir_de)
                loss_DCL =  loss_DCL_rgb + loss_DCL_ir
                loss = loss + 10*loss_DCL

            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

            losses_rgb.update(loss_rgb.item())
            losses_ir.update(loss_ir.item())
            losses_i2t.update(loss_i2t.item())
            losses_i2t_rgb.update(loss_i2t_rgb.item())
            losses_i2t_ir.update(loss_i2t_ir.item())
            losses_DCL_rgb.update(loss_DCL_rgb.item())
            losses_DCL_ir.update(loss_DCL_ir.item())
            losses_DCL.update(loss_DCL.item())

            losses.update(loss.item())
            torch.cuda.synchronize()
            if n_iter % args.print_freq == 0:
                print("Epoch[{}] Iteration[{}/{}], Loss_rgb_ir_i2t:({:.3f})({:.3f})({:.3f})({:.3f}) ({:.3f}), Base Lr: {:.2e}"
                 .format(epoch, (n_iter + 1), len(trainloader), losses_rgb.avg, losses_ir.avg,
                         losses_i2t_rgb.avg, losses_i2t_ir.avg,losses.avg,scheduler.get_lr()[0]))
        scheduler.step()
        if epoch % args.eval_step == 0 or (epoch == args.stage2_maxepochs):
            if args.dataset == 'sysu':
                print('Test Epoch: {}'.format(epoch))
                test_mode = [1, 2]
                query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                for trial in range(10):
                    # print('-------test trial {}-------'.format(trial))
                    gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                    cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                            feat_dim=2048,
                                            query_cam=query_cam, gall_cam=gall_cam)

                    if trial == 0:
                        all_cmc = cmc
                        all_mAP = mAP
                        all_mINP = mINP
                    else:
                        all_cmc = all_cmc + cmc
                        all_mAP = all_mAP + mAP
                        all_mINP = all_mINP + mINP

                cmc = all_cmc / 10
                mAP = all_mAP / 10
                mINP = all_mINP / 10
                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))

                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage2.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))
            elif args.dataset == 'regdb':
                print('Test Epoch: {}'.format(epoch))

                query_img, query_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='visible')
                gall_img, gall_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='thermal')

                test_mode = [2, 1]
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

                # testing data loader
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                        feat_dim=2048)

                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))
                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage2_regdb.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))
            else:
                print('please input correct dataset!!')

        torch.cuda.empty_cache()

    end_time = time.monotonic()
    print('Stage2 running time: ', timedelta(seconds=end_time - start_time))

class RCLoss(nn.Module):
    def __init__(self, sigma=1, delta=1):
        super(RCLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb):

        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W = torch.exp(-T_dist.pow(2) / self.sigma)

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss
def trim_tensor(tensor, target_len):
    """
    将张量在指定维度上裁剪到目标长度。
    """
    num, feature_size = tensor.size()
    if num > target_len:
        return tensor[:target_len, :]
    else:
        return tensor
def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_features = torch.tensor(image_features)
    text_features = torch.tensor(text_features)

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    # image_norm = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    # text_norm = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
    text_norm = text_norm.to('cuda')
    image_norm = image_norm.to('cuda')
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)

      

    return per_loss
def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid= torch.from_numpy(pid)
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 
