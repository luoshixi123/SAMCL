import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.cuda import amp
from util.loss.make_loss import DCLLoss
from util.eval_metrics import extract_features_clip
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from data.data_manager import process_query_sysu, process_gallery_sysu
from data.dataloader import TestData
from data.data_manager import process_test_regdb
from util.eval import tester
from model.make_model_clip import build_model

import torch
import torch.nn as nn
import numpy as np
from model.clip.model import Transformer,LayerNorm###导入模块
from sklearn.metrics import pairwise_distances
def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader

def do_train_stage1(args,
                    unlabel_dataset,
                    model,
                    optimizer,
                    scheduler
                    ):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    device = "cuda"
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    with torch.no_grad():

        print("==> Extract RGB features")
        unlabel_dataset.rgb_cluster = True
        unlabel_dataset.ir_cluster = False
        if args.dataset == 'sysu':
            cluster_loader_rgb = get_cluster_loader(unlabel_dataset,64, args.workers)
        if args.dataset == 'regdb':
            cluster_loader_rgb = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        # cluster_loader_rgb = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        features_rgb, pseudo_labels_rgb = extract_features_clip(model, cluster_loader_rgb, modal=1, get_image=True)
        features_rgb = torch.cat([features_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0).cuda()
        pseudo_labels_rgb = torch.cat([pseudo_labels_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0)

        print("==> Extract IR features")
        unlabel_dataset.ir_cluster = True
        unlabel_dataset.rgb_cluster = False
        if args.dataset == 'sysu':
            cluster_loader_ir = get_cluster_loader(unlabel_dataset,64, args.workers)
        if args.dataset == 'regdb':
            cluster_loader_ir = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        # cluster_loader_ir = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        features_ir, pseudo_labels_ir = extract_features_clip(model, cluster_loader_ir, modal=2, get_image=True)
        features_ir = torch.cat([features_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0).cuda()
        pseudo_labels_ir = torch.cat([pseudo_labels_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0)

    del cluster_loader_rgb, cluster_loader_ir

    # adjust pseudo where label is -1
    valid_idx_rgb = np.where(pseudo_labels_rgb.cpu() != -1)[0]
    features_rgb = features_rgb[valid_idx_rgb,:]
    labels_rgb = pseudo_labels_rgb[valid_idx_rgb].cuda()

    valid_idx_ir = np.where(pseudo_labels_ir.cpu() != -1)[0]
    features_ir = features_ir[valid_idx_ir, :]
    labels_ir = pseudo_labels_ir[valid_idx_ir].cuda()

    nums_rgb = len(labels_rgb)
    nums_ir = len(labels_ir)

    start_time = time.monotonic()
    for epoch in range(1, args.stage1_maxepochs+1):
        scheduler.step(epoch)
        model.train()

        if nums_rgb > nums_ir:
            iter_list_rgb = torch.randperm(nums_rgb).to(device)
            iter_list_ir = torch.cat([torch.randperm(nums_ir),torch.randint(0,nums_ir,(nums_rgb-nums_ir,))],dim=0).to(device)
        elif nums_rgb == nums_ir:
            iter_list_rgb = torch.randperm(nums_rgb).to(device)
            iter_list_ir = torch.randperm(nums_ir).to(device)
        else:
            iter_list_ir = torch.randperm(nums_ir).to(device)
            iter_list_rgb = torch.cat([torch.randperm(nums_rgb), torch.randint(0, nums_rgb, (nums_ir - nums_rgb,))], dim=0).to(device)

        batch = args.stage1_batch_size
        i_ter = len(iter_list_rgb) // batch

        print('-----len of rgb and ir iter_list------',len(iter_list_rgb),len(iter_list_ir))
        print('---------------------------------------------------------------------')
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

        loss_meter = AverageMeter()

        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list_rgb = iter_list_rgb[i*batch:(i+1)* batch]
                b_list_ir = iter_list_ir[i*batch:(i+1)* batch]
            else:
                b_list_rgb = iter_list_rgb[i * batch:len(iter_list_rgb)]
                b_list_ir = iter_list_ir[i * batch:len(iter_list_rgb)]

            target_rgb = labels_rgb[b_list_rgb]
            target_ir = labels_ir[b_list_ir]

            image_features_rgb = features_rgb[b_list_rgb]
            image_features_ir = features_ir[b_list_ir]

            text_features_rgb = model(get_text=True, label=target_rgb, modal=1)
            text_features_ir = model(get_text=True, label=target_ir, modal=2)
            mix_features = model(get_fusion_text=True,l=text_features_rgb,s=text_features_ir)
            mix_features_text_rgb = text_features_rgb + 0.8*mix_features
            mix_features_text_ir = text_features_ir + 0.8*mix_features

            loss_i2t_rgb = xent(image_features_rgb, mix_features_text_rgb, target_rgb, target_rgb)
            loss_t2i_rgb = xent(mix_features_text_rgb, image_features_rgb, target_rgb, target_rgb)
            loss_i2t_ir = xent(image_features_ir, mix_features_text_ir, target_ir, target_ir)
            loss_t2i_ir = xent(mix_features_text_ir, image_features_ir, target_ir, target_ir)

            out_rgb = image_features_rgb
            out_ir = image_features_ir  
            out_rgb_de = out_rgb.detach()
            out_ir_de = out_ir.detach()
            DCLLoss_ = DCLLoss(sigma=1, delta=1)
            loss_RC_rgb = DCLLoss_(out_rgb, out_rgb_de)
            loss_RC_ir = DCLLoss_(out_ir, out_ir_de)
            loss_RC =  loss_RC_rgb + loss_RC_ir
      

            loss = loss_i2t_rgb + loss_t2i_rgb + loss_i2t_ir + loss_t2i_ir
            loss = loss + 10.0*loss_RC
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item())

            torch.cuda.synchronize()
            if i % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss_prompt: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (i + 1), i_ter+1,
                                        loss_meter.avg, scheduler._get_lr(epoch)[0]))

            # if epoch % args.stage1_checkpoint == 0:
            #     torch.save(model.state_dict(), os.path.join(args.model_path, args.logs_file + '_stage1_{}.pth'.format(epoch)))

        if epoch == args.stage1_maxepochs:
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

                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1.pth"))
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

                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1_regdb.pth"))

            else:
                print('please input correct dataset!!')
            
    end_time = time.monotonic()
    print('Stage1 running time: ', timedelta(seconds=end_time - start_time))













