# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
import torch
import os
import MyDataset
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
import gc




if __name__ == '__main__':
    args = utils.get_args()
    # 参数
    #fileRoot = r'/home/hlu/Data/VIPL'
    fileRoot = r'C:\Users\samal\Desktop\Saved'

    
    saveRoot = r'C:\Users\samal\Desktop\Saved' + str(args.fold_num) + str(args.fold_index)
    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(64, frames_num))
    best_mae = 20

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    rPPGNet_name = 'rPPGNet' + input_form + 'n' + str(frames_num) + 'fn' + str(fold_num) + 'fi' + str(fold_index)


    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu') #
        print('on GPU')
    else:
        print('on CPU')



    # 数据集
    if args.reData == 1:
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=fold_num, fold_index=fold_index)
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'img_mvavg_full.png', 5, frames_num)
        Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'img_mvavg_full.png', 5, frames_num)
    train_db = MyDataset.Data_VIPL(root_dir=(saveRoot + '_Train'), frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    test_db = MyDataset.Data_VIPL(root_dir=(saveRoot + '_Test'), frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    train_loader = DataLoader(train_db, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)
    print('trainLen:', len(train_db), 'testLen:', len(test_db))
    print('fold_num:', fold_num, 'fold_index', fold_index)

    rPPGNet = model.UNet(3, 1)

    if reTrain == 1:
        rPPGNet = torch.load(rPPGNet_name, map_location=device)
        print('load ' + rPPGNet_name + ' right')

    rPPGNet.to(device=device)
    optimizer_rPPG = torch.optim.Adam(rPPGNet.parameters(), lr=learning_rate)

    loss_func_rPPG = utils.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_SP = utils.SP_loss(device, clip_length=frames_num).to(device)
    # noises为生成网络的输入
    noises = torch.randn(batch_size_num, 1, 4, int(frames_num/16)).to(device)

    # for step, (samples, targets,gt) in enumerate(train_loader):
    #     print("STEP:",step,"\n")
    #     print("SAMPLE:",samples,"--->",samples.shape)
        
    #     print("\n ")
    #     print("TARGET:",targets,"--->",targets.shape)
    #     print("\n ")
    #     print("GT:",gt,"--->",gt.shape)
    #     exit()

    for epoch in range(epoch_num):
        rPPGNet.train()
        #for step, (data, bvp, HR_rel, idx) in enumerate(train_loader):
        print("for loop start --------------------------------------")
        #for step, (data, HR_rel, idx) in enumerate(train_loader): ]
        print(len(train_loader))
        for step, (data,HR_rel,idx) in enumerate(train_loader):
            print(step,'*********')  
            data = Variable(data).float().to(device=device)
            #bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            #bvp = bvp.unsqueeze(dim=1)
            STMap = data[:, :, :, 0:frames_num]
            #Wave = bvp[:, :, 0:frames_num]
            #b, _, _ = Wave.size()

            #########################################################
            # 训练rPPGNet
            #########################################################
            optimizer_rPPG.zero_grad()
            with torch.no_grad():
                Wave_pr, HR_pr = rPPGNet(STMap)

            loss0 = loss_func_L1(HR_rel, HR_pr)
            loss1 = torch.zeros(1).to(device)
            loss2 = torch.zeros(1).to(device)
            whole_max_idx = []
            for width in range(64):
                #loss1 = loss1 + loss_func_rPPG(Wave_pr[:, :, width, :], Wave)   ####
                #loss1 = loss1 + loss_func_rPPG(Wave_pr[:, :, width, :], Wave)
                loss2_temp, whole_max_idx_temp = loss_func_SP(Wave_pr[:, :, width, :], HR_rel)
                loss2 = loss2_temp + loss2
                whole_max_idx.append(whole_max_idx_temp.data.cpu().numpy())
            HR_Droped = utils.Drop_HR(np.array(whole_max_idx))
            loss1 = loss1/64
            loss2 = loss2/64
            loss = loss0 + loss1 + loss2
            loss.backward()
            optimizer_rPPG.step()
            if step % 50 == 0:
                print('Train Epoch: ', epoch,
                      '| loss: %.4f' % loss.data.cpu().numpy(),
                      '| loss0: %.4f' % loss0.data.cpu().numpy(),
                      '| loss1: %.4f' % loss1.data.cpu().numpy(),
                      '| loss2: %.4f' % loss2.data.cpu().numpy(),
                      )
            del Wave_pr,HR_pr

        print("for loop end--------------------------")
        torch.cuda.empty_cache()
        gc.collect()

        print(" 2nd for loop start -----------------------")

        # 测试
        rPPGNet.eval()
        loss_mean = []
        Label_pr = []
        Label_gt = []
        HR_pr_temp = []
        HR_rel_temp = []
        HR_pr2_temp = []
        #for step, (data, bvp, HR_rel, idx) in enumerate(test_loader):
        print(len(test_loader))
        for step, (data, HR_rel, idx) in enumerate(test_loader):
            
            if step==45:
                break

            print(step,'*********') 
            print("data size:",len(data),"----","target size",len(HR_rel))
            print("data shape:",data.shape,"----","target shape",HR_rel.shape)
            data = Variable(data).float().to(device=device)
           # bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            #bvp = bvp.unsqueeze(dim=1)
            STMap = data[:, :, :, 0:frames_num]
            #Wave = bvp[:, :, 0:frames_num]
            #b, _, _ = Wave.size()

            with torch.no_grad():
                Wave_pr, HR_pr = rPPGNet(STMap)

            loss0 = loss_func_L1(HR_pr, HR_rel)
            loss1 = torch.zeros(1).to(device)
            loss2 = torch.zeros(1).to(device)
            whole_max_idx = []
            for width in range(64):
                #loss1 = loss1 + loss_func_rPPG(Wave_pr[:, :, width, :], Wave)
                loss2_temp, whole_max_idx_temp = loss_func_SP(Wave_pr[:, :, width, :], HR_rel)
                loss2 = loss2_temp + loss2
                whole_max_idx.append(whole_max_idx_temp.data.cpu().numpy())
            HR_Droped = utils.Drop_HR(np.array(whole_max_idx))
            loss1 = loss1 / 64
            loss2 = loss2 / 64
            loss = loss0 + loss1 + loss2
            print(HR_pr.data.cpu().numpy())
            HR_pr_temp.extend(HR_pr.data.cpu().numpy())
            HR_pr2_temp.extend(HR_Droped)
            HR_rel_temp.extend(HR_rel.data.cpu().numpy())
            if step % 50 == 0:
                print('Train Epoch: ', epoch,
                      '| loss0: %.4f' % loss0.data.cpu().numpy(),
                      '| loss1: %.4f' % loss1.data.cpu().numpy(),
                      '| loss2: %.4f' % loss2.data.cpu().numpy(),
                      )
            del Wave_pr,HR_pr
        print('HR:')
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr2_temp, HR_rel_temp)
        torch.save(rPPGNet, rPPGNet_name)
        print('saveModel As ' + rPPGNet_name)

        if best_mae > MAE:
            best_mae = MAE
            io.savemat(rPPGNet_name + 'HR_pr.mat', {'HR_pr': HR_pr_temp})
            io.savemat(rPPGNet_name + 'HR_rel.mat', {'HR_rel': HR_rel_temp})
        


        

