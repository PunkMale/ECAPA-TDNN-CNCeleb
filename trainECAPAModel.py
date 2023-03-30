'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse
import glob
import time
import torch
import warnings

from torch.utils.data import DataLoader
from ECAPAModel import ECAPAModel
from dataset import CNCeleb
from tools import *


#################  你需要修改的一些路径  #################
cn1_root = '/home2/database/sre/CN-Celeb-2022/task1/cn_1/'
cn2_dev = '/home2/database/sre/CN-Celeb-2022/task1/cn_2/data'
train_list_path = 'data/cn2_train_list.csv'
trials_path = "data/trials.lst"
save_path = "exps/cn2"
device = 'cuda:0'
batch_size = 64
initial_model = ''
######################################################

parser = argparse.ArgumentParser(description="ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int, default=200, help='输入语音长度，200帧为2秒')
parser.add_argument('--max_epoch', type=int, default=80, help='训练多少个epoch')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
parser.add_argument('--n_cpu', type=int, default=4, help='DataLoader时使用多少核心')
parser.add_argument('--test_step', type=int, default=1, help='跑几个epoch测试一下性能')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument("--lr_decay", type=float, default=0.97, help='学习率衰减率')
parser.add_argument("--device", type=str, default=device, help='训练设备')

## 训练、测试路径、模型保存路径
parser.add_argument('--train_list', type=str, default=train_list_path, help='训练列表')
parser.add_argument('--train_path', type=str, default=cn2_dev, help='训练数据路径')
parser.add_argument('--eval_list', type=str, default=trials_path, help='测试trails')
parser.add_argument('--eval_path', type=str, default=cn1_root, help='测试数据路径')
parser.add_argument('--save_path', type=str, default=save_path, help='模型保存路径')

## 设置embedding维度和margin loss超参数
parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int, help='Number of speakers')

## 运行模式
parser.add_argument('--eval', dest='eval', action='store_true', help='训练还是测试')
parser.add_argument('--resume', dest='resume', action='store_true', help='是否恢复之前的训练')
parser.add_argument('--initial_model', type=str, default=initial_model, help='从哪个模型继续')

## 初始化、设置模型和打分文件保存路径
warnings.simplefilter("ignore")  # 忽略警告
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## 定义数据集和Dataloader
dataset = CNCeleb(**vars(args))
trainLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
args.n_class = dataset.n_class

## glob:查找符合特定规则的文件路径名
model_files = glob.glob('%s/*.model' % args.model_save_path)
model_files.sort()

## 只进行测试，前提是有初始模型
if args.eval:
    model = ECAPAModel(**vars(args))
    print("Model {} 已加载!".format(args.initial_model))
    model.load_parameters(args.initial_model)
    EER, minDCF = model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
    print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
    quit()

## 如果初始模型存在，系统将从初始模型开始训练
if args.initial_model != "":
    print("Model {} 已加载!".format(args.initial_model))
    model = ECAPAModel(**vars(args))
    model.load_parameters(args.initial_model)
    epoch = 0
## 尝试从已保存的模型和epoch开始
elif args.resume and len(model_files) >= 1:
    print("Model %s loaded from previous state!" % model_files[-1])
    epoch = int(os.path.splitext(os.path.basename(model_files[-1]))[0][6:]) + 1
    model = ECAPAModel(**vars(args))
    model.load_parameters(model_files[-1])
## 系统从头开始训练
else:
    model = ECAPAModel(**vars(args))
    epoch = 0

EERs = []
score_file = open(args.score_save_path, "a+")

while epoch < args.max_epoch:
    ## 训练模型
    # loss, lr, acc = model.train_network(epoch=epoch, loader=trainLoader)

    ## 评估模型
    if epoch % args.test_step == 0:
        # model.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
        EER, minDCF = model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
        print('EER:{:.4}  minDCF:{:.4}'.format(EER, minDCF))
        # EERs.append(EER)
        # print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)))
        # score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (epoch, lr, loss, acc, EERs[-1], min(EERs)))
        # score_file.flush()  # 刷新缓冲区
    epoch += 1
