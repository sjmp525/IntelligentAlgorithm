from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import network
from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from dataloader import get_dataloader
import os, random
import numpy as np
from utils.DF_ABNNlib import KL_SoftLabelloss
from network.resnet_8x import resnet18_1w1a, resnet20_1w1a
from utils.DF_ABNNlib import  BinarizeConv2d_BiPer
from utils.DF_ABNNlib import Log_UP, Hist_Show
from network.vgg import vgg_small_1w1a
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

# vp = VisdomPlotter('8097', env='DFAD-cifar')

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn) # 容器中存储假图片与真图片的特征差距

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # KL = KL_BN().cuda()
        # criterion_kd = nn.MSELoss().cuda()

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2) # 假图片与真图片的特征差距：L2范数计算：真图片方差（不变）-假图片方差（在更新） + 真图片均值（不变）-假图片均值（在更新）(好像反了)
        # r_feature = KL(var, module.running_var.data, 2) + KL(mean, module.running_mean.data, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def train(args, teacher, student, generator, device, optimizer, epoch, bn_hook=None, t_hooks=None, tbn_stats=None, s_hooks=None, sbn_stats=None, SumWriter=None):
    teacher.eval()
    student.train()
    generator.train()
    # model_simkd.train()
    optimizer_S, optimizer_G = optimizer
    KL = KL_SoftLabelloss().cuda() # KL散度损失，可用于标签蒸馏
    inter_loss = SCRM().cuda() # 可计算特征损失
    criterion_kd = nn.MSELoss().cuda() # 均方误差损失
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    for i in range(args.epoch_itrs):
        # 固定生成器不动，学生网络迭代5次
        for k in range (5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_S.zero_grad()
            fake = generator(z)

            # Hist_Show(fake[66], '66')
            # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test0 = vutils.make_grid(fake[66].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("66", test0)

            # Hist_Show(fake[88], '88')
            # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test1 = vutils.make_grid(fake[88].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("88", test1)
 
            # Hist_Show(fake[131], '131')
            # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test2 = vutils.make_grid(fake[131].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("131", test2)

            # Hist_Show(fake[199], '199')
            # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test3 = vutils.make_grid(fake[199].float().detach().cpu(), nrow=8, normalize=True)
            # SumWriter.add_image("199", test3)

            # Hist_Show(fake[222], '222')
            # Tensorboard Record
            # SumWriter = SummaryWriter(log_dir='ggg')
            # test1 = vutils.make_grid(fake[13].float().detach().cpu().unsqueeze(dim=1), nrow=8, normalize=True)
            # test4 = vutils.make_grid(fake[222].float().detach().cpu(), nrow=8, normalize=True)
            # SumWr/iter.add_image("222", test4)

            feat_s, logit_s = student(fake) # 学生网络的特征与输出预测
            feat_t, logit_t = teacher(fake) # 教师网络的特征与输出预测

            # cls_t = teacher.get_feat_modules()[-1] # 获取教师网络的全连接层，即教师网络的分类器
            # trans_feat_s, trans_feat_t, pred_feat_s = model_simkd(feat_s[-2], feat_t[-2].detach(), cls_t) # 第一个输出为学生的最后一层卷积后的特征，第二个输出为教师的最后一层卷积后的特征，第三个输出为学生最后一层的卷积特征送入教师分类器后的预测结果

            # loss_fkd = KL(trans_feat_s, trans_feat_t, 2) # 学生最后一层卷积后的特征与教师最后一层卷积后的特征作均方误差损失
            # loss_ghw = KL(pred_feat_s, logit_t, 2) # 复用教师分类器产生的预测结果与老师模型的预测结果作kl散度损失
            loss_lkd = KL(logit_s, logit_t, 2) # 标签作KL散度蒸馏损失
            loss_1kd = F.l1_loss(logit_s, logit_t)

            loss_reg = torch.zeros_like(loss_1kd)
            for module in student.modules():
                if 'Binarize' in module._get_name():
                    loss_reg += module.tau.item() * torch.mean(0.5 - 0.5 * (torch.cos(2 * module.freq * module.weight)))

            # loss_S = F.l1_loss(logit_s, logit_t.detach())
            loss_S = loss_1kd
            loss_S.backward()
            optimizer_S.step()

        # 学生网络固定不动，生成器迭代1次
        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
        # opt_z = optim.Adam([z], lr=0.1)
        # scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, verbose=False, patience=100)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)

        # loss_G = (- torch.log(KL(s_logit, t_logit.detach(), 2.0) + 1) + loss_bn * 0.07)
        feat_s, logit_s = student(fake)
        feat_t, logit_t = teacher(fake)


        # loss_meanvar = args.b * loss_var

        #  - 0.001 * sum([h.r_feature for h in bn_hook])
        # loss_fkd = KL(trans_feat_s, trans_feat_t, 2)
        # loss_ghw = KL(pred_feat_s, logit_t, 2)
        loss_lkd = KL(logit_s, logit_t, 2)
        loss_1kd = F.l1_loss(logit_s, logit_t)

        loss_G = -loss_1kd
        loss_G.backward()
        optimizer_G.step()
        # scheduler_z.step(loss_G.item())

        # if i % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f} -loss_ghw：{} -loss_fkd：{}  loss_bn：{}'.format(
        #         epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item(), - loss_ghw.item(), - loss_fkd.item(), loss_bn.item()))
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f} '.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))

            # vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            # vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())
    #Hist_Show(fake, 'img')
    #print('66666')

def test(args, student, generator, device, test_loader, epoch=0):
    student.eval()
    generator.eval()

    fps_sum =0
    batches = 0

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            import time
            start_time = time.time()
            kong, output = student(data)
            end_time = time.time()
            elapsed_time = (end_time - start_time)
            fps = 256  / elapsed_time
            print(f"模型的FPS: {fps:.2f}")
            fps_sum += fps
            batches +=1

            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            fake = generator(z)
            # if i==0:
                # vp.add_image( 'input', pack_images( denormalize(data,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ) )
                # vp.add_image( 'generated', pack_images( denormalize(fake,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0,1).detach().cpu().numpy() ) )

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_fps = fps_sum / batches
        print(f"平均FPS: {avg_fps:.8f}")

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct / len(test_loader.dataset)
    return acc

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=0.001,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='/home/ghw/data/CIFAR10')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                         help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='resnet20_1w1a', choices=['resnet18_1w1a, resnet20_1w1a, vgg_small_1w1a'],
                        help='model name (default: resnet18_1w1a)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test-only', action='store_true', default=True)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--progressive', dest='progressive', action='store_true',
                        help='progressive train ')
    parser.add_argument('--Use_tSNE', dest='Use_tSNE', action='store_false',
                        help='use tSNE show the penultimate Feature')
    parser.add_argument('--a',type=float, default=0.05,choices=[0.05],
                        help='BN penalty items between teacher(true) and teacher(fake)(default:0.07)')
    parser.add_argument('--b', type=float, default=0.1,
                        help='BN penalty items between student(fake) and teacher(fake)(default:0.2)')
    # parser.add_argument("--teacher_weights", type=str, default="", help="teacher weights path")
    # parser.add_argument("--teacher", type=str, default="resnet56_cifar", help="teacher architecture")
    parser.add_argument('--step', type=str, default='2step', choices=['1step, 2step'],
                        help='training steps')
    parser.add_argument('--Ablation', type=str, default='base+A+B', choices=['base, base+A, base+B, base+A+B'],
                        help='Ablation_Study')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    _, test_loader = get_dataloader(args)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    # teacher = Models.__dict__[args.teacher](num_class=num_classes)
    # if args.teacher_weights:
    #     print('Load Teacher Weights')
    #     teacher_ckpt = torch.load(args.teacher_weights)['model']
    #     teacher.load_state_dict(teacher_ckpt)
    #
    #     for param in teacher.parameters():
    #         param.requires_grad = False
    student = network.resnet_8x.resnet20_1w1a(num_classes=num_classes)
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32)

    teacher.load_state_dict(torch.load(args.ckpt))
    print("Teacher restored from %s" % (args.ckpt))

    # student.load_state_dict(torch.load('BiPer/checkpoint/student/vggsmall_BiPer_2step_cifar10.pt'))
    # print("buffer_a_2step.pt")

    # # checkpoint/student/resnet20_base+A+B_1step_cifar100.pt    checkpoint/student/generator_resnet20_base+A+B_2step_cifar100.pt
    # generator.load_state_dict((torch.load('BiPer/checkpoint/student/generator_vggsmall_BiPer_2step_cifar10.pt')))
    # print("generator_a_2step.pt")

    teacher.eval()

    # Tensorboard Record
    # SumWriter = SummaryWriter(log_dir='log-000')

    # simkd
    # z = torch.randn((256, args.nz, 1, 1))
    # fake = generator(z)
    # feat_s, logit_s = student(fake)
    # feat_t, logit_t = teacher(fake)

    # s_n = feat_s[-2].shape[1]
    # t_n = feat_t[-2].shape[1]
    # model_simkd = SimKD(s_n=s_n, t_n=t_n, factor=2).cuda()
    # trainable_list = nn.ModuleList([]) # 设置两个模块【学生网络、分类器复用器】，加入优化器中以更新
    # trainable_list.append(student)
    # trainable_list.append(model_simkd)

    teacher = teacher.to(device)
    student = student.to(device)
    # model_simkd = model_simkd.to(device)
    generator = generator.to(device)

    # optimizer_S = optim.SGD(trainable_list.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    if args.scheduler:
        # scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_S, args.epochs, eta_min = 0, last_epoch=-1)
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [100, 200], 0.1)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], 0.1)

    best_acc = 0
    if args.test_only:
        acc = test(args, student, generator, device, test_loader)
        return
    acc_list = []

    from network.resnet_8x import resnet20_1w1a, resnet18_1w1a, vgg_small_1w1a
    model = torch.nn.DataParallel(eval('resnet20_1w1a')())
    model.cuda()

    def cpt_tau(epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.0468).float(), torch.tensor(0.0468).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch/args.epochs)]).float() + B
        return tau

    def cpt_ab(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        a = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        b = max(1/t,torch.tensor(1.)).float()
        return a, b

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.progressive:
            t = Log_UP(epoch, args.epochs)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()

            layer_cnt = 0
            param = []
            for m in model.modules():
                if isinstance(m, AdaBin_Conv2d):
                    m.t = t
                    m.k = k
                    layer_cnt += 1

            a, b = cpt_ab(epoch)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.b = b.cuda()
                    module.a = a.cuda()
            for module in model.modules():
                module.epoch = epoch

            line = f"layer : {layer_cnt}, k = {k.cpu().detach().numpy()[0]:.5f}, t = {t.cpu().detach().numpy()[0]:.5f}"
            # log.write("=> " + line + "\n")
            # log.flush()
            print(line)

        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        conv_modules = []
        for name, module in student.named_modules():
            if isinstance(module, BinarizeConv2d_BiPer):
                conv_modules.append(module)

        # * compute threshold tau
        tau = cpt_tau(epoch)
        for module in conv_modules:
            module.tau = tau.cuda()

        z = torch.randn((256, args.nz, 1, 1))
        # fake = generator(z)0
        # Hist_Show(fake, 'fake')
        train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch)# , t_hooks=t_hooks, tbn_stats=tbn_stats, s_hooks=s_hooks, sbn_stats=sbn_stats, SumWriter=SumWriter)
        # Test
        acc = test(args, student, teacher, generator, device, test_loader, epoch)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            # torch.save(student.state_dict(), "Ablation_Study/student/%s_%s_%s_%s.pt"%(args.model, args.Ablation, args.step, args.dataset))
            # torch.save(generator.state_dict(), "Ablation_Study/student/generator_%s_%s_%s_%s.pt"%(args.model, args.Ablation, args.step, args.dataset))
            torch.save(student.state_dict(),"BiPer/checkpoint/student/test.pt")
            torch.save(generator.state_dict(), "BiPer/checkpoint/student/generator_test.pt")
        # vp.add_scalar('Acc', epoch, acc)

    print("Best Acc=%.6f" % best_acc)

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%s_%s_%s.csv' % (args.dataset, args.Ablation, args.model,), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    # model_names = sorted(name for name in Models.__dict__
    #                      if name.islower() and not name.startswith("__")
    #                      and callable(Models.__dict__[name]))
    main()