import torch
import torch.nn as nn
from torch.autograd import Function
import time

class BinaryQuantize_Tanh(Function):
    """
        反向传播近似函数：aa * tanh(bb * x)
    """
    @staticmethod
    def forward(ctx, input, aa, bb):
        ctx.save_for_backward(input, aa, bb)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, aa, bb = ctx.saved_tensors
        grad_input = aa * bb * (1 - torch.pow(torch.tanh(input * bb), 2)) * grad_output
        return grad_input, None, None


class BinaryQuantize_RBNN(Function):
    """
        反向传播近似函数：...
        Copy from Paper: RBNN
    """
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


class BinaryQuantize(Function):
    """
        反向传播近似函数：...
        可变HardTanh近似函数，高度aa，可更新范围(-1/bb, 1/bb)
    """
    @staticmethod
    def forward(ctx, input, aa, bb):
        ctx.save_for_backward(input, aa, bb)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, aa, bb = ctx.saved_tensors
        grad_input = (aa * bb) * grad_output
        grad_input[input.gt(1./bb)] = 0
        grad_input[input.lt(-1./bb)] = 0
        return grad_input, None, None


class MyApproxSign(nn.Module):
    """
        输入：全精度特征 input = [B, C, W, H]
        输出：二值特征 out = [B, C, W, H]
        操作：Sign-使用反向传播近似函数
        By Wanzixin
    """
    def __init__(self):
        super(MyApproxSign, self).__init__()
        self.buffer_size = 100                              # 数据记录缓存区大小，最小=1；   100为经验所得.

        self.pp = torch.tensor([1.0]).float().cuda()        # 当前数据的可反馈百分比，由外部传入
        self.bb = torch.tensor([1.0]).float().cuda()        # 当前的量化更新范围，默认以0为分界处
        self.buffer = torch.zeros(self.buffer_size).cuda()  # 缓存区，用于存储bb的历史取值
        self.index = 0                                      # 缓存区的存储索引

    def forward(self, x):
        # 仅在训练时
        if x.requires_grad:
            # 分别对 0 左右的数据，根据可反馈百分比，求出当前的量化更新范围
            with torch.no_grad():
                # 注意：torch.nanquantile无法处理大于一千万个数的数据；np.nanquantile虽然可以，但太费时间
                # 左半部分为负值，故需取绝对值
                Left_values = torch.abs(x[x <= 0][:10000000]).float()
                Right_values = x[x >= 0][:10000000].float()

                if len(Left_values) != 0 and len(Right_values) == 0:
                    bb = Left_values.nanquantile(self.pp)
                elif len(Left_values) == 0 and len(Right_values) != 0:
                    bb = Right_values.nanquantile(self.pp)
                else:
                    bb = torch.max(Left_values.nanquantile(self.pp), Right_values.nanquantile(self.pp))

                # 使用buffer区缓存数据生成的平均bb值
                # 记录每个mini-batch的bb取值
                self.buffer[self.index] = bb
                self.index = (self.index + 1) % self.buffer_size  # 最大长度时归0
                # 根据buffer区的有效数据，取平均得到bb最终值
                self.bb = torch.mean(self.buffer[self.buffer != 0.0])
                self.bb = torch.clamp(self.bb, min=1e-1)  # 设置最小值，否则训练末期振荡

        # 获取最终的 aa and bb
        aa = torch.clamp(self.bb, min=1.0)  # 函数的高度
        bb = 1. / self.bb  # 1./ 函数的更新长度

        # 近似函数 三种选择
        x = BinaryQuantize_Tanh().apply(x, aa, bb)
        # out = BinaryQuantize_RBNN().apply(x, aa, bb)
        # out = BinaryQuantize().apply(x, aa, bb)
        # x = BinaryQuantize().apply(x, torch.tensor([1.0]).cuda(), torch.tensor([1.0]).cuda())   # baseline 固定Clip(x)

        return x

class MyBiFunA(nn.Module):
    """
        输入：全精度特征 input
        输出：二值特征 out
        操作：out = sign((input - mean) / alpha) * alpha + mean
             Sign 使用反向传播近似函数
    """
    def __init__(self):
        super(MyBiFunA, self).__init__()
        self.batch_size = 128                               # --- 需在此单独修改 !!!
        self.buffer_size = 1000                             # 数据记录缓存区大小，最小=1

        self.pp = torch.tensor([1.0]).float().cuda()        # 当前数据的可反馈百分比，由外部传入
        self.bb = torch.tensor([0.0]).float().cuda()        # 当前的量化更新范围，默认以0为分界处
        self.buffer = torch.zeros(self.buffer_size).cuda()  # 缓存区，用于存储bb的历史取值
        self.index = 0                                      # 缓存区的存储索引

        # # 激活的均值，模长记录【记录个数 = 自定数 * Batch_Size】
        # self.Mean_A = torch.tensor([0.0]).float().cuda()
        # self.Mean_A_buffer = torch.zeros(self.buffer_size, self.batch_size).cuda()
        # self.Mean_A_index = 0
        # self.alpha_A = torch.tensor([0.0]).float().cuda()
        # self.alpha_A_buffer = torch.zeros(self.buffer_size, self.batch_size).cuda()
        # self.alpha_A_index = 0

    def forward(self, x):
        start_time = time.time()
        # # 作用：减去均值，使信息熵最大化
        # # 训练时：记录经过特征的均值，使用buffer区数据得到 self.Mean_A
        # # 推理时：直接读取 self.Mean_A，进行推理
        # if x.requires_grad:
        #     Mean_A = x.float().view(x.size(0), -1).mean(-1)
        #     self.Mean_A_buffer[self.Mean_A_index] = Mean_A.detach()
        #     self.Mean_A_index = (self.Mean_A_index + 1) % self.buffer_size
        #     self.Mean_A = torch.mean(self.Mean_A_buffer[self.Mean_A_buffer != 0])
        #     Mean_A = Mean_A.view(-1, 1, 1, 1)
        # else:
        #     Mean_A = self.Mean_A
        # x = x - Mean_A
        #
        # # 作用：除以所有元素的均方根，使向量模长和二值参数一致
        # # 训练时：记录经过特征的模长，使用buffer区数据得到 self.alpha_A
        # # 推理时：直接读取 self.alpha_A，进行推理
        # if x.requires_grad:
        #     alpha_A = torch.sqrt((x ** 2).sum((1, 2, 3)) / (x.size(1) * x.size(2) * x.size(3)))
        #     self.alpha_A_buffer[self.alpha_A_index] = alpha_A.detach()
        #     self.alpha_A_index = (self.alpha_A_index + 1) % self.buffer_size
        #     self.alpha_A = torch.mean(self.alpha_A_buffer[self.alpha_A_buffer != 0])
        #     alpha_A = alpha_A.view(-1, 1, 1, 1)
        # else:
        #     alpha_A = self.alpha_A
        # x = x / alpha_A

        # -------------------------- 梯度近似（以下） -------------------------- #
        # 仅在训练时
        if x.requires_grad:
            # 分别对 0 左右的数据，根据可反馈百分比，求出当前的量化更新范围
            with torch.no_grad():
                # 注意：torch.nanquantile无法处理大于一千万个数的数据；np.nanquantile虽然可以，但太费时间
                Left_values = torch.abs(x[x <= 0][:10000000]).float()  # 左半部分为负值，故需取绝对值
                Right_values = x[x >= 0][:10000000].float()
                if len(Left_values) != 0 and len(Right_values) == 0:
                    bb = Left_values.nanquantile(self.pp)
                elif len(Left_values) == 0 and len(Right_values) != 0:
                    bb = Right_values.nanquantile(self.pp)
                else:
                    bb = torch.max(Left_values.nanquantile(self.pp), Right_values.nanquantile(self.pp))

            # 记录每个mini-batch的bb取值
            self.buffer[self.index] = bb
            self.index = (self.index + 1) % self.buffer_size  # 最大长度时归0

            # 根据buffer区的有效数据，取平均得到bb最终值
            # self.bb = (bb + torch.mean(self.buffer[self.buffer != 0])) / 2.
            self.bb = torch.mean(self.buffer[self.buffer != 0])
            self.bb = torch.clamp(self.bb, min=1e-1)  # 设置最小值，否则训练末期振荡

            # if self.index== 0:
            #     print('Activation: ', 'bb =', self.bb.item(), 'pp =', self.pp.item(), 'Mean =', self.Mean_A.item(), 'Alpha =', self.alpha_A.item())

        # 获取最终的 aa and bb
        aa = torch.clamp(self.bb, min=1.0)  # 函数的高度
        bb = 1. / self.bb  # 1./ 函数的更新长度
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time elapsed: {:.4f} seconds".format(elapsed_time))
        # 近似函数
        out = BinaryQuantize_Tanh().apply(x, aa, bb)
        # out = BinaryQuantize_RBNN().apply(x, aa, bb)
        # out = BinaryQuantize().apply(x, aa, bb)
        # -------------------------- 梯度近似（以上） -------------------------- #
        #
        # # 还原，使二值特征分布和全精度特征的分布更加近似
        # out = out * alpha_A + Mean_A

        return out
# class MyBiFunA(nn.Module):
#     """
#         输入：全精度特征 input
#         输出：二值特征 out
#         操作：out = sign((input - mean) / alpha) * alpha + mean
#              Sign = ApproxSign 使用反向传播近似函数
#         By Wanzixin
#     """
#     def __init__(self):
#         super(MyBiFunA, self).__init__()
#         self.batch_size = 128                              # --- 需在此单独修改 !!!
#         self.buffer_size = 1000                             # 数据记录缓存区大小，最小=1；   1000为经验所得.
#
#         # 激活的均值、模长比例因子记录【记录个数 = 自定数 * Batch_Size】
#         self.register_buffer('Mean_A', torch.tensor([0.0]))
#         self.Mean_A_buffer = torch.zeros(self.buffer_size, self.batch_size).cuda()
#         self.Mean_A_index = 0
#         self.register_buffer('alpha_A', torch.tensor([0.0]))
#         self.alpha_A_buffer = torch.zeros(self.buffer_size, self.batch_size).cuda()
#         self.alpha_A_index = 0
#
#         self.myapproxsign = MyApproxSign()
#
#     def forward(self, x):
#         start_time = time.time()
#         # 作用：减去均值，使信息熵最大化
#         # 训练时：记录经过特征的均值，使用buffer区数据得到 self.Mean_A
#         # 推理时：直接读取模型参数 self.Mean_A，进行推理
#         if x.requires_grad:
#             Mean_A = x.float().view(x.size(0), -1).mean(-1)
#             self.Mean_A_buffer[self.Mean_A_index] = Mean_A.detach()
#             self.Mean_A_index = (self.Mean_A_index + 1) % self.buffer_size
#             self.Mean_A = torch.mean(self.Mean_A_buffer[self.Mean_A_buffer != 0.0]).view(1)  # must .view(1) else false infer
#             Mean_A = torch.mean(Mean_A.view(-1, 1, 1, 1))
#         else:
#             Mean_A = self.Mean_A
#         x = x - Mean_A
#
#         # 作用：除以所有元素的均方根，使向量模长和二值参数一致
#         # 训练时：记录经过特征的模长，使用buffer区数据得到 self.alpha_A
#         # 推理时：直接读取 self.alpha_A，进行推理
#         if x.requires_grad:
#             alpha_A = torch.sqrt((x ** 2).sum((1, 2, 3)) / (x.size(1) * x.size(2) * x.size(3)))
#             self.alpha_A_buffer[self.alpha_A_index] = alpha_A.detach()
#             self.alpha_A_index = (self.alpha_A_index + 1) % self.buffer_size
#             self.alpha_A = torch.mean(self.alpha_A_buffer[self.alpha_A_buffer != 0.0]).view(1)
#             alpha_A = torch.mean(alpha_A.view(-1, 1, 1, 1))
#         else:
#             alpha_A = self.alpha_A
#         x = x / alpha_A
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print("Time elapsed: {:.4f} seconds".format(elapsed_time))
#         # -------------------------- 梯度近似 -------------------------- #
#         x = self.myapproxsign(x)
#
#         # 还原，使二值特征分布和全精度特征的分布更加近似
#         x = x * alpha_A + Mean_A
#
#         return x


class MyBiFunW(nn.Module):
    """
        输入：全精度权重 input_weight
        输出：二值权重 out
        操作：out = sign((input_weight - mean) / alpha)
             Sign = ApproxSign 使用反向传播近似函数
        By Wanzixin
    """
    def __init__(self):
        super(MyBiFunW, self).__init__()
        self.myapproxsign = MyApproxSign()

    def forward(self, x):
        # 减去均值，使信息熵最大化
        Mean_W = x.float().view(x.size(0), -1).mean(-1).view(-1, 1, 1, 1)
        x = x - Mean_W
        # 除以所有元素的均方根，使向量模长与二值参数一致==根号n，得到重塑规范后的最终全精度权重
        alpha_W = torch.sqrt((x ** 2).sum((1, 2, 3)) / (x.size(1) * x.size(2) * x.size(3))).view(-1, 1, 1, 1)  # 所有元素的均方根
        x = x / alpha_W

        # -------------------------- 梯度近似 -------------------------- #
        x = self.myapproxsign(x)
        return x


class RPReLU(nn.Module):
    """
        Nonlinear function
        Copy from Paper: ReActNet   https://github.com/liuzechun/ReActNet
    """
    def __init__(self, channel):
        super(RPReLU, self).__init__()
        self.bias1 = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)
        self.prelu = nn.PReLU(channel)
        self.bias2 = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)

    def forward(self, x):
        x = x + self.bias1  # .expand_as(x)
        x = self.prelu(x)
        x = x + self.bias2
        return x


class Maxout(nn.Module):
    """
        Nonlinear function
        Copy from Paper: AdaBin   https://github.com/huawei-noah/Efficient-Computing/tree/master/BinaryNetworks/AdaBin
    """
    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = nn.Parameter(neg_init * torch.ones(channel))
        self.pos_scale = nn.Parameter(pos_init * torch.ones(channel))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pos_scale.view(1, -1, 1, 1) * self.relu(x) - self.neg_scale.view(1, -1, 1, 1) * self.relu(-x)
        return x


def main():
    # By Wanzixin
    tensor_size = (8, 6, 4, 4)
    x = torch.arange(torch.prod(torch.tensor(tensor_size))).reshape(*tensor_size)
    print(x)

    Num = x.shape[1]
    intermedia = x.permute(1, 0, 2, 3).contiguous().view(Num, -1).detach()
    print(intermedia)


if __name__ == "__main__":
    main()