from utils import *
import torch.nn as nn
from configs import get_args
from functions import train, test
from torch.utils.data import DataLoader
from dataset import Moving_MNIST



def setup(args):
    if args.model == 'SwinLSTM-D':
        from models.SwinLSTM_D import SwinLSTM
        model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 加载训练集
    train_dataset = Moving_MNIST(args, split='train')
    # 加载验证集
    valid_dataset = Moving_MNIST(args, split='valid')

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size,
                             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    return model, criterion, optimizer, train_loader, valid_loader

def main():
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, train_loader, valid_loader = setup(args)
    # 记录训练损失和验证损失
    train_losses, valid_losses = [], []
    # 最好的参数
    best_metric = (0, float('inf'), float('inf'))
    # 迭代轮次
    for epoch in range(args.epochs):
        # 记录开始时间
        start_time = time.time()
        # 训练并返回损失
        train_loss = train(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        # 绘制损失曲线
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)
        # 判断是否需要验证
        if (epoch + 1) % args.epoch_valid == 0:
            # 进行验证，返回相关损失和指标
            valid_loss, mse, ssim = test(args, logger, epoch, model, valid_loader, criterion, cache_dir,'valid')

            valid_losses.append(valid_loss)
            # 绘制验证集的损失曲线
            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)

            # 根据mse对模型进行保存
            if mse < best_metric[1]:
                # 保存模型
                torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict_'+args.model)
                # 以元组的形式记录最佳指标
                best_metric = (epoch, mse, ssim)

            # 添加日志
            logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')
        # 打印一轮训练用掉的时间
        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

if __name__ == '__main__':
    main()
