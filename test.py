import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.SwinLSTM_D import SwinLSTM
from configs import get_args
from dataset import Moving_MNIST_Test
from functions import test
from utils import set_seed, make_dir, init_logger

if __name__ == '__main__':
    # 获取参数配置对象
    args = get_args()
    set_seed(args.seed)
    # 创建相关结果目录
    cache_dir, model_dir, log_dir = make_dir(args)
    # 创建日志对象
    logger = init_logger(log_dir)

    # 初始化模型
    model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                     in_chans=args.input_channels, embed_dim=args.embed_dim,
                     depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                     num_heads=args.heads_number, window_size=args.window_size).to(args.device)
    criterion = nn.MSELoss()


    # 加载测试数据
    # test_dataset = BHDataset(args,split='test')
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
    #                          num_workers=0, shuffle=False, pin_memory=True, drop_last=True)

    # 加载模型
    model.load_state_dict(torch.load('./results/model/trained_model_state_dict'))

    # 记录开始时间
    start_time = time.time()

    # 小样本进行测试实验，得到相关评价指标
    _, mse, ssim = test(args, logger, 0, model, test_loader, criterion, cache_dir,'test')

    # 打印结果
    print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
    # 打印训练耗时
    print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

