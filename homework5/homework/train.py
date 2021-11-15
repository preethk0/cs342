from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    model = model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data', num_workers=4, transform=transform)

    det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # size_loss = torch.nn.MSELoss(reduction='none')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for img, act in train_data:
            image, action = img.to(device), act.to(device)
            steer = model(image)
            
            loss_val = det_loss(steer[..., :2], action[..., :2])
            # size_w, _ = gt_det.max(dim=1, keepdim=True)

            # det, size = model(img)
            # Continuous version of focal loss_val
            # p_det = torch.sigmoid(det * (1-2*gt_det))
            # det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
            # size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
            # loss_val = det_loss_val + size_loss_val * args.size_weight

            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, image, action, global_step)

            if train_logger is not None:
                train_logger.add_scalar('det_loss', loss_val, global_step)
                # train_logger.add_scalar('size_loss', size_loss_val, global_step)
                # train_logger.add_scalar('loss_val', loss_val, global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if valid_logger is None or train_logger is None:
            print('epoch %-3d' %
                  (epoch))
        save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
