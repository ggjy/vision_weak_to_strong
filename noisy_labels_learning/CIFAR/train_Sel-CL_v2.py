import argparse
import pathlib
import random
import sys

import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim

from dataset.cifar_dataset import *

parent_path = pathlib.Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)
sys.path.append(parent_path)
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

from utils_sel.init_env import init_env
from utils_sel.utils_noise_v2 import *
from utils_sel.test_eval import test_eval
from utils_sel.queue_with_pro import *
from utils_sel.kNN_test_v2 import *
from utils_sel.MemoryMoCo import MemoryMoCo
from utils_sel.other_utils import *
from utils_sel.models.preact_resnet import *
from utils_sel.lr_scheduler import get_scheduler
from apex import amp


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--epoch', type=int, default=250, help='training epoches')
    parser.add_argument('--warmup_way', type=str, default="uns", help='uns, sup')
    parser.add_argument('--warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr', '--base-learning-rate', '--base-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--lr-warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[125, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")
    
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--noise_type', default='asymmetric', help='symmetric or asymmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='percent of noise')
    parser.add_argument('--train_root', default='./dataset', help='root for train data')
    parser.add_argument('--out', type=str, default='./out', help='Directory of the output')
    parser.add_argument('--experiment_name', type=str, default='Proof',
                        help='name of the experiment (for the output files)')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    
    parser.add_argument('--network', type=str, default='PR18', help='Network architecture')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--DA', type=str, default="complex", help='Choose simple or complex data augmentation')
    
    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--alpha_moving', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--alpha', type=float, default=0.5, help='example selection th')
    parser.add_argument('--beta', type=float, default=0.25, help='pair selection th')
    parser.add_argument('--uns_queue_k', type=int, default=10000, help='uns-cl num negative sampler')
    parser.add_argument('--uns_t', type=float, default=0.1, help='uns-cl temperature')
    parser.add_argument('--sup_t', default=0.1, type=float, help='sup-cl temperature')
    parser.add_argument('--sup_queue_use', type=int, default=1, help='1: Use queue for sup-cl')
    parser.add_argument('--sup_queue_begin', type=int, default=3, help='Epoch to begin using queue for sup-cl')
    parser.add_argument('--queue_per_class', type=int, default=1000,
                        help='Num of samples per class to store in the queue. queue size = queue_per_class*num_classes*2')
    parser.add_argument('--aprox', type=int, default=1,
                        help='Approximation for numerical stability taken from supervised contrastive learning')
    parser.add_argument('--lambda_s', type=float, default=0.01, help='weight for similarity loss')
    parser.add_argument('--lambda_c', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--k_val', type=int, default=250, help='k for k-nn correction')
    
    args = parser.parse_args()
    return args


def data_config(args, transform_train, transform_test):
    trainset, testset = get_dataset(args, TwoCropTransform(transform_train), transform_test)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)
    print('############# Data loaded #############')
    
    return train_loader, test_loader, trainset


def build_model(args, device):
    if args.network == 'PR18':
        model = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
        model_ema = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    elif args.network == 'PR34':
        model = PreActResNet34(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
        model_ema = PreActResNet34(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    elif args.network == 'R18':
        model = ResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
        model_ema = ResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    elif args.network == 'R34':
        model = ResNet34(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
        model_ema = ResNet34(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    elif args.network == 'R50':
        model = ResNet50(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
        model_ema = ResNet50(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    
    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    return model, model_ema


def main(args):
    exp_name = f'noise_models_{args.network}_{args.dataset}_SI{args.seed_initialization}_SD{args.seed_dataset}'
    
    exp_path = os.path.join(args.out, 'noise_models_V2', exp_name, args.noise_type, str(args.noise_ratio))
    res_path = os.path.join(args.out, 'metrics_V2', exp_name, args.noise_type, str(args.noise_ratio))
    
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    
    __console__ = sys.stdout
    name = "/results"
    log_file = open(res_path + name + ".log", 'a')
    sys.stdout = log_file
    print(args)
    
    args.best_acc = 0
    best_acc5 = 0
    best_acc_val = 0.0
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed
    
    random.seed(args.seed_initialization)  # python seed for image transformation
    
    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    if args.DA == "complex":
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
        ])
    
    else:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
        ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
    ])
    
    # data loader
    num_classes = args.num_classes
    
    train_loader, test_loader, trainset = data_config(args, transform_train, transform_test)
    
    model, model_ema = build_model(args, device)
    uns_contrast = MemoryMoCo(args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=2)
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    
    if args.sup_queue_use == 1:
        queue = queue_with_pro(args, device)
    else:
        queue = []
    
    np.save(res_path + '/' + str(args.noise_ratio) + '_noisy_labels.npy', np.asarray(trainset.noisy_labels))
    
    for epoch in range(args.initial_epoch, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name, args.noise_ratio)
        if (epoch <= args.warmup_epoch):
            if (args.warmup_way == 'uns'):
                train_uns(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader, optimizer,
                          epoch)
            else:
                train_selected_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                                                    pin_memory=True,
                                                                    sampler=torch.utils.data.WeightedRandomSampler(
                                                                            torch.ones(len(trainset)), len(trainset)))
                train_sup(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                          train_selected_loader, optimizer, epoch)
        else:
            train_selected_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                                                pin_memory=True,
                                                                sampler=torch.utils.data.WeightedRandomSampler(
                                                                        selected_examples, len(selected_examples)))
            train_sel(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                      train_selected_loader, optimizer, epoch, features, selected_pair_th, selected_examples)
        
        features = compute_features(args, model, train_loader, test_loader)
        if (epoch >= args.warmup_epoch):
            print('######## Pair-wise selection ########')
            selected_examples, selected_pair_th = pair_selection(args, model, device, train_loader, test_loader, epoch,
                                                                 features)
        
        print('Epoch time: {:.2f} seconds\n'.format(time.time() - st))
        log_file.flush()
        print('######## Test ########')
        test_eval(args, model, device, test_loader)
        
        acc, acc5 = kNN(args, model, test_loader, 200, 0.1, features, epoch, train_loader)
        if acc >= args.best_acc:
            args.best_acc = acc
            best_acc5 = acc5
        print('KNN top-1 precion: {:.4f} {:.4f}, best is: {:.4f} {:.4f}'.format(acc * 100.,
                                                                                acc5 * 100., args.best_acc * 100.,
                                                                                best_acc5 * 100))
        
        if (epoch % 10 == 0):
            save_model(model, optimizer, args, epoch, exp_path + "/Sel-CL_model.pth")
            np.save(res_path + '/' + 'selected_examples_train.npy', selected_examples.data.cpu().numpy())


if __name__ == "__main__":
    args = parse_args()
    
    init_env()
    args.train_root = os.getenv('DATA_PATH')
    args.out = os.getenv('OUT_PATH')
    
    print(args)
    main(args)
