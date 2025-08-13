import argparse
import ast
import os

import torch
import yaml

from src.processor import processor



# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='STGTP') 
    parser.add_argument('--P',  default=True, type=ast.literal_eval)
    parser.add_argument('--S',  default=True, type=ast.literal_eval)
    
    parser.add_argument('--model', default='STGTP', choices=["LSTM", "Seq2Seq","STGTP", "GRU", 
                                                            "iTransformer", "Transformer", "BiGRU", "BiLSTM", "Mamba", 'STGCNN', 'NCE'])
    parser.add_argument('--dataset', default='TianJinPort')
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--dim', default=128, type=int, help='Transformer dim_feedforward') # 128 or 64
    parser.add_argument('--image_size', default=112, type=int, help='Gaussian heatmap image size') # 112 or 28
    parser.add_argument('--patch_size', default=16, type=int, help='Vit patch size') # 16 or 4
    parser.add_argument('--vit_dim', default=512, type=int, help='Vit dim') # 512 or 64
    parser.add_argument('--sigma', default=5, type=int, help='Gaussian heatmap sigma') # 512 or 64
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    
    parser.add_argument('--load_model', default='final', type=str, help="load pretrained model for test or training")
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=1024, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=1024, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=0, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=False, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=0.5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--clip', default=1, type=int)

    return parser

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    p.model_dir = p.save_base_dir + str(p.model) + '_' + str(p.dataset) + '/'
    p.save_dir = p.save_base_dir + str(p.dataset) + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'
        
    save_arg(p)
    
    args = p
    if args.using_cuda:
        torch.cuda.set_device(0)

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
