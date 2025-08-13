import torch
import torch.nn as nn

from .utils import *
import torch.distributions as dist
from tqdm import tqdm
import torch.nn.functional as F
from .model.STGTP import STGTP
import torch
import torch.nn as nn

        
def get_net(format):

    net = {
        "STGTP": STGTP
    }[format]
    
    return net

class processor(object):
    def __init__(self, args):

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = get_net(args.model)(args)
        
        self.set_optimizer()

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        if not os.path.isdir(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.net_file = open(os.path.join(self.args.model_dir, f'{self.args.model}_net.txt'), 'w+')
        self.net_file.write(str(self.net))
        self.net_file.close()

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.model_dir + '/' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.model_dir + '/' + str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)
            else:
                print('No')

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        test_error, test_final_error, fd_error = self.test_epoch()
        print('Set: {}, epoch: {},ADE: {} FDE: {} FD: {}'.format(self.args.dataset,
                                                                                          self.args.load_model,
                                                                                       test_error, test_final_error, fd_error))
        return test_error, test_final_error, fd_error
    
    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)
            self.log_file_curve = open(os.path.join(self.args.model_dir, f'{self.args.dataset}_{self.args.model}_log_curve.txt'), 'a+')
            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error, fd_error = self.test_epoch()
                if test_error < self.best_ade or test_final_error < self.best_fde:
                    self.best_ade = test_error
                    self.best_epoch = epoch
                    self.save_model(epoch)
                    self.best_fde = test_final_error
            
                self.log_file_curve.write(
                    str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                        self.args.learning_rate) + '\n')
                self.log_file_curve.close()

            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, ADE={:.5f}, FDE={:.5f}, FD={:.5f}, Best_ADE={:.5f}, Best_FDE={:.5f} at Epoch {}'
                        .format(epoch, train_loss, test_error, test_final_error, fd_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'
                      .format(epoch, train_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])
                loss = torch.zeros(1).cuda()
            else: 
                loss = torch.zeros(1)
               
            batch_abs, seq_list, nei_list, socail_list, nei_num, batch_pednum = inputs

            obs_length = self.args.obs_length
            seq_length = self.args.seq_length
            pred_length = self.args.pred_length
            
            y_max = batch_abs[:seq_length, :, 0].max(axis = 0)[0] + 1e-5
            x_max = batch_abs[:seq_length, :, 1].max(axis = 0)[0] + 1e-5

            y_min = batch_abs[:seq_length, :, 0].min(axis = 0)[0] - 1e-5
            x_min = batch_abs[:seq_length, :, 1].min(axis = 0)[0] - 1e-5
            
            nodes_obs_max = torch.cat([
                y_max.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
                x_max.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
            ]).permute(2, 1, 0)

            nodes_obs_min = torch.cat([
                y_min.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
                x_min.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
            ]).permute(2, 1, 0)
            
            batch_obs = (batch_abs[:obs_length] - nodes_obs_min) / (nodes_obs_max - nodes_obs_min)
           
            

            inputs_forward = batch_obs, seq_list[:obs_length], nei_list[:obs_length], nei_num[:obs_length], batch_pednum
            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)
            lossmask, num = getLossMask(outputs, seq_list[obs_length], seq_list[obs_length:], using_cuda=self.args.using_cuda)
            
            nodes_obs_max = torch.cat([
                y_max.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
                x_max.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
            ]).permute(2, 1, 0)

            nodes_obs_min = torch.cat([
                y_min.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
                x_min.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
            ]).permute(2, 1, 0)
            
#             batch_pred = (batch_abs[obs_length:] - nodes_obs_min) / (nodes_obs_max - nodes_obs_min)
            
            outputs = outputs * (nodes_obs_max - nodes_obs_min) + nodes_obs_min
            loss_o = torch.sum(self.criterion(outputs, batch_abs[obs_length:]), dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()   
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()                
            #self.scheduler.step()
            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(),
                                                                                               end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, fd_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, fd_cnt_epoch = 1e-5, 1e-5, 1e-5
    
        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, seq_list, nei_list, socail_list, nei_num, batch_pednum = inputs
            
            num_of_objs = batch_abs.shape[1]
            
            obs_length = self.args.obs_length
            seq_length = self.args.seq_length
            pred_length = self.args.pred_length
            
            y_max = batch_abs[:seq_length, :, 0].max(axis = 0)[0] + 1e-5
            x_max = batch_abs[:seq_length, :, 1].max(axis = 0)[0] + 1e-5

            y_min = batch_abs[:seq_length, :, 0].min(axis = 0)[0] - 1e-5
            x_min = batch_abs[:seq_length, :, 1].min(axis = 0)[0] - 1e-5

            nodes_obs_max = torch.cat([
                y_max.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
                x_max.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
            ]).permute(2, 1, 0)

            nodes_obs_min = torch.cat([
                y_min.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
                x_min.unsqueeze(1).repeat(1, obs_length).unsqueeze(0),
            ]).permute(2, 1, 0)
            
            batch_obs = (batch_abs[:obs_length] - nodes_obs_min) / (nodes_obs_max - nodes_obs_min)
            

            inputs_forward = batch_obs, seq_list[:obs_length], nei_list[:obs_length], nei_num[:obs_length], batch_pednum
                
            self.net.zero_grad()
            output = self.net.forward(inputs_forward, iftest=True)
            
            self.net.zero_grad()

            lossmask, num = getLossMask(output[:, :, :2], seq_list[obs_length], seq_list[obs_length:], using_cuda=self.args.using_cuda)
            
            nodes_obs_max = torch.cat([
                y_max.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
                x_max.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
            ]).permute(2, 1, 0)

            nodes_obs_min = torch.cat([
                y_min.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
                x_min.unsqueeze(1).repeat(1, pred_length).unsqueeze(0),
            ]).permute(2, 1, 0)
            
            # batch_pred = (batch_abs[obs_length:] - nodes_obs_min) / (nodes_obs_max - nodes_obs_min)
            output = output * (nodes_obs_max - nodes_obs_min) + nodes_obs_min
            error, error_cnt, final_error, final_error_cnt, fd, fd_cnt = L2forTestS(output.unsqueeze(0)[:, :, :, :2], batch_abs[obs_length:],
                                                                            lossmask, num_samples = 1)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            fd_epoch += fd
            fd_cnt_epoch += fd_cnt
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, fd_epoch / fd_cnt_epoch
