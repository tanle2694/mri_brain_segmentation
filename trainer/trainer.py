import modeling.metric as metrics
import torch
from numpy import inf

from torchvision.utils import make_grid
from utils.util import MetricTracker
from logger.visualization import TensorboardWriter
import time
import numpy as np

class Trainer():
    def __init__(self, model, criterion, metrics_name, optimizer, train_loader, logger, log_dir, nb_epochs, save_dir,
                 device="cuda:0", log_step=10, start_epoch=0, enable_tensorboard=True, valid_loader=None, lr_scheduler=None,
                 monitor="max miou", early_stop=10, save_epoch_period=1, resume=""):
        self.model = model
        self.criterion = criterion
        self.metrics_name = metrics_name
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.len_epoch = len(self.train_loader)
        self.do_validation = (self.valid_loader is not None)
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.epochs = nb_epochs
        self.start_epoch = start_epoch + 1

        self.logger = logger
        self.device = device
        self.save_period = save_epoch_period

        self.writer = TensorboardWriter(log_dir, self.logger, enable_tensorboard)
        self.train_metrics = MetricTracker('loss', *self.metrics_name, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *self.metrics_name, writer=self.writer)
        self.checkpoint_dir = save_dir
        if monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = early_stop
        if resume != "":
            self._resume_checkpoint(resume_path=resume)
        self.model.to(self.device)

    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            self.logger.info('    {:15s}: {}'.format(str("mnt best"), self.mnt_best))
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                if (not_improved_count > self.early_stop) and (self.early_stop > 0):
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, best)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        start_time = time.time()

        for batch_idx, sample in enumerate(self.train_loader):
            data = sample['image']
            target = sample['mask']
            data, target = data.to(self.device), target.to(self.device)
            current_lr = self.lr_scheduler(self.optimizer, batch_idx, epoch)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met_name in self.metrics_name:
                self.train_metrics.update(met_name, getattr(metrics, met_name)(output, target))
            if batch_idx % self.log_step == 0:
                time_to_run = time.time() - start_time
                start_time = time.time()
                speed = self.log_step / time_to_run
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} LR: {:.6f}  Speed: {:.4f}iters/s' \
                                  .format(epoch, self._progress(batch_idx), loss.item(), current_lr, speed))
                for met_name in self.metrics_name:
                    self.writer.add_scalar(met_name, self.train_metrics.avg(met_name))
                self.writer.add_scalar('loss', self.train_metrics.avg('loss'))
                self.writer.add_scalar("lr", current_lr)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            assert batch_idx <= self.len_epoch
        log = self.train_metrics.result()
        if self.do_validation:
            print("Start validation")
            val_log, iou_classes = self._valid_epoch(epoch)

            log.update(**{'val_' + k: v for k, v in val_log.items()})
            for key, value in iou_classes.items():
                log.update({key: value})
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        iou_tracker = metrics.IoU(2)
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_loader):
                data = sample['image']
                target = sample['mask']
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                output = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(output, axis=1)
                for i in range(len(target)):
                    iou_tracker.add_batch(target[i], pred[i])
        iou_classes = iou_tracker.get_iou()
        for key, value in iou_classes.items():
            self.writer.add_scalar(key, value)
        self.writer.add_scalar('loss', self.valid_metrics.avg('loss'))

        for met_name in self.metrics_name:
            self.writer.add_scalar(met_name, self.valid_metrics.avg(met_name))

        # for name, p in self.model.named_parameters():
        #     print(name, p)
        #     self.writer.add_histogram(name, p.cpu().data.numpy(), bins='auto')
        #
        return self.valid_metrics.result(), iou_classes


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            # 'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))