
class Poly_Scheduler(object):

    def __init__(self, base_lr, num_epochs, iters_each_epoch):
        self.lr = base_lr
        self.number_all_step = num_epochs * iters_each_epoch
        self.iters_each_epoch = iters_each_epoch

    def __call__(self, optimizer, i, epoch):
        current_iter = epoch * self.iters_each_epoch + i
        lr = self.lr * pow((1 - 1.0 * current_iter / self.number_all_step), 0.9)
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr
