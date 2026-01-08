import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class HyperbolicScheduler:
    """
    Scheduler for hyperbolic loss weight warmup.

    The hyperbolic loss is gradually introduced after a warmup period
    to allow the model to first learn basic representations before
    being constrained by the hyperbolic geometry.
    """

    def __init__(self, warmup_epochs=20, max_weight=1.0):
        """
        Initialize hyperbolic scheduler.

        Args:
            warmup_epochs: Number of epochs before hyperbolic loss kicks in
            max_weight: Maximum weight for hyperbolic loss
        """
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight

    def get_weight(self, epoch):
        """
        Get hyperbolic loss weight for current epoch.

        Returns 0 during warmup, then linearly increases to max_weight.

        Args:
            epoch: Current epoch number

        Returns:
            Weight for hyperbolic loss
        """
        if epoch <= self.warmup_epochs:
            return 0.0
        progress = (epoch - self.warmup_epochs) / float(self.warmup_epochs)
        return min(progress * self.max_weight, self.max_weight)


def get_scheduler(optimizer, args):

    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.9 * args['base']['n_epochs'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=0.1 * args['base']['n_epochs'], after_scheduler=scheduler_steplr)

    return scheduler_warmup


def get_hyperbolic_scheduler(args):
    """
    Create hyperbolic loss weight scheduler.

    Args:
        args: Configuration dictionary

    Returns:
        HyperbolicScheduler instance
    """
    return HyperbolicScheduler(
        warmup_epochs=args['base'].get('hyp_warmup_epochs', 30),
        max_weight=args['base'].get('beta_max', 0.1)
    )