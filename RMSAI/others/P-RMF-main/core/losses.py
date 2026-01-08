from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.kl = args['base']['kl']
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss()

    def forward(self, out, label):
        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out[
            'complete_feats'] is not None else 0
        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])
        l_kl = out["kl_loss"]
        loss = self.gamma * l_rec + self.sigma * l_sp + self.kl * l_kl

        return {'loss': loss, 'l_sp': l_sp, 'l_rec': l_rec, 'l_kl': l_kl}
