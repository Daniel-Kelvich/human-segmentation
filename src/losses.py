class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
​
    @staticmethod
    def _lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
​
    @staticmethod
    def _flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels
​
    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
        return loss
​
    def forward(self, y_pred, y_true):
        """
        Binary Lovasz hinge loss
            y_pred: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
            y_true: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        """
        return self._lovasz_hinge_flat(*self._flatten_binary_scores(y_pred, y_true))
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
​
    def _focal_loss(self, p, q, gamma):
        return p * torch.log(q + self.eps) * (1 - q + self.eps) ** gamma
​
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        f_loss = self._focal_loss(y_true, y_pred, self.gamma) + self._focal_loss(1 - y_true, 1 - y_pred, self.gamma)
        return -f_loss.mean()

    
class IoULoss(nn.Module):
    """ IoU loss
    """
​
    def __init__(self):
        super(IoULoss, self).__init__()
​
    @staticmethod
    def iou_metric(y_pred, y_true):
        _EPSILON = 1e-6
        op_sum = lambda x: x.sum(2).sum(2)
        loss = (op_sum(y_true * y_pred) + _EPSILON) / (
                op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON)
​
        loss = torch.mean(loss)
        return loss
​
    def forward(self, y_pred, y_true):
        """ Compute IoU loss
        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): target values
        """
        return 1 - self.iou_metric(torch.sigmoid(y_pred), y_true)