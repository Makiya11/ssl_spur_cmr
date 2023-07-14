"""
Implementations of MoCo in PyTorch
Momentum Contrast for Unsupervised Visual Representation Learning:
    https://arxiv.org/pdf/1911.05722.pdf
Acknowledgments:
    https://github.com/Lightning-AI/lightning-bolts
    https://github.com/facebookresearch/moco
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F

class Moco_v2(nn.Module):
    def __init__(self, encoder,emb_dim=128,num_negatives=65536,
        encoder_momentum=0.999, softmax_temperature=0.07,learning_rate=0.03,
        momentum=0.9, weight_decay=1e-4, use_mlp=False,):
        """
        Args:
            encoder: encoder model
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            use_mlp: add an mlp to the encoders
        """
        super().__init__()
        self.num_negatives = num_negatives
        self.softmax_temperature = softmax_temperature
        self.encoder_momentum = encoder_momentum
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.encoder.classifier.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.encoder.classifier)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.encoder.classifier)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # # create the validation queue
        # self.register_buffer("val_queue", torch.randn(emb_dim, num_negatives))
        # self.val_queue = nn.functional.normalize(self.val_queue, dim=0)

        # self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.optimizer = self.configure_optimizers()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        if self.num_negatives % batch_size != 0:
            breakpoint()
        assert self.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.num_negatives  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """
        self._momentum_update_key_encoder()

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            # img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
 
        self._dequeue_and_enqueue(k, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(logits.float(), labels.long())
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        return loss, log


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.encoder_q.parameters()),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        return optimizer
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        # return [optimizer], [scheduler]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
