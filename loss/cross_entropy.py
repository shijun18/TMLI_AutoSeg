import torch


class CrossentropyLoss(torch.nn.CrossEntropyLoss):

    def forward(self, inp, target):
        if target.size()[1] > 1:
            target = torch.argmax(target,1)
        target = target.long()
        num_classes = inp.size()[1]

        inp = inp.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1,)

        return super(CrossentropyLoss, self).forward(inp, target)


