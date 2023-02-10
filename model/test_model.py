from pickle import TRUE
import sys
sys.path.append('..')
from model.sfnet import sfnet

if __name__ == '__main__':

    from torchsummary import summary
    import torch
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    sfnet
    net = sfnet('sfnet','resnet18',in_channels=1,classes=2)
    # net = sfnet('sfnet','simplenet',in_channels=1,classes=2)
    # net = sfnet('sfnet','swin_transformer',in_channels=1,classes=2)

    summary(net.cuda(),input_size=(1,512,512),batch_size=1,device='cuda')
    
    
    net = net.cuda()
    net.eval()
    input = torch.randn((1,1,512,512)).cuda()
    with torch.no_grad():
        output = net(input)
    print(output.size())
    
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))