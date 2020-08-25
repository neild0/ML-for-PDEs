import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
class fdnet(nn.Module):

    def __init__(self,inputLength,fdfilter, xLen):
        super(fdnet, self).__init__()
        
        self.xLen = xLen
        
        self.fdfilter = fdfilter
        
        self.lossFunction = nn.MSELoss()
        
        self.inputLength = inputLength
        
        self.conv1 = nn.Conv2d(1, self.fdfilter, (3,3),padding=(0, 0),bias=False) 
        self.conv1L = nn.Conv2d(1, self.fdfilter, (2,2),padding=(0, 0),bias=False ) 
        self.conv1R = nn.Conv2d(1, self.fdfilter, (2,2),padding=(0, 0),bias=False ) 
        
        self.conv2 = nn.Conv2d(self.fdfilter, self.fdfilter, (3,3),padding=(0, 0),bias=False )
        self.conv2L = nn.Conv2d(self.fdfilter, self.fdfilter, (2,2),padding=(0, 0),bias=False )
        self.conv2R = nn.Conv2d(self.fdfilter, self.fdfilter, (2,2),padding=(0, 0),bias=False )
    
        self.fc1   = nn.Linear(self.fdfilter*2, 1, bias=False)
        
    def numberOfParameters(self):
        return np.sum([  w.numel() for w  in self.parameters()])

    def forward(self, xInput, fdblock): 
        beforeDNN_x = xInput.contiguous().view([-1,1,self.inputLength,self.xLen]) 
        
        for k in range(fdblock): 
            beforeDNN_xDIF = self.conv1(beforeDNN_x)
            beforeDNN_xDIFL = self.conv1L(beforeDNN_x[:,:,:,0:2])
            beforeDNN_xDIFR = self.conv1R(beforeDNN_x[:,:,:,-2:])
            
            beforeDNN_xDIF = torch.cat([beforeDNN_xDIFL,beforeDNN_xDIF,beforeDNN_xDIFR],3)

            beforeDNN_xDIF2 = self.conv2(beforeDNN_xDIF)
            beforeDNN_xDIF2L = self.conv2L(beforeDNN_xDIF[:,:,:,0:2])
            beforeDNN_xDIF2R = self.conv2R(beforeDNN_xDIF[:,:,:,-2:])

            beforeDNN_xDIF2 = torch.cat([beforeDNN_xDIF2L,beforeDNN_xDIF2,beforeDNN_xDIF2R],3)

            beforeDNN_xDIFCat = torch.cat([beforeDNN_xDIF.transpose(2,3),beforeDNN_xDIF2.transpose(2,3)],3)
                
            beforeDNN_xDIFCat = beforeDNN_xDIFCat.transpose(1,2) .contiguous()
            beforeDNN_xDIFCat = (beforeDNN_xDIFCat).view([-1,self.xLen,self.fdfilter*2])
            
            deviation = (self.fc1(beforeDNN_xDIFCat)).transpose(1,2)
            deviation = deviation.unsqueeze(1)
            
            beforeDNN_x = beforeDNN_x + deviation
   
        return beforeDNN_x.squeeze(1)
    