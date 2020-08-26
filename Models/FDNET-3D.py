import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
class Fdnet3D(nn.Module):

    def __init__(self,inputLength,fdfilter, xLen):
        super(Fdnet3D, self).__init__()
         
        self.xLen = xLen
        
        self.fdfilter = fdfilter
        
        self.lossFunction = nn.MSELoss()
        
        self.inputLength = inputLength
        
        self.convX1 = nn.Conv2d(1, self.fdfilter, (1,3),padding=(0, 0),bias=False) 
        self.convX1L = nn.Conv2d(1, self.fdfilter, (1,2),padding=(0, 0),bias=False ) 
        self.convX1R = nn.Conv2d(1, self.fdfilter, (1,2),padding=(0, 0),bias=False ) 

        self.convY1 = nn.Conv2d(1, self.fdfilter, (3,1),padding=(0, 0),bias=False) 
        self.convY1L = nn.Conv2d(1, self.fdfilter, (2,1),padding=(0, 0),bias=False ) 
        self.convY1R = nn.Conv2d(1, self.fdfilter, (2,1),padding=(0, 0),bias=False ) 
        
        self.convX2 = nn.Conv2d(self.fdfilter, self.fdfilter, (1,3),padding=(0, 0),bias=False) 
        self.convX2L = nn.Conv2d(self.fdfilter, self.fdfilter, (1,2),padding=(0, 0),bias=False ) 
        self.convX2R = nn.Conv2d(self.fdfilter, self.fdfilter, (1,2),padding=(0, 0),bias=False ) 

        self.convY2 = nn.Conv2d(self.fdfilter, self.fdfilter, (3,1),padding=(0, 0),bias=False) 
        self.convY2L = nn.Conv2d(self.fdfilter, self.fdfilter, (2,1),padding=(0, 0),bias=False ) 
        self.convY2R = nn.Conv2d(self.fdfilter, self.fdfilter, (2,1),padding=(0, 0),bias=False ) 
        
        self.fc1   = nn.Linear(self.fdfilter*2*2, 1, bias=False)  # (first, 2nd )*(x,y)
        
    def numberOfParameters(self):
        return np.sum([  w.numel() for w  in self.parameters()])

    def forward(self, xInput, fdblock): 
        
        BS = xInput.shape[0]
        beforeDNN_x = xInput.contiguous().view([-1,1,self.xLen,self.xLen]) 
        

        for k in range(fdblock): 
            beforeDNN_xDIFX = self.convX1(beforeDNN_x)
            beforeDNN_xDIFXL = self.convX1L(beforeDNN_x[:,:,:,0:2])
            beforeDNN_xDIFXR = self.convX1R(beforeDNN_x[:,:,:,-2:])

            beforeDNN_yDIFY = self.convY1(beforeDNN_x)
            beforeDNN_yDIFYL = self.convY1L(beforeDNN_x[:,:,0:2,:])
            beforeDNN_yDIFYR = self.convY1R(beforeDNN_x[:,:,-2:,:])
            
            
            beforeDNN_xDIF = torch.cat([beforeDNN_xDIFXL,beforeDNN_xDIFX,beforeDNN_xDIFXR],3)
            beforeDNN_yDIF = torch.cat([beforeDNN_yDIFYL,beforeDNN_yDIFY,beforeDNN_yDIFYR],2)

            
            
            beforeDNN_xDIFX2 = self.convX2(beforeDNN_xDIF)
            beforeDNN_xDIFXL2 = self.convX2L(beforeDNN_xDIF[:,:,:,0:2])
            beforeDNN_xDIFXR2 = self.convX2R(beforeDNN_xDIF[:,:,:,-2:])

            beforeDNN_yDIFY2 = self.convY2(beforeDNN_yDIF)
            beforeDNN_yDIFYL2 = self.convY2L(beforeDNN_yDIF[:,:,0:2,:])
            beforeDNN_yDIFYR2 = self.convY2R(beforeDNN_yDIF[:,:,-2:,:])
            

            beforeDNN_xDIF2 = torch.cat([beforeDNN_xDIFXL2,beforeDNN_xDIFX2,beforeDNN_xDIFXR2],3)
            beforeDNN_yDIF2 = torch.cat([beforeDNN_yDIFYL2,beforeDNN_yDIFY2,beforeDNN_yDIFYR2],2)
            
            
            
            
            combineAllDerivatives=torch.cat([beforeDNN_xDIF, 
                                             beforeDNN_yDIF,
                                             beforeDNN_xDIF2,
                                             beforeDNN_yDIF2,]
                                             ,1).transpose(1,3)
            
            deviation = (self.fc1(combineAllDerivatives)).transpose(1,3) # .contiguous()
            
            beforeDNN_x = beforeDNN_x + deviation
   
        return beforeDNN_x.view([BS,-1,self.xLen,self.xLen])