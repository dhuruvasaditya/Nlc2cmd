from torch.nn import Module
from torch import Tensor

class OmniBash(Module):
    def __init__(self,Transformer,Device):
        self.Trans = Transformer # Transformer Model
        self.Device = Device
    def forward(self,Input,Label):
        Attn_Mask = Labels
        Output = self.Trans(Input,attention_mask=Attn_Mask)