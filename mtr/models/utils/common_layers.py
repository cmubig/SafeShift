# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self, config: dict, device: str = "cuda:0", softmax: bool = False
    ) -> None:
        """ Implments a simple MLP with ReLU activations. 
        Inputs:
        -------
        config[dict]: network configuration parameters.
        device[str]: device used by the module. 
        """
        self._name = self.__class__.__name__
        super(MLP, self).__init__()
        
        self._config = config
        self.device = device
        
        self.dropout = self.config.dropout
        self.layer_norm = 'layer_norm' in self.config and self.config.layer_norm
        
        # ----------------------------------------------------------------------
        # Network architecture 
        feats = [config.in_size, *config.hidden_size, config.out_size]
        mlp = []
        mlp_norm = []
        for i in range(len(feats)-1):
            mlp.append(
                nn.Linear(in_features=feats[i], out_features=feats[i+1])
            )
            mlp_norm.append(
                nn.LayerNorm(feats[i+1])
            )
            
        if softmax:
            mlp.append(nn.Softmax(dim=-1))
            
        self.net = nn.ModuleList(mlp)
        self.net_norm = nn.ModuleList(mlp_norm) if self.layer_norm else None
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
    
    def forward(self, x: torch.tensor, training=True) -> torch.tensor:
        """ Forward propagation of x.
        Inputs:
        -------
        x[torch.tensor(batch_size, input_size)]: input tensor
            
        Outputs:
        -------
        x[torch.tensor(batch_size, output_size)]: output tensor
        """ 
        for i in range(len(self.net)-1):
            x = self.net[i](x)
            if self.net_norm:
                x = self.net_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.net[-1](x)
        return x

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

