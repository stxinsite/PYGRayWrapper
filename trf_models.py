## Transformer-based network with edges attributes for parameters optimization and fine tuning
import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

## Transformer-based network with edges attributes
class TrfEdgeNet(torch.nn.Module):
    ## Default parameters correspond to the best architecture according to optimization
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, drop_p=0.3, heads=8,
                 concat=False, aggr="add"):
        super(TrfEdgeNet, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=interlayer_dim,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf2 = TransformerConv(in_channels=interlayer_dim, out_channels=interlayer_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.trf3 = TransformerConv(in_channels=interlayer_dim, out_channels=output_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.classifier = Linear(output_dim, num_classes)
        self.__activation = ReLU()
        self.__drop_p = drop_p
        
    def get_embeddings(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.trf1(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)
        return x    
    
    def get_classification(self, x):
        return self.classifier(x)

    def forward(self, data):
        
        embeddings = self.get_embeddings(data)
        classification = self.get_classification(embeddings)

        return classification