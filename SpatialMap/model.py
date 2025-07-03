import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, act=F.relu):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act
        
        if len(hidden_dims) >= 1:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x
    
    
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.fc(z)

    
class Decoder(nn.Module):
    def __init__(self, hidden_dims,input_dim,act=F.relu):
        super(Decoder, self).__init__()
        reversed_dims = hidden_dims[::-1]
        self.layers = nn.ModuleList()
        self.act = act
        
        if len(reversed_dims) > 1:
            for i in range(len(reversed_dims) - 1):
                self.layers.append(nn.Linear(reversed_dims[i], reversed_dims[i+1]))
            
        self.layers.append(nn.Linear(reversed_dims[-1], input_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x

    
class Pretrain_model(nn.Module):
    def __init__(self, input_dim, hidden_dims,num_classes):
        super(Pretrain_model, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.cls = Classifier(hidden_dims[-1], num_classes)
        self.decoder=Decoder(hidden_dims,input_dim)

    def forward(self, x):
        z = self.encoder(x)
        class_logits = self.cls(z)
        recon_x=self.decoder(z)
        return z, class_logits,recon_x
    

class Annotation_model(nn.Module):
    def __init__(self, in_features, embedding_dims, cls_num, act=F.relu):
        super(Annotation_model, self).__init__()
        self.in_features = in_features
        self.embedding_dims = embedding_dims
        self.act = act
        self.cls_num=cls_num
        
        self.g_encoder = nn.ModuleList()
        self.g_encoder.append(SAGEConv(in_features, embedding_dims[0]))

        for i in range(len(embedding_dims) - 1):
            self.g_encoder.append(SAGEConv(embedding_dims[i], embedding_dims[i+1]))

        self.cls2 = nn.Linear(embedding_dims[-1], cls_num)

    def forward(self, x, edge_index):
        
        for conv in self.g_encoder[:-1]:
            x = self.act(conv(x, edge_index))
            
        x = self.g_encoder[-1](x, edge_index)
        
        emb = self.act(x)
        
        clas = self.cls2(emb)
        
        return clas
    
    
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
            )
    
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)