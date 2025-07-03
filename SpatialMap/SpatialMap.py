import torch
import torch.optim as optim
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader

from .preprocess import *
from .utils import *
from .model import *



class SpatialMap():
    def __init__(self,
        sc_file,
        srt_file,
        device=torch.device('cpu'),
        lr1=1e-4,
        lr2=1e-4,
        epochs1=30,
        epochs2=100,
        seed=2024,
        Kn=3,
        gamma=2.0,
        hidden_dims=[512,256],
        embedding_dims=[128],
        batch_size1=2048,
        weight_dict=None,
        alpha=2.0,
        beta=0.3,
        num_parts=10,
        new=False,
        new_rate=0.03,
        lr3=1e-4,
        epochs3=100,
        ):
        '''

        Parameters
        ----------
        sc_file (path, required): Path of an AnnData object of reference single cell dataset.

        srt_file (path, required): Path of an AnnData object of target spatial transcriptomics dataset.

        device (string, optional): Device for runing SpatialMap model. The default is 'cpu'.

        lr1 (float, optional): Learning rate in the pretrain stage. The default is 1e-4.

        lr2 (float, optional): Learning rate in the annotation stage. The default is 1e-4.

        epochs1 (int, optional): Training epochs in the pretrain stage. The default is 30.

        epochs2 (int, optional): Training epochs in the annotation stage. The default is 100.

        seed (int, optional): Random seed to fix model initialization. The default is 2024.

        Kn (int, optional): The K used for constructed KNN spatial graph. The default is 3.

        gamma (float, optional): Control the relative loss for well-classified samples. The default is 2.0. 
        
        hidden_dims (list, optional): List of dimensions of latent vectors in Encoder and Decoder. The length is the number of layers. The default is [512,256].
        
        embedding_dims (list, optional): List of dimensions of latent vectors in GNN-Encoder. The length is the number of layers. The default is [128].
        
        batch_size1 (int, optional): Batch size in pretrain stage. The default is 2048.
        
        weight_dict(dict, optional): Specify the loss weight for each cell type during the pre-training process. The larger the weight, the more attention model pays to that type. It is recommended to reasonably increase the weight of rare cells to facilitate identification {'type1': 2, 'type2': 3, ... }. The default is None, that is, 1 for all types. 

        alpha (float, optional): Weight of reconstruction loss in pretrain stage. The default is 2.0. Suggested values: 10.0 for CosMx platform, 2.0 for other platforms.
        
        beta (float, optional): Ratio of pseudo-labeled cells to the total number of cells for each predicted cell type. The default is 0.3.
        
        num_parts (int, optional): The number of ClusterData partitions. The default is 10.
        
        
        ## new type discovery
        
        new (bool, optional): Whether to identify new cell type. The default is False.
            # Notice: Training GAN is unstable and highly dependent on hyperparameter choices, architecture design, and training strategies

        new_rate (float, optional): Ratio of pseudo-labeled new cell types over the total number of cells in the target data. Only used when 'new==True'. The default is 0.03.

        lr3 (float, optional): Learning rate for traning GAN model. Only used when 'new==True'. The default is 1e-4.
        
        epochs3 (int, optional): Training epochs for GAN model. Only used when 'new==True'. The default is 100.


        Return
        -------
        The cell type annotation of target data.

        '''
        self.sc_file=sc_file
        self.srt_file=srt_file
        self.device=device
        self.lr1=lr1
        self.lr2=lr2
        self.lr3=lr3
        self.epochs1=epochs1
        self.epochs2=epochs2
        self.epochs3=epochs3
        self.seed=seed
        self.Kn=Kn
        self.gamma=gamma
        self.hidden_dims=hidden_dims
        self.embedding_dims=embedding_dims
        self.batch_size1=batch_size1
        self.alpha=alpha
        self.beta=beta
        self.num_parts=num_parts
        self.new=new
        self.new_rate=new_rate
        self.weight_dict=weight_dict
        self.weight=None
        
        
        #preprocess and load datasets
        self.sc_x, self.sc_y, self.srt_x, self.srt_edges, self.cell_type_dict, self.inverse_dict = load_data(sc_file,srt_file,Kn=3)
        
        num_classes = len(self.cell_type_dict)
        self.num_classes=num_classes
        self.input_dim = self.sc_x.shape[1]
        
        
    def pretrain(self):
        #fix random seed
        setup_seed(self.seed)
        
        if self.weight_dict:
            self.weight = torch.tensor([self.weight_dict.get(self.inverse_dict[i], 1.0) for i in range(len(self.inverse_dict))],dtype=torch.float32)
    
        criterion = FocalLoss(gamma=self.gamma, weight=self.weight)
        
        self.model1 = Pretrain_model(self.input_dim, self.hidden_dims, self.num_classes)
        
        optimizer = Adam(self.model1.parameters(), lr=self.lr1)
        
        dataset = SingleCellDataset(self.sc_x, self.sc_y)

        dataloader = DataLoader(dataset, batch_size=self.batch_size1, shuffle=True)

        self.model1.to(self.device)
        

        for epoch in range(self.epochs1):
            self.model1.train()
            
            all_loss=0
            all_cls_loss=0
            all_rec_loss=0
            all_pre=[]
            all_label=[]
            
            for batch_idx,(x,y) in enumerate(dataloader):

                x = x.clone().detach().to(torch.float32).to(self.device)
                y = y.clone().detach().to(torch.long).to(self.device)

                z, class_logits, recon_x = self.model1(x)

                class_loss = criterion(class_logits, y)

                recon_loss = F.mse_loss(x, recon_x)*self.alpha

                loss=class_loss+recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_pre.append(torch.argmax(class_logits, dim=1).cpu().detach().numpy())
                all_label.append(y.cpu().detach().numpy())
                
                all_loss+=loss.item()
                all_cls_loss+=class_loss.item()
                all_rec_loss+=recon_loss.item()

            predictions = np.concatenate(all_pre)
            label=np.concatenate(all_label)
            acc=((predictions==label).sum()/len(predictions))

            print(f'Epoch {epoch+1}, Loss: {all_loss/(batch_idx+1):.4f},  CLSLoss: {all_cls_loss/(batch_idx+1):.4f}, RECLoss: {all_rec_loss/(batch_idx+1):.4f}, Train_acc: {acc:.4f}')
           
            
    def pretrain_GAN(self):
        
        setup_seed(self.seed)
        
        self.generator = Generator(self.input_dim).to(self.device)
        
        self.discriminator = Discriminator(self.input_dim).to(self.device)
        
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)
        
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr3, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr3, betas=(0.5, 0.999))
        
        bce_loss = nn.BCELoss()
        
        patience = 10
        cycle = 3
        best_loss=float('inf')
        alpha=1
        beta=1
        loss1=[]
        loss2=[]
        loss3=[]
        all_auroc=[]

        for epoch in range(self.epochs3):


            self.generator.train()
            self.discriminator.train()


            for _ in range(2):
                x = torch.tensor(self.sc_x).to(torch.float32).to(self.device)

                noise=torch.randn(self.sc_x.shape).to(self.device)
                x_noise = noise + 0.1 * torch.randn_like(noise).to(self.device)

                x_fake=self.generator(x_noise)

                dis_real=self.discriminator(x).view(-1)
                dis_fake=self.discriminator(x_fake).view(-1)
                
                loss_dis = bce_loss(dis_real, torch.ones(dis_real.shape).to(self.device))+bce_loss(dis_fake, torch.zeros(dis_fake.shape).to(self.device))
                
                optimizer_D.zero_grad()
                loss_dis.backward()
                optimizer_D.step()
                loss1.append(loss_dis.item())

            for _ in range(1):
                noise=torch.randn(self.sc_x.shape).to(self.device)
                x_noise = noise + 0.1 * torch.randn_like(noise).to(self.device)
                
                x_fake=self.generator(x_noise)
                dis_fake=self.discriminator(x_fake).view(-1)

                loss_g=bce_loss(dis_fake,torch.ones(dis_fake.shape).to(self.device))

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()
                loss2.append(loss_g.item())
                
#             if (epoch+1)%10==0:
#                 print(f'Epoch {epoch+1}, Loss D: {loss_dis.item():.4f}, Loss G: {loss_g.item():.4f}')
        print('finished')

        
        
    def pseudo_label(self):
       
        self.model1.eval()
        x = torch.tensor(self.srt_x, dtype=torch.float32).to(self.device)
        z, class_logits,_ = self.model1(x)
        
        predictions=torch.argmax(class_logits, dim=1).cpu().detach().numpy()
      
        class_logits=torch.nn.Softmax(dim=1)(class_logits).cpu().detach().numpy()
        
        # gain the biggest score and label of each sample
        max_probs = np.max(class_logits, axis=1)
        pred_labels = np.argmax(class_logits, axis=1)

        pseudo_label_indices = []

        unique_labels = np.unique(pred_labels)

        for label in unique_labels:

            label_indices = np.where(pred_labels == label)[0]

            label_probs = max_probs[label_indices]

            sorted_indices = np.argsort(label_probs)[::-1]
            cutoff_index = int(len(sorted_indices) * self.beta)
            selected_label_indices = label_indices[sorted_indices[:cutoff_index]]

            pseudo_label_indices.extend(selected_label_indices)

        pseudo_label_indices = np.array(pseudo_label_indices)
        
        if not self.new:
            
            self.pseudo_labels=np.full(len(self.srt_x), -1)
            self.pseudo_labels[pseudo_label_indices]=pred_labels[pseudo_label_indices]
            self.pseudo_mask=self.pseudo_labels>-1
            
        else:
            # select new type
            self.discriminator.eval()
            prob_known = self.discriminator(torch.tensor(self.srt_x).to(torch.float32).to(self.device)).flatten().cpu().detach()
            prob_new = -prob_known
            _, selected_new_indices = torch.topk(prob_new, int(self.new_rate*len(self.srt_x)))

            self.pseudo_labels=np.full(len(self.srt_x), -1)
            self.pseudo_labels[selected_new_indices]=self.num_classes
            self.pseudo_labels[pseudo_label_indices]=pred_labels[pseudo_label_indices]
            self.pseudo_mask=self.pseudo_labels>-1
        
        return self.pseudo_labels, self.pseudo_mask
        
        
    def annotate(self):
        #fix random seed
        setup_seed(self.seed)
        
        target_data = Data(x=torch.FloatTensor(self.srt_x), edge_index=torch.LongTensor(self.srt_edges).T, y=torch.LongTensor(self.pseudo_labels))
        target_data.pseudo_mask=torch.BoolTensor(self.pseudo_mask)
        target_data.index=torch.LongTensor(list(range(len(self.srt_x))))

        target_dataset = ClusterData(target_data, num_parts=self.num_parts, recursive=False)
        target_loader = ClusterLoader(target_dataset, batch_size=1, shuffle=True, num_workers=0)
        
        if not self.new:
            self.model2=Annotation_model(self.input_dim, self.embedding_dims, self.num_classes).to(self.device)
        else:
            self.model2=Annotation_model(self.input_dim, self.embedding_dims, self.num_classes+1).to(self.device)

        loss_CSL = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model2.parameters(),lr=self.lr2)

        for epoch in range(self.epochs2): 
            self.model2.train()
            
            sum_loss=0
            
            for batch_idx, data in enumerate(target_loader):
                data=data.to(self.device)

                clas= self.model2(data.x, data.edge_index)

                loss_cls = criterion(clas[data.pseudo_mask], data.y[data.pseudo_mask])


                optimizer.zero_grad()
                loss_cls.backward() 
                optimizer.step()

                sum_loss += loss_cls.item()

            self.model2.eval()
            
            all_clas=[]
            all_index=[]
            
            for batch_idx, data in enumerate(target_loader):
                data=data.to(self.device)

                clas= self.model2(data.x, data.edge_index)
                
                all_clas.append(clas.cpu().detach().numpy())
                all_index.append(data.index.cpu().detach().numpy())
                
            all_clas=np.concatenate(all_clas)
            annotation = all_clas.argmax(axis=1)
            
            print(f'Epoch {epoch+1}, Loss: {sum_loss / (batch_idx + 1):.4f}')
            
        all_index=np.concatenate(all_index)
        sorted_indices = np.argsort(all_index)
        
        if not self.new:
            self.annotation=[self.inverse_dict[p] for p in annotation[sorted_indices]]
        else:
            self.inverse_dict[self.num_classes]='new'
            self.annotation=[self.inverse_dict[p] for p in annotation[sorted_indices]]
        
        return self.annotation






