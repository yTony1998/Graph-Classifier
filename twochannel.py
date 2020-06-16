                                                                                                                                                                         
import torch                                                                                                                                                             
import torch.nn.functional as F                                                                                                                                          
from torch_geometric.datasets import TUDataset                                                                                                                           
from torch_geometric.data import DataLoader                                                                                                                              
from torch_geometric.nn import GraphConv, TopKPooling                                                                                                                    
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp                                                                                           
from torch_geometric.nn import GCNConv,SAGEConv,GATConv                                                                                                                  
                                                                                                                                                                         
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')                                                                                          
dataset = TUDataset(root="/content/sample_data", name='ENZYMES')                                                                                                         
dataset = dataset.shuffle()                                                                                                                                              
n = len(dataset) // 5                                                                                                                                                   
test_dataset = dataset[:n]                                                                                                                                               
train_dataset = dataset[n:]                                                                                                                                              
test_loader = DataLoader(test_dataset, batch_size=120)                                                                                                                    
train_loader = DataLoader(train_dataset, batch_size=120)                                                                                                                  
                                                                                                                                                                         
                                                                                                                                                                         
                                                                                                                                                                         
class Net(torch.nn.Module):                                                                                                                           
    def __init__(self):                                                                                                                                                  
        super(Net, self).__init__()                                                                                                                                      
                                                                                                                                                                         
        self.conv10 = GATConv(dataset.num_features, 128)                                                                                                                 
        self.pool10 = TopKPooling(128, ratio=0.8)                                                                                                                         
        self.conv20 = GATConv(128, 128)                                                                                                                                  
        self.pool20 = TopKPooling(128, ratio=0.8) 
        self.conv30 = GATConv(128, 128)                                                                                                                                  
        self.pool30 = TopKPooling(128, ratio=0.8)                                                                                                                          
        self.conv11 = GraphConv(dataset.num_features, 128)                                                                                                                                  
        self.pool11 = TopKPooling(128, ratio=0.8)                                                                                                                         
        self.conv21 = GraphConv(128,128)                                                                                                                                  
        self.pool21 = TopKPooling(128,ratio=0.8) 
        self.conv31 = GraphConv(128,128)                                                                                                                                  
        self.pool31 = TopKPooling(128,ratio=0.8)                                                                                                                        
                                                                                                                                                                         
        self.lin1 = torch.nn.Linear(256, 128)                                                                                                                            
        self.lin2 = torch.nn.Linear(128, 64)                                                                                                                             
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)                                                                                                             
                                                                                                                                                                         
    def forward(self, data):                                                                                                                                             
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge,batch1=edge_index,batch
        # print(x.shape)                                                                                                     
        # y = x
        # print(y.shape)
        # x1 = x                                                                                                                                                             
        x0 = F.relu(self.conv10(x, edge_index))  
        # print(x0.shape)  
        # print(1)                                                                                                                        
        # x0 = gap(x0, batch)                                                                                           
        x0, edge_index, _, batch, _, _ = self.pool20(x0, edge_index, None, batch)
        x1 = torch.cat([gmp(x0, batch), gap(x0, batch)], dim=1) 
        # print(1)                                                                                                                            
        x0 = F.relu(self.conv20(x0, edge_index))                                                                                                                            
        x0, edge_index, _, batch, _, _ = self.pool20(x0, edge_index, None, batch)
        x2 = torch.cat([gmp(x0, batch), gap(x0, batch)], dim=1)
        # print(2)
        x0 = F.relu(self.conv30(x0, edge_index))                                                                                                                            
        x0, edge_index, _, batch, _, _ = self.pool30(x0, edge_index, None, batch)
        x3 = torch.cat([gmp(x0, batch), gap(x0, batch)], dim=1)
        # print(3)
        # print(x0.shape)
        # print(2)
        # x0 = gap(x0, batch) 

        z = F.relu(self.conv11(x, edge))                                                                                                                            
        z, edge, _, batch1, _, _ = self.pool11(z, edge, None, batch1) 
        z1 = torch.cat([gmp(z, batch1), gap(z, batch1)], dim=1) 
        # print(4)                                                                                         
        # x1 = gap(x1, batch)                                                                                                       
        z = F.relu(self.conv21(z, edge))                                                                                                                            
        z, edge, _, batch1, _, _ = self.pool21(z, edge, None, batch1)
        z2 = torch.cat([gmp(z, batch1), gap(z, batch1)], dim=1) 
        # print(5)   
        z = F.relu(self.conv31(z, edge))                                                                                                                            
        z, edge, _, batch1, _, _ = self.pool31(z, edge, None, batch1)
        z3 = torch.cat([gmp(z, batch1), gap(z, batch1)], dim=1) 
        # print(6)                                                                                       
        # x1 = gap(x1, batch)                                                                                                
        # print(3)
        x=x1+x2+x3+z1+z2+z3
        # print(7)                                                                                                                                                                 
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)                                                                                                                                                 
        # print(8)                                                                                                                                                         
        x = F.relu(self.lin1(x))                                                                                                                                         
        x = F.dropout(x, p=0.5, training=self.training)                                                                                                                  
        x = F.leaky_relu(self.lin2(x))                                                                                                                                         
        x = F.log_softmax(self.lin3(x), dim=-1)                                                                                                                          
                                                                                                                                                                         
        return x                                                                                                                                                         
device = torch.cuda.set_device(0)                                                                                                    
model = Net().to(device)                                                                                                                                                 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)                                                                                                              
                                                                                                                                                                         
                                                                                                                                                                         
def train(epoch):                                                                                                                                                        
    model.train()                                                                                                                                                        
                                                                                                                                                                         
    loss_all = 0                                                                                                                                                         
    for data in train_loader:                                                                                                                                            
        data = data.to(device)                                                                                                                                           
        optimizer.zero_grad()                                                                                                                                            
        output = model(data)                                                                                                                                             
        loss = F.nll_loss(output, data.y)                                                                                                                                
        loss.backward()                                                                                                                                                  
        loss_all += data.num_graphs * loss.item()                                                                                                                        
        optimizer.step()                                                                                                                                                 
    return loss_all / len(train_dataset)                                                                                                                                 
                                                                                                                                                                         
                                                                                                                                                                         
def test(loader):                                                                                                                                                        
    model.eval()                                                                                                                                                         
                                                                                                                                                                         
    correct = 0                                                                                                                                                          
    for data in loader:                                                                                                                                                  
        data = data.to(device)                                                                                                                                           
        pred = model(data).max(dim=1)[1]                                                                                                                                 
        correct += pred.eq(data.y).sum().item()                                                                                                                          
    return correct / len(loader.dataset)                                                                                                                                 
                                                                                                                                                                         
epoch_loss=[]                                                                                                                                                            
all_train_acc=[]                                                                                                                                                         
all_test_acc=[]                                                                                                                                                          
for epoch in range(0, 10001):                                                                                                                                            
    loss = train(epoch)                                                                                                                                                  
    train_acc = test(train_loader)                                                                                                                                       
    test_acc = test(test_loader)                                                                                                                                         
    state={'net':model.state_dict(),"optimizer":optimizer.state_dict(),'epoch_loss':epoch_loss,"all_test_acc":all_test_acc,"all_train_acc":all_train_acc}                
    if epoch%100==0:                                                                                                                                                    
        torch.save(state,"TwoChannel"+str(epoch)+".pth")                                                                                                                     
    epoch_loss.append(loss)                                                                                                                                              
    all_test_acc.append(test_acc)                                                                                                                                        
    all_train_acc.append(train_acc)                                                                                                                                      
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.                                                                                            
          format(epoch, loss, train_acc, test_acc))                                                                                                                      
# Epoch: 044, Loss: 1.60927, Train Acc: 0.36296, Test Acc: 0.40000                                                                                                       