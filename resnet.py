                                                                                                                                                                         
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
                                                                                                                                                                         

        self.conv1 = GraphConv(dataset.num_features, 128)#num_features表示节点的特征数，为3
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.conv4 = GraphConv(128, 128)



        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)                                                                                                 
                                                                                                                                                                         
    def forward(self, data):                                                                                                                                             
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.relu(self.conv1(x, edge_index))
        # print(x1.shape)
        x2 = F.relu(self.conv2(x1, edge_index))
        # print(x2.shape)
        
        x3 = F.relu(self.conv3(x2, edge_index))


        x3=x3+x1
        x4 = F.relu(self.conv3(x3, edge_index))
        # print(x3.shape)
        x = x4
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.lin1(x))
        # print(4)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        # print(4)
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
for epoch in range(0, 601):                                                                                                                                            
    loss = train(epoch)                                                                                                                                                  
    train_acc = test(train_loader)                                                                                                                                       
    test_acc = test(test_loader)                                                                                                                                         
    state={'net':model.state_dict(),"optimizer":optimizer.state_dict(),'epoch_loss':epoch_loss,"all_test_acc":all_test_acc,"all_train_acc":all_train_acc}                
    if epoch%50==0:                                                                                                                                                    
        torch.save(state,"resnet"+str(epoch)+".pth")                                                                                                                     
    epoch_loss.append(loss)                                                                                                                                              
    all_test_acc.append(test_acc)                                                                                                                                        
    all_train_acc.append(train_acc)                                                                                                                                      
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.                                                                                            
          format(epoch, loss, train_acc, test_acc))                                                                                                                      
# Epoch: 044, Loss: 1.60927, Train Acc: 0.36296, Test Acc: 0.40000                                                                                                       