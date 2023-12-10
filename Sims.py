from Settings import *
from Utils import *

class Client_Sim:
    def __init__(self, ALoader, PLoader, Model, Lr, wdecay, epoch=2, fixlr=False, nclass=10):
        self.TrainData = cp.deepcopy(ALoader)
        self.PTrainData = cp.deepcopy(PLoader)
        self.CXs = None
        self.CYs = None
        self.NClass = nclass
        self.Model = cp.deepcopy(Model)
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Round = 0
        self.LR = Lr
        self.FixLR = fixlr
        self.gradnorm = 0
        self.trainloss = 0
        
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=self.Wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.loss_fn = nn.CrossEntropyLoss()

    def reload_data(self, loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)

    def updateLR(self, lr):
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=lr, momentum=0.9, weight_decay=self.Wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR

    def selftrain(self):
        self.Round += 1
        
        TLoader = cp.deepcopy(self.TrainData)
        self.gradnorm = 0
        self.trainloss = 0
        SLoss = []
        GNorm = []
        self.Model.train()
        for r in range(self.Epoch):
            sum_loss = 0
            grad_norm = 0
            C = 0
            for batch_id, (inputs, targets) in enumerate(TLoader):
                C = C + 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.Model(inputs)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                temp_norm = 0
                c = 0 
                for parms in self.Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                    c += 1
                if grad_norm == 0:
                    grad_norm = temp_norm
                else:
                    grad_norm = grad_norm + temp_norm
            SLoss.append(sum_loss/C)
            GNorm.append(grad_norm)
        
        self.trainloss = np.mean(SLoss)
        Lrnow = self.getLR()
        self.gradnorm = np.sum(GNorm) * Lrnow
        self.scheduler.step()  
    
    def evaluate(self, loader=None, max_samples=100000):
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        if loader == None:
            loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                iters += 1
                if samples >= max_samples:
                    break
        return correct / samples, loss / iters

    def fim(self,loader=None,nout=10):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)
        self.Model.eval()
        Ts = []
        K = 50000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break
        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=100, shuffle=False)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=nout,
            variant="classif_logits",
            device="cuda"
        )
        Tr = F_Diag.trace().item()
        return Tr



class Server_Sim:
    def __init__(self, TrainLoader, TestLoader, ValidLoader, DifLoader, Model, Lr, wdecay=0, Fixlr=False, epoch=2):
        self.TrainData = cp.deepcopy(TrainLoader)
        self.TestData = cp.deepcopy(TestLoader)
        self.ValidData = cp.deepcopy(ValidLoader)
        self.DifData = cp.deepcopy(DifLoader)
        self.Gamma = loadGamma()
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Model = cp.deepcopy(Model)
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=self.Gamma)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLr = Fixlr
        self.BackModels = []
        self.RecvParas = []
        self.RecvLens = []
        self.aggAFA = AFA()
        self.KeepParas = []
        self.KeepGParam = None

    def reload_data(self, loader):
        self.TestData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR
    
    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)

    def avgParas(self, Paras, Lens):
        Res = cp.deepcopy(Paras[0])
        Sum = np.sum(Lens)
        for ky in Res.keys():
            Mparas = 0
            for i in range(len(Paras)):
                Pi = Lens[i] / Sum
                Mparas += Paras[i][ky] * Pi
            Res[ky] = Mparas
        return Res

    def aggParas(self, aggmethod, uids, attacknum=0):
        UIDs = uids
        frac = attacknum
        GParas = None
        if aggmethod == "MKrum":
            num = max(1, len(self.RecvParas) - frac - 2)
            GParas,_ = AggMKrum(self.RecvParas, frac, num)
        if aggmethod == "TrimMean":
            GParas,_ = AggTrimMean(self.RecvParas, frac)
        if aggmethod == "TrimMed":
            GParas,_ = AggTrimMed(self.RecvParas, frac)
        if aggmethod == "Bulyan":
            num = max(1, len(self.RecvParas) - frac)
            GParas,_ = AggBulyan(self.RecvParas, frac, num)
        if aggmethod == "AFA":
            RPara = self.getParas()
            GParas,_ = self.aggAFA.AggParas(UIDs,RPara,self.RecvParas,self.RecvLens)
        if aggmethod == "FLTrust":
            RPara = self.getParas()
            PLens = []
            PPras = []
            Num = int(len(self.RecvParas) * 0.5)
            for i in range(Num):
                PLens.append(1)
                PPras.append(self.RecvParas[-1-i])
            PurePara = wavgParas(PPras,PLens)
            GParas,_ = AggTrust(PurePara,UIDs,RPara,self.RecvParas,self.RecvLens)
        if aggmethod == "Union":
            RPara = self.getParas()
            NModel = cp.deepcopy(self.Model)
            VData = cp.deepcopy(self.TestData)
            GParas = AggUnion(RPara,self.RecvParas,self.RecvLens,VData,NModel,frac,UIDs)
        self.updateParas(GParas)
        self.RecvParas = []
        self.RecvLens = []
        if self.FixLr == False:
            self.optimizer.step()
            self.scheduler.step()

    def recvInfo(self, Para, Len):
        self.RecvParas.append(Para)
        self.RecvLens.append(Len)

    def evaluate(self, loader=None, max_samples=100000):
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        if loader == None:
            loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                loss += self.loss_fn(y_, y).item()
                correct += (preds == y).sum().item()
                samples += y_.shape[0]
                iters += 1
                if samples >= max_samples:
                    break
        return loss / iters, correct / samples

    def saveModel(self, Path):
        torch.save(self.Model, Path)
