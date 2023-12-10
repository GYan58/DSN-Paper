from Settings import *
from Sims import *
from Utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class FL_Proc:
    def __init__(self, configs, model):
        self.DataName = configs["dname"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"]
        self.PClients = configs["pclients"]
        self.CLClients = configs["clclients"]
        self.IsIID = configs["isIID"]
        self.Alpha = configs["alpha"]
        self.Aug = configs["aug"]
        self.MaxIter = configs["iters"]
        self.LogStep = configs["logstep"]
        self.LR = configs["learning_rate"]
        self.FixLR = configs["fixlr"]
        self.GlobalLR = configs["global_lr"]
        self.WDecay = configs["wdecay"]
        self.BatchSize = configs["batch_size"]
        self.Epoch = configs["epoch"]
        self.DShuffle = configs["data_shuffle"]
        self.Normal = configs["normal"]
        self.SaveModel = configs['save_model']
        self.AttackRate = configs["attkrate"]
        self.AttackLayer = configs["attklayer"]
        self.AggMethod = configs["aggmethod"]
        self.Multiply = configs["multi"]
        self.Know = configs["know"]
        self.CheckCLP = configs["check_clp"]
        self.CheckDelta = configs["check_delta"]
        self.Budget = "off"
        self.ValidP = 0.02
        self.Server = None
        self.GModel = model
        self.Clients = {}
        self.ClientLoaders = None
        self.TrainLoader = None
        self.TestLoader = None
        self.ValidLoader = None
        self.DifLoader = None
        self.attackIDs = []
        self.Selection = RandomGet(self.NClients)

    def get_train_datas(self):
        self.ClientLoaders, self.TrainLoader, self.TestLoader, self.ValidLoader, self.DifLoader = get_loaders(self.DataName, self.NClients, self.IsIID,self.Alpha, self.Aug, False, False,self.Normal, self.DShuffle,self.BatchSize,1,self.ValidP)

    def logging(self):
        teloss, teaccu = self.Server.evaluate(self.TestLoader)

    def main(self):
        self.get_train_datas()
        self.Server = Server_Sim(self.TrainLoader, self.TestLoader, self.ValidLoader, self.DifLoader, self.GModel, self.LR, self.WDecay, self.FixLR, self.Epoch)
        for c in range(self.NClients):
            self.Clients[c] = Client_Sim(self.ClientLoaders[c], None, self.GModel, self.LR, self.WDecay, self.Epoch, self.FixLR, 10)
            self.Selection.register_client(c)
        IDs = []
        for c in range(self.NClients):
            IDs.append(c)
        FirstMLY = float(self.Multiply.split("-")[0])
        SecondMLY = float(self.Multiply.split("-")[1])
        ThirdMLY = SecondMLY
        AttackR = self.AttackRate * FirstMLY
        AttackNum = int(self.CLClients)
        AttackNum = max(0, AttackNum)
        AttackNum = min(AttackNum,len(IDs))
        self.BRecover = 10000
        BRecovered = False
        self.Selection.updateAtt(AttackNum,AttackR,self.PClients)
        Keys = list(self.Server.getParas().keys())
        Recovered = False
        NumNaN = 0
        CLPC = CPA(Delta=self.CheckDelta)
        CountNotIn = 0
        RIn = 1
        self.logging()

        for It in range(self.MaxIter):
            print(It + 1, "-th Round")
            if RIn == 0 and Recovered == False:
                Recovered = True
                AttackR = self.AttackRate * SecondMLY
                AttackNum = int(self.CLClients)
                AttackNum = max(0, AttackNum)
                AttackNum = min(AttackNum, len(IDs))
                self.Selection.updateAtt(AttackNum, AttackR, self.PClients)
            if It + 1 > self.BRecover and BRecovered == False and Recovered == True:
                BRecovered = True
                AttackR = self.AttackRate * ThirdMLY
                AttackNum = int(self.CLClients)
                AttackNum = max(0, AttackNum)
                AttackNum = min(AttackNum, len(IDs))
                self.Selection.updateAtt(AttackNum, AttackR, self.PClients)
            updateIDs, attackIDs = self.Selection.select_participant(self.PClients)
            GlobalParms = self.Server.getParas()
            TransParas = []
            TransLens = []
            TransGNs = []
            IsNaN = 0
            NaNs = []
            NaNIds = []
            for ky in updateIDs:
                if self.GlobalLR:
                    LrNow = self.Server.getLR()
                    self.Clients[ky].updateLR(LrNow, 1)
                self.Clients[ky].updateParas(GlobalParms)
                self.Clients[ky].selftrain()
                ParasNow = self.Clients[ky].getParas()
                LenNow = self.Clients[ky].DLen
                GNNow = self.Clients[ky].gradnorm
                KId = Keys[-1]
                CheckVec = ParasNow[KId].cpu()
                CheckVal = torch.sum(torch.isnan(CheckVec))
                if CheckVal == 0:
                    TransParas.append(ParasNow)
                    TransLens.append(LenNow)
                    TransGNs.append(GNNow)
                    NaNs.append(0)
                else:
                    IsNaN += 1
                    NaNIds.append(ky)
                    NaNs.append(1)
            
            NattackIDs = []
            NupdateIDs = []
            for l in range(len(attackIDs)):
                if attackIDs[l] not in NaNIds:
                    NattackIDs.append(attackIDs[l])
            for l in range(len(updateIDs)):
                if updateIDs[l] not in NaNIds:
                    NupdateIDs.append(updateIDs[l])
            
            if IsNaN >= len(updateIDs) - 2:
                NumNaN += 1
            if NumNaN >= 10:
                break
            attackIDs = NattackIDs
            updateIDs = NupdateIDs

            if self.CheckCLP:
                RIn = CLPC.Judge(TransLens, TransGNs)
            if RIn == 0:
                CountNotIn += 1
            if CountNotIn == 1 and self.Budget == "on" and FirstMLY < SecondMLY:
                ThirdMLY = 1.0
                self.BRecover = RBudget(self.MaxIter, FirstMLY, SecondMLY, It+1)

            KnowParas = []
            ExtraParas = []
            AttkParas = []
            if self.Know == "Part":
                VL = len(attackIDs)
                for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        KnowParas.append(TransParas[l])
                for l in range(len(TransLens)):
                    if len(KnowParas) >= VL:
                        break
                    if updateIDs[l] not in attackIDs:
                        KnowParas.append(TransParas[l])
            if self.Know == "Full":
                KnowParas = cp.deepcopy(TransParas)
                for l in range(len(TransLens)):
                    if updateIDs[l] not in attackIDs:
                        ExtraParas.append(TransParas[l])
            for l in range(len(TransLens)):
                    if updateIDs[l] in attackIDs:
                        AttkParas.append(TransParas[l])
            GenNum = len(attackIDs)
            AttkLayer = self.AttackLayer
            
            BadParas = attkMinMax(GlobalParms, KnowParas, GenNum, AttkLayer, AttkParas)
            count = 0
            for l in range(len(TransLens)):
                if updateIDs[l] in attackIDs:
                    TransParas[l] = BadParas[count]
                    count += 1

            for l in range(len(TransLens)):
                if NaNs[l] == 0:
                    self.Server.recvInfo(TransParas[l], TransLens[l])
            self.Server.aggParas(self.AggMethod,updateIDs,len(attackIDs))
            self.updateIDs = updateIDs
            self.attackIDs = attackIDs
            if (It + 1) % self.LogStep == 0:
                self.logging()


if __name__ == '__main__':
    Dataname = "fmnist"
    Type = "alex"
    Model = load_Model(Type, Dataname)

    Configs = {}
    Configs["check_clp"] = True
    Configs["check_delta"] = 0.01
    Configs["epoch"] = 2
    Configs['isIID'] = False
    Configs["normal"] = True
    Configs["fixlr"] = False
    Configs["global_lr"] = True
    Configs["aug"] = False
    Configs["data_shuffle"] = True
    Configs["save_model"] = False
    Configs['logstep'] = 2
    Configs['dname'] = Dataname
    Configs["mname"] = Type
    Configs['nclients'] = 128
    Configs['pclients'] = 32
    Configs['clclients'] = 32
    Configs['multi'] = "1.0-1.0"
    Configs["learning_rate"] = 0.01
    Configs["wdecay"] = 1e-4
    Configs["batch_size"] = 16
    Configs["iters"] = 200
    Configs["attklayer"] = "Full"
    Configs["attkrate"] = 0.125
    Configs["know"] = "Part"
    Configs["alpha"] = 0.1
    Configs["aggmethod"] = "TrimMean"

    FLSim = FL_Proc(Configs, Model)
    FLSim.main()



