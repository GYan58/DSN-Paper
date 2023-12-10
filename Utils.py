from Settings import *

from Models import Alex as Ax
from Models import VGG as Vg
from Models import FC as Fc
from Models import ResNet as RT
from Models import LSTM as LTM

import warnings
warnings.filterwarnings("ignore")

def load_Model(Type, Name):
    Model = None
    if Type == "fc":
        if Name == "mnist":
            Model = Fc.fc_mnist()

    if Type == "alex":
        if Name == "fmnist":
            Model = Ax.alex_fmnist()

        if Name == "cifar10":
            Model = Ax.alex_cifar10()

    if Type == "vgg":
        if Name == "fmnist":
            Model = Vg.vgg_fmnist()

        if Name == "cifar10":
            Model = Vg.vgg_cifar10()
            
    if Type == "resnet":
        if Name == "cifar100":
            Model = RT.resnet_cifar100()

    if Type == "lstm":
        if Name == "shake":
            Model = LTM.CharLSTM()

    return Model

class RandomGet:
    def __init__(self, Nclients=0):
        self.totalArms = OrderedDict()
        self.training_round = 0
        self.IDsPool = []
        self.Round = 0
        self.Clients = Nclients
        self.FixAIDs = []
        self.ANum = 0

    def updateAtt(self, CLNum, AttRate, PClient):
        self.CLNum = CLNum
        self.ANum = int(AttRate * PClient)
        self.FixAIDs = []
        self.FixBPool = []
        self.FixGPool = []
        for i in range(CLNum):
            self.FixAIDs.append(i)

    def register_client(self, clientId):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['status'] = True

    def updateStatus(self, Id, Sta):
        self.totalArms[Id]['status'] = Sta
    
    def select_participant(self, num_of_clients):
        viable_clients = [x for x in self.totalArms.keys() if self.totalArms[x]['status']]
        return self.getTopK(num_of_clients, viable_clients)

    def getTopK(self, numOfSamples, feasible_clients):
        pickedClients = []
        attackClients = []
        self.Round += 1
        IDs = []
        for i in range(len(feasible_clients)):
            IDs.append(i)
        BIDs = cp.deepcopy(self.FixAIDs)
        rd.shuffle(IDs)
        rd.shuffle(BIDs)
        for i in range(self.ANum):
            attackClients.append(BIDs[i])
            pickedClients.append(BIDs[i])
        c = 0
        while len(pickedClients) < numOfSamples:
            if IDs[c] not in attackClients:
                pickedClients.append(IDs[c])
            c += 1
        return pickedClients, attackClients

class CPA:
    def __init__(self, Delta=0.05, Recent = 10):
        self.R = Recent
        self.Count = 0
        self.Threshold = Delta
        self.FGNs = []
        self.In = 1
        for i in range(20):
            self.FGNs.append(0)
        self.MLim = 15

    def Proc(self,Ls,Gs):
        self.Count += 1
        SumL = np.sum(Ls)
        FGN = 0
        for i in range(len(Ls)):
            FGN += Ls[i] / SumL * Gs[i]
        return FGN
    
    def Judge(self,Ls,Gs):
        FGN = self.Proc(Ls,Gs)
        Old = np.mean(self.FGNs[-self.R:]) + 0.00000001
        self.FGNs.append(FGN)
        New = np.mean(self.FGNs[-self.R:])
        if (New - Old) / Old >= self.Threshold:
            self.In = 1
        else:
            self.In = 0
        if self.Count <= self.MLim:
            self.In = 1
        return self.In

def RBudget(Iter,FMY,SMY,CP):
    TotalB = Iter
    LeftB = TotalB - CP * FMY
    LeftR = Iter - CP
    BRecover = (LeftB - LeftR) / (SMY - 1)
    if int(BRecover) < BRecover:
        BRecover = int(BRecover) + 1 + CP
    else:
        BRecover = int(BRecover) + CP
    return BRecover

def loadGamma():
    Ga = 0.95
    return Ga
    
def toStr(V):
    S = ""
    for v in V:
        S += str(v) +","
    S = S[:-1]
    return S

from collections import defaultdict
import json

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def _one_hot(index, size):
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec

def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return index

def word_to_indices(word):
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def batch_data(data, batch_size):
    data_x = data['x']
    data_y = data['y']
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    assert train_clients == test_clients
    assert train_groups == test_groups
    return train_clients, train_groups, train_data, test_data

class ShakeSpeare(Dataset):
    def __init__(self, train=True):
        super(ShakeSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(ShakeRoot + "train",ShakeRoot  + "test")
        self.train = train
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                Length = int(len(cur_x) * 1)
                for j in range(Length):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                Length = int(len(cur_x))
                for j in range(Length):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        indices = torch.LongTensor(np.array(indices))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_sloaders(n_clients,dshuffle,batchsize):
    train_loader = ShakeSpeare(train=True)
    test_loader = ShakeSpeare(train=False)
    dict_users = train_loader.get_client_dic()
    dicts = []
    for ky in dict_users.keys():
        dicts += list(dict_users[ky])
    ELen = int(len(dicts) / n_clients)
    client_loaders = []
    for i in range(n_clients - 1):
        s_index = i * ELen
        e_index = (i + 1) * ELen
        new_dict = dicts[s_index:e_index]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)
    cloader = DataLoader(DatasetSplit(train_loader, dicts[(n_clients - 1) * ELen:]), batch_size=batchsize, shuffle=dshuffle)
    client_loaders.append(cloader)
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)
    return client_loaders, train_loader, test_loader, None

def get_shakeloader(n_clients,dshuffle,batchsize,partitions):
    train_loader = ShakeSpeare(train=True)
    test_loader = ShakeSpeare(train=False)
    client_loaders = []
    for i in range(n_clients):
        new_dict = partitions[i]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)
    return client_loaders, train_loader, test_loader, None, None

def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_cifar100():
    data_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_mnist():
    data_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_fmnist():
    data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

class Addblur(object):
    def __init__(self, blur="Gaussian"):
        self.blur = blur

    def __call__(self, img):
        if self.blur == "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        if self.blur == "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        if self.blur == "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img

class AddNoise(object):
    def __init__(self, noise="Gaussian"):
        self.noise = noise
        self.density = 0.8
        self.mean = 0.0
        self.variance = 10.0
        self.amplitude = 10.0

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        if self.noise == "pepper":
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 2] = 0
            img[mask == 1] = 255
        if self.noise == "Gaussian":
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class split_image_data(object):
    def __init__(self, dataset, labels, workers, balance=True, isIID=True, alpha=0.0):
        Perts = []
        self.Dataset = dataset
        self.Labels = labels
        self.workers = workers
        if alpha == 0 and not isIID:
            print("* Split Error...")
        if balance:
            for i in range(workers):
                Perts.append(1 / workers)
        else:
            Sum = workers * (workers + 1) / 2
            SProb = 0
            for i in range(workers - 1):
                prob = int((i + 1) / Sum * 10000) / 10000
                SProb += prob
                Perts.append(prob)
            Left = 1 - SProb
            Perts.append(Left)
            bfrac = 0.1 / workers
            for i in range(len(Perts)):
                Perts[i] = Perts[i] * 0.9 + bfrac

        if not isIID and alpha > 0:
            self.partitions = self.__getDirichlet__(labels, Perts, alpha)
        if isIID:
            self.partitions = []
            rng = rd.Random()
            data_len = len(labels)
            indexes = [x for x in range(0, data_len)]
            for frac in Perts:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def __getDirichlet__(self, data, psizes, alpha):
        n_nets = len(psizes)
        K = len(np.unique(self.Labels))
        labelList = np.array(data)
        min_size = 0
        N = len(labelList)
        net_dataidx_map = {}
        idx_batch = []
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_nets):
            net_dataidx_map[j] = idx_batch[j]
        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)
        return idx_batch

    def get_splits(self):
        clients_split = []
        for i in range(self.workers):
            IDx = self.partitions[i]
            Ls = self.Labels[IDx]
            Ds = self.Dataset[IDx]
            Xs = []
            Ys = []
            Datas = {}
            for k in range(len(Ls)):
                L = Ls[k]
                D = Ds[k]
                if L not in Datas.keys():
                    Datas[L] = [D]
                else:
                    Datas[L].append(D)
            Kys = list(Datas.keys())
            Kl = len(Kys)
            CT = 0
            k = 0
            while CT < len(Ls):
                Id = Kys[k % Kl]
                k += 1
                if len(Datas[Id]) > 0:
                    Xs.append(Datas[Id][0])
                    Ys.append(Id)
                    Datas[Id] = Datas[Id][1:]
                    CT += 1
            clients_split += [(np.array(Xs), np.array(Ys))]
            del Xs, Ys
            gc.collect()
        n_labels = len(np.unique(self.Labels))
        def print_split(clients_split):
            print("Data split:")
            for i, client in enumerate(clients_split):
                split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
                print(" - Client {}: {}".format(i, split))
        return clients_split

def get_train_data_transforms(name, aug=False, blur=False, noise=False, normal=False):
    Ts = [transforms.ToPILImage()]
    if name == "mnist" or name == "fmnist":
        Ts.append(transforms.Resize((32, 32)))
    if aug == True and name == "cifar10":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())
    if blur == True:
        Ts.append(Addblur())
    if noise == True:
        Ts.append(AddNoise())
    Ts.append(transforms.ToTensor())
    if normal == True:
        if name == "mnist":
            Ts.append(transforms.Normalize((0.06078,), (0.1957,)))
        if name == "fmnist":
            Ts.append(transforms.Normalize((0.1307,), (0.3081,)))
        if name == "cifar10":
            Ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        if name == "cifar100":
            Ts.append(transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761)))
    return transforms.Compose(Ts)

def get_test_data_transforms(name, normal=False):
    transforms_eval_F = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
    }

    transforms_eval_T = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761))
        ]),
    }

    if normal == False:
        return transforms_eval_F[name]
    else:
        return transforms_eval_T[name]

class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]

def get_loaders(Name, n_clients=10, isiid=True, alpha=0.0, aug=False, noise=False, blur=False, normal=False,dshuffle=True, batchsize=128, partial=1, vpert=0.1):
    TrainX, TrainY, TestX, TestY = [], [], [], []
    if Name == "mnist":
        TrainX, TrainY, TestX, TestY = get_mnist()
    if Name == "fmnist":
        TrainX, TrainY, TestX, TestY = get_fmnist()
    if Name == "cifar10":
        TrainX, TrainY, TestX, TestY = get_cifar10()
    if Name == "cifar100":
        TrainX, TrainY, TestX, TestY = get_cifar100()
    if Name == "shake":
        cloader, trloader, teloader, _ = get_sloaders(n_clients, False, batchsize)
        for batch_id, (inputs, targets) in enumerate(trloader):
            TrainX += list(inputs.detach().numpy())
            TrainY += list(targets.detach().numpy())
        for batch_id, (inputs, targets) in enumerate(teloader):
            TestX += list(inputs.detach().numpy())
            TestY += list(targets.detach().numpy())
        TrainY = np.array(TrainY)
        TrainX = np.array(TrainX)
        SPL = split_image_data(TrainX, TrainY, n_clients, True, isiid, alpha)
        return get_shakeloader(n_clients,dshuffle,batchsize,SPL.partitions)

    transforms_train = get_train_data_transforms(Name, aug, blur, noise, normal)
    transforms_eval = get_test_data_transforms(Name, normal)
    splits = split_image_data(TrainX, TrainY, n_clients, True, isiid, alpha).get_splits()

    client_loaders = []
    valid_x = []
    valid_y = []
    SumL = 0
    VdL = 0
    DifXs = {}
    DifYs = {}
    for i in range(10):
        DifXs[i] = []
        DifYs[i] = []
    for x, y in splits:
        if vpert < 0.5:
            L = int(len(x) * vpert)
            vx = list(x[:L])
            vy = list(y[:L])
            valid_x += vx
            valid_y += vy
            VdL += L
            for i in range(len(vy)):
                yi = vy[i]
                xi = vx[i]
                DifXs[yi].append(xi)
                DifYs[yi].append(yi)
        if partial < 1:
            L = int(len(x) * partial)
            x = x[:L]
            y = y[:L]
            SumL += L
        client_loaders.append(
            torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batchsize, shuffle=dshuffle))

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(TrainX, TrainY, transforms_eval), batch_size=2000,shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(TestX, TestY, transforms_eval), batch_size=2000,shuffle=False, num_workers=2)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)
    valid_loader = torch.utils.data.DataLoader(CustomImageDataset(valid_x, valid_y, transforms_train), batch_size=batchsize,shuffle=False)
    dif_loaders = {}
    for ky in DifXs.keys():
        X = np.array(DifXs[ky])
        Y = np.array(DifYs[ky])
        loader_now = torch.utils.data.DataLoader(CustomImageDataset(X, Y, transforms_train), batch_size=batchsize,shuffle=False)
        dif_loaders[ky] = loader_now
    return client_loaders, train_loader, test_loader, valid_loader, dif_loaders

def replaceLayer(GParas, BPara, Layer):
    Res = []
    for i in range(len(GParas)):
        ParaNow = cp.deepcopy(GParas[i])
        if Layer != "Full" and Layer != "No":
            C = 0
            for ky in BPara.keys():
                C += 1
                if C == Layer:
                    ParaNow[ky] = cp.deepcopy(BPara[ky])
        Res.append(ParaNow)
    return Res

def avgParas(Paras):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = 1 / len(Paras)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res

def wavgParas(Paras, Lens):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = Lens[i] / np.sum(Lens)
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res

def minusParas(Para1, Multi, Para2):
    Res = cp.deepcopy(Para1)
    for ky in Res.keys():
        Res[ky] = Para1[ky] - Para2[ky] * Multi
    return Res

def getGrad(P1, P2):
    Res = cp.deepcopy(P1)
    for ky in Res.keys():
        if "weight" in ky or "bias" in ky:
            Res[ky] = P1[ky] - P2[ky]
        else:
            Res[ky] -= Res[ky]
    return Res

def getDirc(Paras, RePara, Pt=0):
    RPara = cp.deepcopy(RePara)
    APara = avgParas(Paras)
    Grad = getGrad(APara, RPara)
    Kys = Grad.keys()
    Direction = cp.deepcopy(RPara)
    Ths = 0
    Num0 = 0
    NumS = 0
    NumB = 0
    for ky in Kys:
        GParas = Grad[ky].cpu().detach().numpy()
        SParas = np.sign((np.abs(GParas) > Ths) * GParas)
        IsFloat = 0
        if type(SParas) == type(np.float32(1.0)):
            IsFloat = 1
        if type(SParas) == type(np.float64(1.0)):
            IsFloat = 1
        if IsFloat == 0:
            Direction[ky] = torch.from_numpy(SParas).to(device)
        else:
            Grad[ky] -= Grad[ky]
        Num0 += np.sum(SParas == 0)
        NumS += np.sum(SParas == -1)
        NumB += np.sum(SParas == 1)
    return Direction

def getcosSim(w0, w1):
    Kys = w0.keys()
    Norm0 = 0.000001
    Norm1 = 0.000001
    Dots = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V0 = w0[ky].cpu()
            V1 = w1[ky].cpu()
            Norm0 += torch.norm(V0) ** 2
            Norm1 += torch.norm(V1) ** 2
            Dots += torch.sum(torch.mul(V0, V1))
    Sim = Dots / np.sqrt(Norm0 * Norm1)
    return Sim

def getDist(w0, w1):
    Dist = 0
    Kys = w0.keys()
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            Dist += np.linalg.norm(w0[ky].cpu().detach().numpy() - w1[ky].cpu().detach().numpy()) ** 2
    Dist = np.sqrt(Dist)
    return Dist

def genParas(Para, Epsi, N):
    Paras = []
    NumP = 0
    Kys = Para.keys()
    for ky in Kys:
        GPara = Para[ky].cpu().detach().numpy()
        Shape = GPara.shape
        Mult = 1
        L = len(Shape)
        for i in range(L):
            Mult *= Shape[i]
        NumP += Mult
    Gap = np.sqrt(Epsi ** 2 / NumP) / N
    for i in range(N):
        NPara = cp.deepcopy(Para)
        for ky in Kys:
            GPara = Para[ky].cpu().detach().numpy() + i * Gap
            IsFloat = 0
            if type(GPara) == type(np.float64(1.0)):
                IsFloat = 1
            if type(GPara) == type(np.float32(1.0)):
                IsFloat = 1
            if IsFloat == 0:
                NPara[ky] = torch.from_numpy(GPara).to(device)
        Paras.append(NPara)
    return Paras

def attkKrum(RePara, KParas, EParas, GNum, ALayer, AParas):
    if GNum <= 1:
        return [RePara]
    Direction = getDirc(KParas, RePara, 1)
    GoalIDs = []
    FindPara = None
    FindLam = 0.1
    Stop = False
    C = 0
    while Stop == False:
        RPara = cp.deepcopy(RePara)
        NPara = minusParas(RPara, FindLam, Direction)
        AParas = []
        Goals = []
        for i in range(GNum):
            AParas.append(NPara)
            Goals.append(i)
        AParas += EParas
        _, ID = AggMKrum(AParas, len(Goals))
        if ID in Goals:
            Stop = True
            FindPara = NPara
        else:
            FindLam = FindLam * 0.5
        if FindLam < 0.000001:
            Stop = True
            FindPara = NPara
        C += 1
    AttParas = []
    if ALayer == "Full":
        for i in range(GNum):
            AttParas.append(FindPara)
    else:
        AttParas = replaceLayer(AParas, FindPara, ALayer)
    return AttParas

def attkPKrum(RePara, KParas, EParas, GNum, ALayer, AParas):
    if GNum <= 1:
        return [RePara]
    Direction = getDirc(KParas, RePara, 1)
    FindPara = None
    FindLam = 0.1
    Stop = False
    C = 0
    while Stop == False:
        RPara = cp.deepcopy(RePara)
        NPara = minusParas(RPara, FindLam, Direction)
        AParas = [NPara]
        Goals = [0]
        AParas += KParas
        _, ID = AggMKrum(AParas, len(Goals))
        if ID in Goals:
            Stop = True
            FindPara = NPara
        else:
            FindLam = FindLam * 0.5
        if FindLam < 0.000001:
            Stop = True
            FindPara = NPara
        C += 1
    AttParas = []
    if ALayer == "Full":
        for i in range(GNum):
            AttParas.append(FindPara)
    else:
        AttParas = replaceLayer(AParas, FindPara, ALayer)
    return AttParas

def attkMinMax(RePara, KParas, GNum, ALayer, AParas):
    if GNum <= 1:
        return [RePara]
    RPara = cp.deepcopy(RePara)
    Direction = getDirc(KParas, RePara, 1)
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RPara))
    AGD = avgParas(Grads)
    Dists = defaultdict(dict)
    MaxDist = -1
    N = len(KParas)
    for i in range(N):
        G1 = Grads[i]
        for j in range(i, N):
            G2 = Grads[j]
            dist = getDist(G1, G2)
            if dist > MaxDist:
                MaxDist = dist
    Gamma = 0.1
    Stop = False
    FindGrad = None
    while Stop == False:
        NGrad = minusParas(AGD, Gamma, Direction)
        Maxdist = -1
        for i in range(N):
            dist = getDist(NGrad, Grads[i])
            if dist > Maxdist:
                Maxdist = dist
        if Maxdist < MaxDist or Gamma < 0.000001:
            Stop = True
            FindGrad = NGrad
        else:
            Gamma = Gamma * 0.5
    AttParas = []
    FindPara = minusParas(RPara, -1, FindGrad)
    if ALayer == "Full":
        for i in range(GNum):
            AttParas.append(FindPara)
    else:
        AttParas = replaceLayer(AParas, FindPara, ALayer)
    return AttParas

def attkMinSum(RePara, KParas, GNum, ALayer, AParas):
    if GNum <= 1:
        return [RePara]
    RPara = cp.deepcopy(RePara)
    Direction = getDirc(KParas, RePara, 1)
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RPara))
    AGD = avgParas(Grads)
    Dists = defaultdict(dict)
    N = len(KParas)
    MaxDist = -1
    for i in range(N):
        G1 = Grads[i]
        GetDist = 0
        for j in range(N):
            G2 = Grads[j]
            dist = getDist(G1, G2)
            GetDist += dist

        if GetDist > MaxDist:
            MaxDist = GetDist
    Gamma = 0.1
    Stop = False
    FindGrad = None
    while Stop == False:
        NGrad = minusParas(AGD, Gamma, Direction)
        Maxdist = 0
        for i in range(N):
            dist = getDist(NGrad, Grads[i])
            Maxdist += dist

        if Maxdist < MaxDist or Gamma < 0.000001:
            Stop = True
            FindGrad = NGrad
        else:
            Gamma = Gamma * 0.5
    AttParas = []
    FindPara = minusParas(RPara, -1, FindGrad)
    if ALayer == "Full":
        for i in range(GNum):
            AttParas.append(FindPara)
    else:
        AttParas = replaceLayer(AParas, FindPara, ALayer)
    return AttParas

def attkLie(RePara, KParas, GNum, ANum, ALayer, AParas):
    if GNum <= 1:
        return [RePara]
    N = ANum
    M = int(N * 0.25)
    S = int(N / 2 + 1) - M
    Z = st.norm.ppf((N - M - S) / (N - M)) + 0.01
    Grads = []
    for i in range(len(KParas)):
        Grads.append(getGrad(KParas[i], RePara))
    AGD = avgParas(Grads)
    Direction = getDirc(KParas, RePara, 1)
    FGrad = cp.deepcopy(AGD)
    Kys = KParas[0].keys()
    for ky in Kys:
        Gs = []
        for i in range(len(Grads)):
            grad = Grads[i][ky].cpu().detach().numpy()
            Gs.append(grad)
        Mu = np.mean(Gs, axis=0)
        Std = np.std(Gs, axis=0)
        Dirc = Direction[ky].cpu().detach().numpy()
        Res = Mu + Z * Std * (Dirc < 0) - Z * Std * (Dirc > 0)
        FGrad[ky] = torch.from_numpy(Res).to(device)
    AttParas = []
    FPara = minusParas(RePara, -1, FGrad)
    if ALayer == "Full":
        for i in range(GNum):
            AttParas.append(FPara)
    else:
        AttParas = replaceLayer(AParas, FPara, ALayer)
    return AttParas

def getNorm(W, pt=0):
    Kys = W.keys()
    Norm = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V = W[ky].cpu().detach().numpy()
            Norm += np.linalg.norm(V) ** 2
            if pt == 1:
                print(ky, Norm, V)
    return np.sqrt(Norm)

def getDots(w0, w1):
    Kys = w0.keys()
    Dots = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V0 = w0[ky].cpu()
            V1 = w1[ky].cpu()
            Dots += torch.sum(torch.mul(V0, V1))
    return Dots

def getNegLam(G, S, A):
    SP = 0.1
    EP = 0.0001
    L = 100
    Gap = (SP - EP) / L
    D1 = getDots(G, G)
    D2 = getDots(G, S)
    D3 = getNorm(G)
    FindVal = 100000
    FindLam = -1
    for i in range(L):
        l = SP - Gap * i
        D4 = minusParas(G, l, S)
        D5 = getNorm(D4)
        val = abs(A - (D1 - l * D2) / D3 / D5)
        if val < FindVal:
            FindVal = val
            FindLam = l
    return FindLam

def getPGrad(Gnow, Dirc):
    Res = cp.deepcopy(Gnow)
    Kys = Res.keys()
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            GDirc = Dirc[ky].cpu().detach().numpy()
            NG = Gnow[ky].cpu().detach().numpy()
            R = (GDirc != 0) * NG
            Res[ky] = torch.from_numpy(R).to(device)
    return Res


def AggMKrum(Paras, Frac, Num=1):
    N = len(Paras)
    M = N - Frac
    if M <= 1:
        M = N
    Distances = defaultdict(dict)
    Kys = Paras[0].keys()
    for i in range(N):
        Pa1 = Paras[i]
        for j in range(i, N):
            Pa2 = Paras[j]
            distance = 0
            if i != j:
                for ky in Kys:
                    if "weight" in ky or "bias" in ky:
                        distance += np.linalg.norm(Pa1[ky].cpu().detach().numpy() - Pa2[ky].cpu().detach().numpy()) ** 2
                distance = np.sqrt(distance)
            Distances[i][j] = distance
            Distances[j][i] = distance
    if Num == 1:
        FindID = -1
        FindVal = pow(10, 20)
        PDict = {}
        for i in range(N):
            Dis = sorted(Distances[i].values())
            SumDis = np.sum(Dis[:M])
            PDict[i] = SumDis
            if FindVal > SumDis:
                FindVal = SumDis
                FindID = i
        return Paras[FindID], FindID
    if Num >= 2:
        Dist = cp.deepcopy(Distances)
        PDict = {}
        for i in range(N):
            Dis = sorted(Dist[i].values())
            SumDis = np.sum(Dis[:M])
            PDict[i] = SumDis
        SDict = sorted(PDict.items(), key=lambda x: x[1])
        GParas = []
        for i in range(Num):
            Ky = SDict[i][0]
            GParas.append(Paras[Ky])
        return avgParas(GParas), -1

def AggBulyan(Paras, Frac, Num=1):
    N = len(Paras)
    M = N - Frac - 1
    if 2 * Frac >= int(N / 2):
        Frac = int(N / 4 - 2)
        M = N - Frac - 1
    Distances = defaultdict(dict)
    Kys = Paras[0].keys()
    for i in range(N):
        Pa1 = Paras[i]
        for j in range(i, N):
            Pa2 = Paras[j]
            distance = 0
            if i != j:
                for ky in Kys:
                    if "weight" in ky or "bias" in ky:
                        distance += np.linalg.norm(Pa1[ky].cpu().detach().numpy() - Pa2[ky].cpu().detach().numpy()) ** 2
                distance = np.sqrt(distance)
            Distances[i][j] = distance
            Distances[j][i] = distance
    Dist = cp.deepcopy(Distances)
    PDict = {}
    for i in range(N):
        Dis = sorted(Dist[i].values())
        SumDis = np.sum(Dis[:M])
        PDict[i] = SumDis
    SDict = sorted(PDict.items(), key=lambda x: x[1])
    GParas = []
    for i in range(Num):
        Ky = SDict[i][0]
        GParas.append(Paras[Ky])
    FPara = cp.deepcopy(GParas[0])
    Kys = GParas[0].keys()
    C = 0
    for ky in Kys:
        Ms = []
        for i in range(len(GParas)):
            Para = GParas[i][ky].cpu().detach().numpy()
            Ms.append(Para)
        SMs = np.sort(Ms, axis=0)
        GMs = []
        for i in range(Frac, len(GParas) - Frac):
            GMs.append(SMs[i])
        GetParas = np.mean(GMs, axis=0)
        FPara[ky] = torch.from_numpy(GetParas).to(device)
    return FPara, -1

def AggTrimMean(Paras, Frac):
    N = len(Paras)
    K = Frac
    if K >= int(N / 2):
        K = int(N / 2) - 1
    FPara = cp.deepcopy(Paras[0])
    Kys = Paras[0].keys()
    C = 0
    for ky in Kys:
        Ms = []
        for i in range(N):
            Para = Paras[i][ky].cpu().detach().numpy()
            Ms.append(Para)
        SMs = np.sort(Ms, axis=0)
        GMs = []
        for i in range(K, N - K):
            GMs.append(SMs[i])
        GetParas = np.mean(GMs, axis=0)
        FPara[ky] = torch.from_numpy(GetParas).to(device)
    return FPara, -1

def AggTrimMed(Paras, Frac=0):
    N = len(Paras)
    M = int(N * 0.25)
    K = int(N / 2) - M
    FPara = cp.deepcopy(Paras[0])
    Kys = Paras[0].keys()
    for ky in Kys:
        Ms = []
        for i in range(N):
            Para = Paras[i][ky].cpu().detach().numpy()
            Ms.append(Para)
        SMs = np.sort(Ms, axis=0)
        GMs = []
        for i in range(K, N - K):
            GMs.append(SMs[i])
        GetParas = np.mean(GMs, axis=0)
        FPara[ky] = torch.from_numpy(GetParas).to(device)
    return FPara, -1

def getSim(w0, w1):
    Kys = w0.keys()
    Norm0 = 0
    Norm1 = 0
    Dots = 0
    for ky in Kys:
        if "weight" in ky or "bias" in ky:
            V0 = w0[ky].cpu()
            V1 = w1[ky].cpu()
            Norm0 += torch.norm(V0) ** 2
            Norm1 += torch.norm(V1) ** 2
            Dots += torch.sum(torch.mul(V0, V1))
    Sim = Dots / np.sqrt(Norm0 * Norm1)
    return Sim

class AFA:
    def __init__(self):
        self.Alphas = {}
        self.Betas = {}

    def Add(self, Id):
        self.Alphas[Id] = 0.5
        self.Betas[Id] = 0.5

    def AggParas(self, IDs, RPara, Paras, Lens):
        for ky in IDs:
            if ky not in self.Alphas.keys():
                self.Add(ky)
        LcGrads = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            LcGrads[Ky] = getGrad(Paras[i], RPara)
        Good = []
        Ls = {}
        Pks = {}
        for i in range(len(Paras)):
            Ky = IDs[i]
            Good.append(Ky)
            Ls[Ky] = Lens[i]
            Pks[Ky] = self.Alphas[Ky] / (self.Alphas[Ky] + self.Betas[Ky])
        Bad = []
        R = [0]
        Epi = 0.5
        Step = 2
        while len(R) > 0:
            R = []
            GDs = []
            GLs = []
            for ky in Good:
                GDs.append(LcGrads[ky])
                GLs.append(Ls[ky] * Pks[ky])
            GR = wavgParas(GDs, GLs)
            Sims = {}
            ASims = []
            for ky in Good:
                sim = getSim(LcGrads[ky], GR)
                Sims[ky] = sim
                ASims.append(sim)
            Mu = np.mean(ASims)
            Std = np.std(ASims)
            Med = np.median(ASims)
            for ky in Good:
                sim = Sims[ky]
                IAdd = False
                if Mu < Med:
                    if sim < Med - Std * Epi:
                        IAdd = True
                else:
                    if sim > Med + Std * Epi:
                        IAdd = True
                if IAdd:
                    Bad.append(ky)
                    Good = list(set(Good) - set([ky]))
                    R.append(ky)
            Epi = Epi + Step
        GDs = []
        GLs = []
        for ky in Good:
            GDs.append(LcGrads[ky])
            GLs.append(Ls[ky] * Pks[ky])
        GRad = wavgParas(GDs, GLs)
        Res = minusParas(RPara, -1, GRad)
        for ky in Good:
            self.Alphas[ky] += 1
        for ky in Bad:
            self.Betas[ky] += 1
        return Res, -1

def AggTrust(Pure, IDs, RPara, Paras, Lens):
    PureGrad = getGrad(Pure, RPara)
    LcGrads = []
    for i in range(len(Paras)):
        LcGrads.append(getGrad(Paras[i], RPara))
    if len(Paras) != len(Lens):
        exit()
    TSc = []
    for i in range(len(Paras)):
        TSc.append(getSim(PureGrad, LcGrads[i]))
    Ws = []
    Bad = []
    for i in range(len(Paras)):
        Ky = IDs[i]
        if TSc[i] > 0:
            Ws.append(TSc[i])
        else:
            Ws.append(0.0)
            Bad.append(Ky)
    Res = wavgParas(Paras, Ws)
    return Res, -1

def EVal(Model=None, loader=None):
    LData = cp.deepcopy(loader)
    Model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss, correct, samples, iters = 0, 0, 0, 0
    C = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(LData):
            x, y = x.to(device), y.to(device)
            C += 1
            y_ = Model(x)
            _, preds = torch.max(y_.data, 1)
            loss += loss_fn(y_, y).item()
            correct += (preds == y).sum().item()
            samples += y_.shape[0]
            iters += 1
            if C >= 1:
                break
    return loss / iters, correct / samples

def AggUnion(RPara, Paras, Lens, VData, Model, Frac, Uids):
    LcGrads = []
    for i in range(len(Paras)):
        LcGrads.append(getGrad(Paras[i], RPara))
    APara = avgParas(Paras)
    MLA = cp.deepcopy(Model)
    MLA.load_state_dict(APara)
    La, Ea = EVal(MLA, VData)
    EBs = {}
    LBs = {}
    for i in range(len(Paras)):
        SParas = []
        for j in range(len(Paras)):
            if j != i:
                SParas.append(Paras[j])
        BPara = avgParas(SParas)
        MLB = cp.deepcopy(Model)
        MLB.load_state_dict(BPara)
        Lb, Eb = EVal(MLB, VData)
        EBs[i] = Eb - Ea
        LBs[i] = La - Lb
    NEs = sorted(EBs.items(), key=lambda x: x[1], reverse=True)
    NLs = sorted(LBs.items(), key=lambda x: x[1], reverse=True)
    Dels = []
    Find = []
    Num = int(Frac / 2)
    for i in range(len(NEs)):
        Ky = NEs[i][0]
        Val = NEs[i][1]
        if Val > 0:
            Dels.append(Ky)
            Find.append(Uids[Ky])
        if len(Dels) >= Num:
            break
    for i in range(len(NLs)):
        Ky = NLs[i][0]
        Val = NLs[i][1]
        if Ky not in Dels and Val >= 0:
            Dels.append(Ky)
            Find.append(Uids[Ky])
        if len(Dels) >= Frac:
            break
    GParas = []
    GLens = []
    for i in range(len(Paras)):
        if i not in Dels:
            GParas.append(Paras[i])
            GLens.append(Lens[i])
    return wavgParas(GParas, GLens)





