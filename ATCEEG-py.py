import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt

from preprocess import get_data
from module import ATCNet, EEGNet

def train():
    data_path = 'D:/BCI_Competition_IV/data/'
    dataset_conf = {'n_sub': 9, 'data_path': data_path, 'isStandard': True, 'LOSO': False}
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()

    for i in range(n_sub):

        model = ATCNet()

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, i, LOSO, isStandard)
        eeg_train = torch.from_numpy(X_train)
        label_train = y_train_onehot
        eeg_test = torch.from_numpy(X_test)
        label_test = y_test_onehot

        train_dataset = TensorDataset(eeg_train, label_train)
        test_dataset = TensorDataset(eeg_test, label_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=300, shuffle=False)

        n_epochs = 1000

        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0009)
        TL = []
        TA = []
        VL = []
        VA = []

        for epoch in range(n_epochs):
            model.train()
            train_loss = []
            train_accs = []

            for k, batch in enumerate(train_loader):
                t1 = time.time()
                eeg, label = batch
                eeg = eeg.to(torch.float32)
                pred = model(eeg.to(device))
                loss = criterion(pred.cpu(), label.to(float))
                optim.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.conv_block.depthwise.parameters(), 1.0)
                for j in range(5):
                    nn.utils.clip_grad_norm_(model.slideOut_list[j].parameters(), 0.25)
                optim.step()
                train_loss.append(loss.item())
                acc = (pred.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()
                train_accs.append(acc)
                
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            TL.append(train_loss)
            TA.append(train_acc.cpu().item())
            t2 = time.time()
            print(t2-t1)
            print(f'[ Train | {epoch + 1: 03d} / {n_epochs: 03d} ] loss = {train_loss: .5f}, acc = {train_acc:.5f}')
            model.eval()
            valid_loss = []
            valid_accs = []

            for k, batch in enumerate(test_loader):
                eeg, label = batch
                eeg = eeg.to(torch.float32)

                with torch.no_grad():
                    logits = model(eeg.to(device))
                    loss = criterion(logits.cpu(), label.to(float))
                
                acc = (logits.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            VL.append(valid_loss)
            VA.append(valid_acc.cpu().item())
            print(f"[ Valid | {epoch + 1: 03d} / {n_epochs: 03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            
        plt.plot(TA)
        plt.plot(VA)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.plot(TL)
        plt.plot(VL)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.close()

        print('the Test Acc Best is {}, index is {}'.format(max(VA),VA.index (max(VA))))

        break #모든 subject 순회

def eegtrain():
    data_path = 'D:/BCI_Competition_IV/data/'
    dataset_conf = {'n_sub': 9, 'data_path': data_path, 'isStandard': True, 'LOSO': False}
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()

    for i in range(n_sub):

        model = EEGNet()

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, i, LOSO, isStandard)
        eeg_train = torch.from_numpy(X_train)
        label_train = y_train_onehot
        eeg_test = torch.from_numpy(X_test)
        label_test = y_test_onehot

        train_dataset = TensorDataset(eeg_train, label_train)
        test_dataset = TensorDataset(eeg_test, label_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=300, shuffle=False)

        n_epochs = 200

        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0009)
        TL = []
        TA = []
        VL = []
        VA = []

        for epoch in range(n_epochs):
            model.train()
            train_loss = []
            train_accs = []

            for k, batch in enumerate(train_loader):
                t1 = time.time()
                eeg, label = batch
                eeg = eeg.to(torch.float32)
                pred = model(eeg.to(device))
                loss = criterion(pred.cpu(), label.to(float))
                optim.zero_grad()
                loss.backward()

                optim.step()
                train_loss.append(loss.item())
                acc = (pred.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()
                train_accs.append(acc)
                
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            TL.append(train_loss)
            TA.append(train_acc.cpu().item())
            t2 = time.time()
            print(t2-t1)
            print(f'[ Train | {epoch + 1: 03d} / {n_epochs: 03d} ] loss = {train_loss: .5f}, acc = {train_acc:.5f}')
            model.eval()
            valid_loss = []
            valid_accs = []

            for k, batch in enumerate(test_loader):
                eeg, label = batch
                eeg = eeg.to(torch.float32)

                with torch.no_grad():
                    logits = model(eeg.to(device))
                    loss = criterion(logits.cpu(), label.to(float))
                
                acc = (logits.argmax(dim=-1) == label.to(device).argmax(dim=-1)).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            VL.append(valid_loss)
            VA.append(valid_acc.cpu().item())
            print(f"[ Valid | {epoch + 1: 03d} / {n_epochs: 03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            
        plt.plot(TA)
        plt.plot(VA)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.plot(TL)
        plt.plot(VL)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'val'], loc='upper left')
        plt.show()
        plt.close()

        print('the Test Acc Best is {}, index is {}'.format(max(VA),VA.index (max(VA))))

        break #모든 subject 순회

if __name__ == "__main__":
    train() # eegtrain() 사용시 eegnet으로 돌아감.