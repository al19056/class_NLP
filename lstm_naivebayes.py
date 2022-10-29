import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

class LSTMClassifier(nn.Module):
    # vacab_size=tokenの種類数, embedding_dim=埋め込み次元, hidden_dim=隠れ状態の次元
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_size=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.is_trained = False
        self.device = device
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedd_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifer = nn.Sequential(
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=1)
        )
        self.to(device)
    
    # 埋め込み->lstm->全結合層->softmax層
    def forward(self, tokens):
        embeddings = self.embedd_layer(tokens)
        _, (last_hidden_states, _) = self.lstm(embeddings)
        return self.classifer(last_hidden_states.view(-1,self.hidden_dim))
    
    # 訓練
    def fit(self, X, y, epochs, optimizer, criterion, batch_size=32):
        self.train()
        
        #labelをindexに変換する辞書作成 eg.{"waka":0, "tanka":1}
        self.label_names = {}
        label = np.unique(y)
        num_label = len(label)
        if num_label != self.output_size:
            raise Exception("num_label != output_size")
        for i, c in enumerate(label):
            self.label_names[c] = i
        
        X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor([self.label_names[c] for c in y]).to(self.device) #index
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

        for epoch in range(epochs):
            for i, (b_X, b_y) in enumerate(train_dataloader):
                print(f'\repoch:{epoch+1} {i}/{len(train_dataloader)}', end='')
                optimizer.zero_grad()
                outputs = self.__call__(b_X)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()
            
        self.is_trained=True
    
    # 予測 (確信度)
    @torch.no_grad()
    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("not trained")
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
        result = self.__call__(X_tensor)
        return result.cpu().numpy()

    #予測 (ラベル)
    def predict(self, X):
        prob = self.predict_proba(X)
        
        list_of_key = list(self.label_names.keys())
        list_of_value = list(self.label_names.values())

        return np.array([list_of_key[list_of_value.index(v)] for v in prob.argmax(axis=1)])


class NaiveBayes():
    def __init__(self):
        self.is_trained = False

    def fit(self, X, y, smooth=1e-9): #今回のタスクではsmooth=1e-9ならばアンダーフローしない
        if len(y)!=len(X):
            raise Exception("len(X) != len(y)")

        #ラベルの辞書作成 eg:{"waka":0, "tanka":1}
        self.label_names = {}
        label = np.unique(y)
        num_label = len(label)
        for i, c in enumerate(label):
            self.label_names[c] = i
        
        y_np = np.array([self.label_names[c] for c in y], dtype=np.int32)
        X_np = np.array(X, dtype=np.float64)
        X_dim = X_np.shape[1]

        #P(c): カテゴリcの文章数 / コーパス全文章数
        y_count = np.bincount(y_np)
        self.label_prob = y_count / np.sum(y_count) #P(cj) ndarray [num_label]
        
        #P(w|c): カテゴリcかつ単語wを含む文章数 / カテゴリcの文章数
        X_hard = np.zeros_like(X_np) 
        X_hard[np.where(X_np>0)] = 1 #トークンが出現するか否か(0 or 1)の形式に変換
        self.p = np.empty((num_label, X_dim)) #P(w|c): ndarray [num_label, X_dim]
        for i in range(num_label):
            self.p[i] = np.sum(X_hard[np.where(y_np==i)], axis=0, dtype=np.float64) / y_count[i]

        self.p[np.where(self.p==0)] = smooth
        self.is_trained = True
        return

        
    # P(c)*prod(P(w|c))を正規化した確率を返す(行:データ, 列:ラベル)
    def predict_proba(self, X):
        #check trained
        if not self.is_trained:
            raise Exception("not trained")

        X_np = np.array(X, dtype=np.float64)
        #check feature dim
        if X_np.shape[1] != self.p.shape[1]:
            raise Exception("X_dim != X_trained_dim")

        prob = self._calc_prob(X_np)
        prob_sum = np.sum(prob, axis=1)
        prob = prob / prob_sum.reshape((prob_sum.shape[0], 1))
        return prob

    # ラベルを返す
    def predict(self, X):
        prob = self.predict_proba(X)
        
        list_of_key = list(self.label_names.keys())
        list_of_value = list(self.label_names.values())

        return np.array([list_of_key[list_of_value.index(v)] for v in prob.argmax(axis=1)])

    # p(c) * prod(p(w|c))
    def _calc_prob(self, X: np.ndarray):
        num_label = self.label_prob.shape[0]
        num_data = X.shape[0]

        result = np.empty((num_data, num_label))
        for i in range(num_data):
            x_feats_idx = np.where(X[i]>0) #0ではない特徴の列番号
            for j in range(num_label):
                Pwc = self.p[j][x_feats_idx] # P(w|c)
                result[i, j] = self.label_prob[j] * np.prod(Pwc)
        return result #result[n, c] = p(c)*product(p(Wi)|c)    