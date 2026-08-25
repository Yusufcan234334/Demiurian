#This code is human made. İ take this code from my Slayer Minecraft AI project.

import torch
import torch.nn as siniragi
from torch.utils.data import Dataset, DataLoader, TensorDataset

x = torch.load("dataset/x.pt")
y = torch.load("dataset/y.pt")
print("x:",x.shape,"y:",y.shape)
torch.manual_seed(42)


#Daha önce yaptığımKatastrofe mimarisi ile yapıcam çünkü amacımız deli bir şey üretmek ve arada nöronları karıştırmak
#için fusion tarafını basitleştirip branchlerin böyle her kafadan farklı ses çıkan sıkıcı grup gibi davranmasını sağlayabiliriz
#şuanda zeki halinde.
class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()

        self.branchler = siniragi.ModuleList()
        self.fusion_weights = siniragi.Parameter(torch.ones(branchsayisi))
        self.residual_weight = siniragi.Parameter(torch.tensor(1.0))

        for branch in range(branchsayisi):
            katmanlar = siniragi.ModuleList()

            neuroncountin = giris
            neuroncountout = genislemecikis

            for i in range(katmansayisi):
                katmanlar.append(
                    siniragi.Linear(neuroncountin, neuroncountout)
                )

                if i == katmansayisi - 1:
                    break
                else:
                    neuroncountin = neuroncountout
                    neuroncountout = neuroncountout * 2


            for i in range(katmansayisi):
                if i == katmansayisi - 1:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2
                    neuroncountoutson = neuroncountout
                else:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2

                katmanlar.append(
                    siniragi.Linear(neuroncountin, neuroncountout)
                )

            self.branchler.append(katmanlar)
        heads = 4
        while neuroncountoutson % heads != 0 and heads > 1:
            heads //= 2
        self.attention = siniragi.MultiheadAttention(embed_dim=neuroncountoutson,num_heads=heads,batch_first=True)
        self.output1 = siniragi.Linear(neuroncountoutson,cikis)

    def forward(self, x):
        ilkx = x
        branchciktilari = []

        for katmanlar in self.branchler:
            x = ilkx

            for katmannum, katman in enumerate(katmanlar):
                x = katman(x)

                if katmannum != len(katmanlar) - 1:
                    x = torch.relu(x)

            branchciktilari.append(x)

        branchciktilari = torch.stack(branchciktilari,dim=1)
        attended, attentionweights = self.attention(branchciktilari,branchciktilari,branchciktilari)
        weights = torch.softmax(self.fusion_weights,dim=0)
        fused = (attended+ self.residual_weight * branchciktilari)
        fused = fused * weights.view(1, -1, 1)
        fused = fused.sum(dim=1)
        fused = self.output1(fused)
        return fused

model = muhtisimmodel(22,44, 2,3,13)


print("Model parametre sayısı:")
print(sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

losshesaplayici = siniragi.BCEWithLogitsLoss()
toplam_veri = len(x)
val_size = int(toplam_veri * 0.2)
indices = torch.randperm(toplam_veri)
val_indices = indices[:val_size]
train_indices = indices[val_size:]
x_val = x[val_indices]
y_val = y[val_indices]
x_train = x[train_indices]
y_train = y[train_indices]
print("Train:", x_train.shape, y_train.shape)
print("Validation:", x_val.shape, y_val.shape)
dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset,batch_size=2048,shuffle=True)


def train(model, loader, debug=True):
    losslar = []
    for i in range(220):
        for x_batch, y_batch in loader:

            optimizer.zero_grad()

            tahmin = model(x_batch)

            loss = losshesaplayici(tahmin, y_batch)
            losslar.append(loss.item())

            loss.backward()

            optimizer.step()
        tamlosslar = sum(losslar) / len(losslar)
        losslar = []
        if debug == True: print(f"Epoch {i} ortalama loss: {tamlosslar}")
        wakywakyitstimeforval(model, debug=False)
    torch.save(model.state_dict(), "demiurian.pth")


def wakywakyitstimeforval(model, debug=True):
    with torch.no_grad():
        tahmin = model(x_val)
        tahminler = (torch.sigmoid(tahmin) >= 0.5).float()
        toplam = y_val.numel()
        dogru = (tahminler == y_val).sum()
        oran = dogru / toplam
        print(f"Doğruluk: %{oran.item() * 100:.2f}")
        if debug:
            if oran * 100 >= 31.80:
                torch.save(model.state_dict(),"demiurian.pth")
                return False
            else: return True

#bu fonksiyon AI
def test_creation(model):
    print("test:")

    model.eval()

    # Zorlu bir Çöl / Volkanik ortam x vektörü hayal edelim
    # (Sıcaklık yüksek, nem düşük, yemek az, tehlike çok)
    sample_environment = torch.tensor([[
        0.95, 0.10, 0.15, 0.05, 0.90, 0.5, 0.10, 0.0, 0.30, 0.85, 0.95,
        0.33, 0.05, 0.40, 0.0, 0.05, 0.0, 0.0, 0.20, 0.90, 0.85, 0.10
    ]], dtype=torch.float32)

    with torch.no_grad():
        raw_out = model(sample_environment)
        design = torch.sigmoid(raw_out)[0]

    print("Sıcak, Kurak ve Bol Avcılı ortam için AI'nin ürettiği Canlı:")
    print(f"Su Yaşamı      : {design[0].item():.2f}")
    print(f"Otçul Eğilimi  : {design[1].item():.2f}")
    print(f"Etçil Eğilimi  : {design[2].item():.2f}")
    print(f"Hepçil Eğilimi : {design[3].item():.2f}")
    print(f"Bacak (x8)     : {int(design[4].item() * 8)}")
    print(f"Boyut          : {design[5].item():.2f}")
    print(f"Hız            : {design[6].item():.2f}")
    print(f"Görüş          : {design[7].item():.2f}")
    print(f"Zeka           : {design[8].item():.2f}")
    print(f"Güç            : {design[9].item():.2f}")
    print(f"Zırh           : {design[10].item():.2f}")
    print(f"Kamuflaj       : {design[11].item():.2f}")
    print(f"Saldırganlık   : {design[12].item():.2f}\n")


if __name__ == "__main__":
    train(model,loader, )
    test_creation(model)