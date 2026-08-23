from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

#123

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class StockFocalLoss(nn.Module):
    def __init__(self, alpha=0.77, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, true):
        # 1. 將真實標籤還原為乾淨的 0 和 1
        true_binary = (true > 0).float()

        # 🚨 2. 最關鍵的一步：使用帶有 Logits 的 BCE！
        # 這裡的函數內建了極度穩定的 Sigmoid，我們不需要 (也不可以) 自己做 clamp 或 sigmoid
        bce_loss = F.binary_cross_entropy_with_logits(logits, true_binary, reduction='none')

        # 3. 數學黑魔法：反推真實機率 pt
        # 因為 bce_loss = -log(pt)，所以用 exp(-bce_loss) 就可以完美還原 pt，且絕對不會有極限值報錯
        pt = torch.exp(-bce_loss)

        # 4. Focal Loss 核心加權
        alpha_factor = true_binary * self.alpha + (1 - true_binary) * (1 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


# ══════════════════════════════════════════════════════════════
# Round 3 損失函數比較實驗（見計畫文件）。
#
# 現行 StockFocalLoss 是為極端類別不平衡設計的（RetinaNet 原始場景 1:1000），
# 我們的標籤是 50.1/49.9，focal_alpha≈0.5 讓 α 項幾乎是 no-op，又因為
# pt≈0.5 幾乎處處成立，(1-pt)^gamma 近乎均勻縮放——主要效果是把模型答對
# 樣本的梯度壓掉，弱訊號問題上等於丟掉本來就稀少的可用訊號。以下變體用來
# 實測哪個訓練目標最適合這個問題，全部只用 train_event.py 的 --loss_variant
# 切換，預設值 'focal_g2' 與既有行為完全相同，不影響 Round 1/2 的可重現性。
# ══════════════════════════════════════════════════════════════
class BCELoss(nn.Module):
    """標準 logistic 回歸，等於 StockFocalLoss(alpha=0.5, gamma=0)——
    移除 focal 的梯度抑制，直接測試「不加任何加權」的基準。"""
    def forward(self, logits, true):
        true_binary = (true > 0).float()
        return F.binary_cross_entropy_with_logits(logits, true_binary)


class BCELabelSmoothLoss(nn.Module):
    """BCE + label smoothing：把 0/1 標籤軟化成 [smoothing, 1-smoothing]，
    金融標籤本身雜訊大，抑制模型對單一根 K 棒過度自信。"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, true):
        true_binary = (true > 0).float()
        target = true_binary * (1 - 2 * self.smoothing) + self.smoothing
        return F.binary_cross_entropy_with_logits(logits, target)


class BrierLoss(nn.Module):
    """Brier score（proper scoring rule）：直接在機率空間算均方誤差，
    通常比 BCE 校準更好、機率不容易被壓縮到極窄範圍。"""
    def forward(self, logits, true):
        true_binary = (true > 0).float()
        prob = torch.sigmoid(logits)
        return F.mse_loss(prob, true_binary)


class PairwiseRankLoss(nn.Module):
    """
    Pairwise logistic ranking loss（RankNet 核心項）：對 batch 內每一組
    (正樣本, 負樣本)，懲罰「正樣本 logit 沒有比負樣本 logit 高」的程度。

    這是六個變體裡最貼合關卡目標的一個——叢集進場關卡衡量的是「機率排序
    前段的勝率」，本質是排序品質而非逐點校準品質，用排序損失直接優化這件
    事，比用分類損失間接優化更貼近實際被評估的指標。

    若某個 batch 剛好全是同一類別（正樣本或負樣本數為 0），回傳 0（不提供
    梯度但不報錯）——在 ~50/50 標籤、batch_size>=32 下機率趨近於零，只是
    防禦性處理。
    """
    def forward(self, logits, true):
        true_binary = (true > 0).float().reshape(-1)
        logits = logits.reshape(-1)

        pos = logits[true_binary == 1]
        neg = logits[true_binary == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.sum() * 0.0

        diff = pos.unsqueeze(1) - neg.unsqueeze(0)  # [P, Q]
        return F.softplus(-diff).mean()


def build_loss_criterion(args):
    """依 args.loss_variant 選損失函數，預設 'focal_g2' 與既有行為完全相同。"""
    variant = getattr(args, 'loss_variant', 'focal_g2')
    alpha = getattr(args, 'focal_alpha', 0.77)

    if variant == 'focal_g2':
        return StockFocalLoss(alpha=alpha, gamma=2.0)
    if variant == 'focal_g05':
        return StockFocalLoss(alpha=alpha, gamma=0.5)
    if variant == 'bce':
        return BCELoss()
    if variant == 'bce_ls':
        return BCELabelSmoothLoss(smoothing=0.05)
    if variant == 'brier':
        return BrierLoss()
    if variant == 'rank':
        return PairwiseRankLoss()
    raise ValueError(f"不支援的 loss_variant: {variant}")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return build_loss_criterion(self.args)


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
