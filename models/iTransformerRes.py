"""
iTransformer 的殘差深層版（Round 3）。

不動 models/iTransformer.py、layers/Transformer_EncDec.py——那些是 Round 1/2
checkpoint 依賴的既有邏輯。這裡是完全獨立的新模型檔案，靠 exp_basic.py 的
_scan_models_directory() 自動註冊機制被 --model iTransformerRes 選用。

iTransformer 的 EncoderLayer 本來就有殘差連接（Post-LN transformer），所以
「加殘差」不是從零開始，真正要解決的是「讓深度堆得動、不會馬上過擬合」，
用三個公認有效的技術（皆為業界標準做法，非本專案發明）：

1. Pre-LN 取代 Post-LN（GPT-2 之後的標準做法）—— Post-LN 深層堆疊時梯度
   容易爆炸/消失，Pre-LN 讓每個殘差分支都有一條不經過 LayerNorm 的直通路徑。
2. LayerScale（CaiT / ReZero）—— 每個殘差分支乘一個可學習的逐通道縮放，
   初始值很小（1e-4），讓深層網路一開始幾乎等價於淺層網路，深度是訓練中
   「長出來的」而不是一開始就硬塞給模型。這是本專案過擬合體質下最關鍵的一項。
3. Stochastic Depth / DropPath（ResNet / ViT）—— 訓練時隨機把整個殘差分支
   歸零，丟棄率隨深度線性增加，是比一般 dropout 更強的深層正則化。

最後一欄輸出仍是 raw logit（denorm_stdev/denorm_means 的處理跟
models/iTransformer.py 完全相同，一字不改）——訓練用的 StockFocalLoss 與
所有下游推論程式碼都依賴這個假設。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class DropPath(nn.Module):
    """Stochastic depth：訓練時以機率 drop_prob 把整個殘差分支歸零（batch
    內每個樣本獨立抽樣，要嘛整條路徑活著、要嘛整條死掉，不是逐元素 dropout）。
    drop_prob=0 時完全等於恆等函式，推論（eval）時也是恆等函式。
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """每個殘差分支一個可學習的逐通道縮放，初始值很小（預設 1e-4）。"""
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class PreLNEncoderLayer(nn.Module):
    """
    Pre-LN 版 EncoderLayer，殘差分支包 LayerScale + DropPath。

    對照 layers/Transformer_EncDec.py 的 EncoderLayer（Post-LN）：
      Post-LN: x = norm1(x + attn(x));  x = norm2(x + ffn(x))
      Pre-LN:  x = x + layerscale(droppath(attn(norm1(x))))
               x = x + layerscale(droppath(ffn(norm2(x))))
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 drop_path=0.0, layer_scale_init=1e-4):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.ls1 = LayerScale(d_model, layer_scale_init)
        self.ls2 = LayerScale(d_model, layer_scale_init)
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        y = self.norm1(x)
        new_x, attn = self.attention(y, y, y, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dp1(self.ls1(self.dropout(new_x)))

        y = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + self.dp2(self.ls2(y))

        return x, attn


class PreLNEncoder(nn.Module):
    """每層已是 Pre-LN（各自帶 norm），收尾再加一次 LayerNorm 是標準做法
    （GPT-2 style：輸出前正規化一次，避免尺度隨深度飄移）。"""
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    """
    iTransformer 殘差深層版。架構與 models/iTransformer.py 相同（inverted
    embedding + full self-attention encoder + 線性投影），差別只在 encoder
    內部用 Pre-LN + LayerScale + DropPath 取代 Post-LN。

    新增設定（皆有預設值，不傳也能跑，run.py 追加 --drop_path 供覆寫）：
      drop_path       : 最後一層的 DropPath 機率，隨深度線性增加到這個值（預設 0.1）
      layer_scale_init : LayerScale 初始值（預設 1e-4）
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        max_drop_path = getattr(configs, 'drop_path', 0.1)
        layer_scale_init = getattr(configs, 'layer_scale_init', 1e-4)
        e_layers = configs.e_layers
        # 隨深度線性增加的 DropPath schedule：第一層 0，最後一層 max_drop_path
        drop_path_rates = (
            [max_drop_path * i / max(1, e_layers - 1) for i in range(e_layers)]
            if e_layers > 1 else [0.0]
        )

        self.encoder = PreLNEncoder(
            [
                PreLNEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    drop_path=drop_path_rates[l],
                    layer_scale_init=layer_scale_init,
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer——只對非 target 欄
        # 做 de-norm，最後一欄保留 raw logit（跟 models/iTransformer.py 一致，
        # 訓練用的 StockFocalLoss 與所有下游推論都依賴這個假設，不可更動）。
        denorm_stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        denorm_means = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        denorm_stdev[:, :, -1] = 1.0
        denorm_means[:, :, -1] = 0.0
        dec_out = dec_out * denorm_stdev
        dec_out = dec_out + denorm_means

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
