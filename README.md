# SatCloudRestore

SatCloudRestore 是一個可執行的教學型條件擴散專案：以 EuroSAT RGB 64×64 影像為乾淨目標，程序化合成雲層，再用 `xt + cloudy + mask`（7 channels）的小型 Conditional U-Net 預測 DDPM noise，並以 DDIM 少步數取樣修復。

> 這不是實務衛星去雲系統。厚雲後方沒有單張 RGB 觀測資訊，輸出只是資料分布下的合理推測，不能視為真實地理資訊。多次取樣變異也只是簡單的不確定性代理，不代表完整機率校準。

## 系統流程

```text
Clean EuroSAT image -> Synthetic cloud + mask -> Cloudy condition
          |                                      |
          +-> x0 -> q(x_t|x0) -> [x_t, cloudy, mask]
                                      |
                               Conditional U-Net
                                      |
                             epsilon prediction -> DDIM
                                      |
               mask composite -> restored image + uncertainty proxy
```

影像合成與指標固定使用 `[0,1]`；模型與 diffusion 使用 `[-1,1]`。雲層使用多尺度隨機場、Gaussian blur、quantile threshold、soft edge 與灰白/灰藍紋理：

```text
cloudy = clean * (1 - opacity * soft_mask)
       + cloud_texture * (opacity * soft_mask)
```

`binary_mask` 用來統計 coverage 和 Telea，`soft_mask` 用於合成、條件與最終 compositing。驗證與測試的雲參數由 image index 和固定 seed 決定；訓練集則在每次讀取時重新生成。

## 安裝

建議 Python 3.10–3.13。不要在 requirements 中固定 CUDA wheel；請依平台需求先從 PyTorch 官方安裝 CPU/CUDA 版本，再 editable install：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest
```

也可使用 `python -m pip install -r requirements.txt`。從 repository root 執行 scripts 時會加入正確 import root，不依賴 shell 的 `PYTHONPATH`。

## 資料準備

```powershell
python scripts/prepare_data.py --config configs/quick.yaml
```

程式透過 `torchvision.datasets.EuroSAT` 下載 RGB 版本，先以原始影像做近似 stratified split，再生成雲層。`data/splits/manifest.json` 保存相對 path、split、class label 與穩定 image ID；建立時會檢查 path/ID 不跨 split。quick 預設為 12,000/1,000/1,000；若資料不足會明確失敗。若已下載資料，可加 `--no-download`。

## Cloud demo

```powershell
python scripts/generate_demo.py --mode clouds --config configs/quick.yaml
```

輸出 `results/cloud_demo.png`，包含 10%、30%、50%、70% 四種 coverage。

## 實驗目錄、pilot training 與 resume

```powershell
python scripts/train.py --config configs/pilot_gpu.yaml --device cuda --experiment-name pilot_gpu --max-steps 500
python scripts/train.py --config configs/quick.yaml --device cuda --experiment-name quick_20260906
python scripts/train.py --config configs/quick.yaml --device cuda --experiment-name quick_20260906 --resume checkpoints/quick_20260906/latest.pt
```

預設輸出為 `checkpoints/<experiment-name>/`；也可用 `--output-dir` 指定完整目錄，例如 Google Drive。新訓練若發現該目錄已有 `best.pt`、`latest.pt`、config 或 log，會拒絕覆寫；必須改用新名稱或明確 `--resume`。quick/full 的核心模型與 diffusion 設定不因 pilot 而改變。

訓練使用 AdamW、mask-weighted epsilon MSE、gradient clipping、可選 AMP、EMA、固定驗證 noise、best/latest checkpoint、JSONL training log、config snapshot 與 DDIM preview。CLI 支援 `--max-steps`、`--output-dir`、`--experiment-name`、`--resume`、`--device`、`--num-workers`、`--batch-size`、`--gradient-accumulation`。例如 T4 顯存不足時，可將 `batch-size 16 / accumulation 4` 改成 `8 / 8`，維持 effective batch size 64。

指定 `--device cuda` 而 CUDA 不可用時會立即失敗，不會退回 CPU。啟動時會列印 Python/PyTorch/torchvision 版本、CUDA/GPU、AMP、effective batch、trainable parameters、資料量與實際輸出路徑。

checkpoint 包含 model、EMA、optimizer、scheduler、AMP scaler、epoch、global step、best validation loss、有效 config、experiment name，以及 Python、NumPy、Torch 和 CUDA RNG states。resume 會拒絕 experiment、image size、normalization、model 或 diffusion 不相容的 checkpoint。

每個實驗目錄包含：

```text
checkpoints/<experiment>/
├── best.pt
├── latest.pt
├── epoch_NNN.pt
├── config.yaml
├── training_log.jsonl
└── preview_epoch_NNN.png
```

## 評估

建議分成兩階段，避免對全部 1,000 張影像執行八次 diffusion：

```powershell
# A. 主要量化評估：完整 test split，每張只取樣一次
python scripts/evaluate.py --config configs/quick.yaml --checkpoint checkpoints/quick_20260906/best.pt --device cuda --output-dir results/quick_20260906/primary_1000 --max-images 1000 --ddim-steps 25 --samples 1

# B. uncertainty：精選 20–50 張，每張取樣八次
python scripts/evaluate.py --config configs/quick.yaml --checkpoint checkpoints/quick_20260906/best.pt --device cuda --output-dir results/quick_20260906/uncertainty_32 --image-ids ID1 ID2 ID3 --ddim-steps 25 --samples 8 --save-samples

# 也可只評估指定 coverage
python scripts/evaluate.py --config configs/quick.yaml --checkpoint checkpoints/quick_20260906/best.pt --output-dir results/quick_20260906/coverage_50 --coverage 0.5 --max-images 50 --samples 1
```

評估另支援 `--max-images`、`--image-ids`、`--coverage`、`--output-dir`、`--device`、`--ddim-steps`、`--samples`。輸出目錄已有 metrics/summary 時會拒絕覆寫。評估使用 EMA，比較 Cloudy、OpenCV Telea（RGB/BGR 明確轉換，binary uint8 mask）與 diffusion。輸出：

```text
results/<experiment>/<evaluation-name>/
├── comparisons/       # Clean | Cloudy | Mask | Telea | Diffusion | Error | Uncertainty
├── denoising_gifs/
├── uncertainty_maps/  # 未正規化的原始 variance .npy
├── metrics.csv        # per-image/per-method
└── summary.json       # 自動由評估資料聚合
```

指標 data range 固定 1.0：full PSNR/SSIM；cloud-region masked PSNR、SSIM-map 加權 SSIM、MAE；平均推論時間；多樣本時另算 cloud region 內 uncertainty–absolute-error Spearman。空 mask、常數陣列及樣本不足有安全處理。summary 依 coverage 與 method 分組，不含手填數字。

## 本地端到端 smoke

```powershell
python scripts/smoke_pipeline.py
```

它每次建立全新的 `results/gpu_ready_smoke/run_NNN/`，以小型 synthetic fixtures 驗證 experiment/output CLI、gradient accumulation、拒絕覆寫、checkpoint resume、coverage subset、image-ID subset、samples=1 與 uncertainty sampling。三步 checkpoint 的數字只驗證程式管線，**不是模型效能結果**。

## Google Colab 與 Kaggle

完整無預填輸出的 notebook 位於 [`notebooks/SatCloudRestore_Colab.ipynb`](notebooks/SatCloudRestore_Colab.ipynb)。先將 `REPO_URL` 改成實際 Git URL，再選擇 GPU runtime 依序執行。Notebook 包含 GPU 檢查、clone、安裝、Google Drive mount、資料準備、cloud demo、pilot、pilot evaluation、quick training、resume、訓練前後測試、1,000-image 主評估、32-image uncertainty 與 Gradio。

Notebook 將下列成果直接保存至 `/content/drive/MyDrive/SatCloudRestore/`：best/latest checkpoint、config snapshot、training log、validation previews、metrics.csv、summary.json 與 comparison images。

Kaggle 可使用相同 CLI，把 `--output-dir` 改到持久化輸出位置，例如 `/kaggle/working/SatCloudRestore/checkpoints/pilot_gpu`；資料若位於 Kaggle Dataset，請讓 config 的 manifest/path 指向 `/kaggle/input/...`，或在 working directory 執行資料準備。必須在 Notebook Settings 啟用 GPU；`--device cuda` 會驗證設定。

## Gradio

```powershell
python app.py --checkpoint checkpoints/quick_20260906/best.pt --config configs/quick.yaml
```

介面可上傳 RGB 圖或從 manifest test split 選圖，調整 coverage、opacity、DDIM 25/50 steps、sample count 與 seed，並顯示 clean、cloudy、mask、Telea、diffusion、uncertainty 和主要指標。輸入會 resize 到 config image size。checkpoint 不存在或 model config 不相容時停止，不以隨機模型冒充已訓練能力。CPU 可啟動但取樣較慢；import 不會啟動 server。

## 設定與測試

- `configs/quick.yaml`：64×64、base 32、multipliers `[1,2,4]`、time dim 128、200 diffusion steps、25 DDIM steps、batch 64、12 epochs。
- `configs/pilot_gpu.yaml`：與 quick 相同核心模型，batch 16、gradient accumulation 4、AMP、500 optimizer steps、EMA/checkpoint 與 25-step validation preview，適合 T4 或 8–12 GB VRAM。
- `configs/full.yaml`：base 64、1000 diffusion steps、50 DDIM steps、20 epochs，需要更多 GPU 時間。

設定 loader 會驗證必要 sections、7/3 channels、image size、split sizes、cloud ranges、timesteps 與 schedule。U-Net 使用 residual blocks、GroupNorm、SiLU、sinusoidal time embedding、down/up sampling 與 skip connections，不依賴 Diffusers model/scheduler。

```powershell
python -m compileall satcloudrestore scripts app.py
pytest -q
```

截至 2026-09-06，本機 Python 3.11 / PyTorch 2.14.0+cpu 實測：

- editable install、compileall：成功。
- pytest：16 passed（CPU）。
- cloud demo：成功，產生 `results/cloud_demo_final.png`。
- synthetic evaluation smoke：成功，4 個 test fixtures、3 methods，產生 CSV/JSON/PNG/GIF/NPY。
- synthetic tiny overfit CLI：成功，5 updates，寫入 smoke checkpoints 與 previews。
- EuroSAT quick overfit CLI：成功，固定 4-image batch 執行 3 updates，寫入 `checkpoints/best.pt`、`latest.pt` 與 previews。
- EuroSAT 4-image evaluation smoke：成功，涵蓋 10/30/50/70%，輸出 12-row metrics CSV 與自動聚合 summary；checkpoint 只訓練 3 steps，數值不是正式效能結論。
- GPU-ready CLI smoke：成功，驗證獨立 output、覆寫保護、gradient accumulation、resume 與兩種 evaluation subset。
- Gradio Blocks 建構：成功（Gradio 6.26.0），未啟動常駐 server。
- EuroSAT 官方下載與 quick 12,000/1,000/1,000 manifest：成功。
- 完整訓練、正式評估：未執行；本機無 CUDA，避免未經評估啟動長時間工作。

README 不宣稱 diffusion 優於 Telea，也不列出未正式訓練的 PSNR/SSIM。

## 限制與學術說明

- 合成雲與真實雲的光學、陰影和大氣效應有落差。
- EuroSAT RGB 解析度低；沒有多時相、SAR、熱紅外或完整多光譜資訊。
- PSNR/SSIM 高不表示生成內容具有地理或科學真實性。
- sample variance 是未校準的 uncertainty proxy。
- 本專案不是 CloudDiff 論文重製。

References: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [Palette](https://arxiv.org/abs/2111.05826), [EuroSAT](https://arxiv.org/abs/1709.00029), [U-Net](https://arxiv.org/abs/1505.04597).

## License

程式碼使用 [MIT License](LICENSE)。EuroSAT 與第三方套件仍依各自授權條款使用。
