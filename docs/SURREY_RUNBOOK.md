# Surrey HPC：demo 與後續正式實驗

## 已核對環境

SSH alias `surrey-aisurrey` 使用既有gateway與使用者zw00924。
登入節點的工作目錄為 `/mnt/fast/nobackup/users/zw00924/VLanchor`。
demo: Slurm debug，1 GPU、4 CPU、48GB RAM、1小時；實際分配RTX A5000 24GB，
節點aisurrey-debug03。安全檢查只讀squeue/sinfo/nvidia-smi，不終止其他作業。

共享home當時約200GB且接近額滿；初始Qwen快取只有config/preprocessor設定，
沒有權重。程式把環境、pip cache、HF weights、媒體中間檔放node-local
`/var/tmp/$USER/vlanchor/$SLURM_JOB_ID`。該節點scratch當時约1.2TB可用。
這不是長期儲存承諾，下一次運行需要重新檢查。

唯讀借用基礎Python：
`/mnt/fast/nobackup/users/zw00924/miniforge3/envs/dapo-py311/bin/python`。
script在節點建立system-site-packages venv，再只於venv shadow指定HF套件。
沒有改動共享base環境。base內的vLLM依賴舊Transformers，pip會顯示其衝突；
本計分流程不import/use vLLM。正式大規模運行宜從實測依賴建立乾淨環境。
Qwen缺FLA/causal-conv1d時使用官方PyTorch參照kernel，速度較慢；若後續安裝加速
kernel，必須建立新環境並重做數值等價驗證。

## 提交新 demo

本機：

```bash
cd /Volumes/OutsourceData/VLanchor
rsync -az --exclude=.venv --exclude=.git --exclude=runs --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.DS_Store ./ surrey-aisurrey:/mnt/fast/nobackup/users/zw00924/VLanchor/
ssh -o BatchMode=yes surrey-aisurrey 'cd /mnt/fast/nobackup/users/zw00924/VLanchor && mkdir -p logs && sbatch --parsable scripts/hpc/demo.sbatch'
```

保留返回job ID。script先複製src/tests/scripts/configs/pyproject為獨立snapshot，
記錄source SHA256，再執行測試與run_demo。正常或失敗均透過EXIT trap回傳小結果。
不將模型權重/venv複製至共享home。

## 讀取狀態與取回結果

以下44314是本輪第二次作業，其他運行換成實際返回ID：

```bash
ssh -o BatchMode=yes surrey-aisurrey 'squeue -j 44314 -o "%.18i %.8T %.10M %.20R"'
ssh -o BatchMode=yes surrey-aisurrey 'tail -40 /mnt/fast/nobackup/users/zw00924/VLanchor/logs/demo-44314.out'
ssh -o BatchMode=yes surrey-aisurrey 'tail -40 /mnt/fast/nobackup/users/zw00924/VLanchor/logs/demo-44314.err'
ssh -o BatchMode=yes surrey-aisurrey 'sacct -j 44314 --format=JobID,State,Elapsed,ExitCode,AllocTRES'
rsync -az surrey-aisurrey:/mnt/fast/nobackup/users/zw00924/VLanchor/runs/surrey-44314/ runs/surrey-44314/
```

以exit_code.txt、demo_report.json、native_verification.json與tests.xml共同判斷。
squeue中消失只表示不再排隊/運行，不等於成功；warning也不等於失敗。

## 可選重用本人 node-local cache

為避免重下載，第二次作業要求同一節點，並使用第一次本人的venv/HF cache：

```bash
ssh -o BatchMode=yes surrey-aisurrey 'cd /mnt/fast/nobackup/users/zw00924/VLanchor && sbatch --parsable --nodelist=aisurrey-debug03 --export=ALL,VL_ENV_PYTHON=/var/tmp/zw00924/vlanchor/44308/venv/bin/python,VL_HF_HOME=/var/tmp/zw00924/vlanchor/44308/hf-home scripts/hpc/demo.sbatch'
```

此命令仍由Slurm排隊，不占用未分配GPU。路徑只有在資料尚存且同一節點時有效；
script若找不到venv會重新建立。HF cache不足也會重新下載。不要把此暫存路徑
寫成下一session持久依賴。預設全新demo不必指定節點或重用變量。

## 驗收層次與診斷

1. 單元測試：數學、編譯、媒體、IO、cache、分析、資料、優化與CLI。
2. GPU參照：英文10樣本×4anchors×2bridges=80項，中文3×3×1=9項。
   英文含6段文字、2圖、1圖文、1影片；影像/影片是明確標記的合成幾何刺激。
3. 完整處理：固定6個train文字參照→校準→10×4矩陣→聚類/PCA/網路/shift。
4. 正確性：warm cache精確一致、錨點重排/增加與跨模態狀態、EOT事件排序、
   selected/full logits逐token對照；共享prefill過門檻才可啟用。
5. 科學驗證：E1–E6沒有在demo中執行；合成刺激無人工gold，不能報準確率。

首作業44308已完成全部模態分數，但demo將可選prefill超閾當作失敗。原失敗run
保留。更新後將其正確記錄為disabled/reference fallback；核心full-logits檢查仍
維持0.01nat，沒有為取得passed調寬門檻。

OOM/ResourceUnavailable後停止新的模型呼叫，保留已成功/已cache項與其餘失敗grid。
不得静默換CPU、降低精度或減幀。資料預算調整需明確新spec及ID。無可用GPU則留
在排隊，繼續CPU資料/文件工作。

## 全程執行前

先確認持久scratch配額、資料保留期、完整依賴lock和輸出大小。依HANDOFF_SPEC
以32個train/dev材料估成本再拆Slurm jobs。禁止直接把demo脚本換成全benchmark
後使用共享home儲存全部權重/影片。保留每run不可變source/config/data manifests。
