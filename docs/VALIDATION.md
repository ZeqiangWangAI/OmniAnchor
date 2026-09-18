# 實際驗證紀錄

本文件以下保存初始 demo 的歷史驗證結果。完整研究已接續執行，部分外部效度與
基線比較已完成，但 E1–E6 尚未全部完成；最新狀態與結果邊界見
[完整研究執行狀態](RESEARCH_STATUS.md)，不能將以下歷史「尚未取得」列表當作最新進度。
詳細機器可讀證據：[surrey-44314-evidence.json](../reports/surrey-44314-evidence.json)。

## Surrey GPU demo

| 項目 | 實測 |
|---|---|
| Slurm job | 44314，COMPLETED，ExitCode 0:0 |
| 日期 | 2026-09-09 UTC / 2026-09-10 Asia/Shanghai |
| 節點 / GPU | aisurrey-debug03 / NVIDIA RTX A5000 24GB |
| allocation | 1 GPU、4 CPU、48GB host RAM |
| Slurm全程 / demo計時 | 3分50秒 / 177.08秒 |
| peak CUDA allocated | 9,184,399,872 bytes = 8.5536 GiB |
| Python / PyTorch / CUDA runtime | 3.11.15 / 2.9.0+cu128 / 12.8 |
| Transformers / tokenizer | 5.18.0.dev0 / tokenizers 0.23.2 |
| 模型 | Qwen/Qwen3.5-4B，BF16、eval、thinking off |
| model revision | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| TF source commit | 4815a0a6a064214f2d8208c094464a5a6b76ca8d |
| 該source snapshot單元測試 | 167 passed，14.27秒 |
| 原生計分 | 英文80項 + 中文9項，0失敗 |
| 再次cache讀取 | 英文80/80命中，逐分數完全一致 |
| 錨點增加/重排後分數差 | 4個原錨點均為0 nat |
| selected/full logits對照 | 2個多token錨點、4 tokens，最大差0 nat |
| 可選shared prefill | 最大差0.01558018 nat > 0.01，disabled，保留參照路徑 |

英文10個材料包含6個文本、2張圖片、1個圖文、1個8幀影片；4 anchors×2 bridges。
中文3個文本×3 anchors×1 bridge。圖片與影片是幾何合成刺激，文本也是合成例句，
不是公開benchmark或人工價值標註。原始英文logp範圍約[-24.7771,-5.1744]。
實作成功不表示任何價值/情緒/詞義準確率。

完整鏈條生成 raw scores → 固定6文本train reference → log-ratio/z → 10×4矩陣
→ KMeans(k=2)/PCA/概念網路/可靠度/模板半徑/示範shift。
示範shift僅20 bootstrap、19 permutation用來驗證輸出路徑，不能當正式推論；
正式協議為1000次並按真實來源單位重抽樣。

跨模態狀態檢查是在上述圖/圖文/影片處理後，重新計分最初的文本，與先前逐anchor
結果一致。完整logits對照只在有明確長度上限的短文本執行，不冒稱覆蓋所有輸入。
目前沒有啟用多token cache。shared-prefill失敗保留數值、停用加速；門檻未調寬。

原始證據：

* [demo_report.json](../reports/surrey-44314/demo_report.json)
* [native_verification.json](../reports/surrey-44314/native_verification.json)
* [job source hashes](../reports/surrey-44314/job_manifest.json)
* [unit-test JUnit](../reports/surrey-44314/tests.xml)
* [raw英文](../reports/surrey-44314/scores.parquet)、[raw中文](../reports/surrey-44314/scores-zh.parquet)
* [原始環境快照](../locks/surrey-44314-observed.txt)

sidecar中的媒體路徑保留原node-local位置，取回後不要手改其歷史provenance。
相同stimuli已複製到run的stimuli/；重跑可由run_demo重新生成，或建新Sample指向
複製後媒體並記錄新run身份。

## 保留的失敗與修正

首作業44308：87項當時snapshot測試通過，原生英文/中文及全部媒體計分完成，
demo因可選shared-prefill數值差超閾而raise，Slurm FAILED/1:0，全程5分37秒。
backend已有安全回退，但demo把可選優化不通過錯列成整體失敗。更新測試程式後
以新作業44314驗證並記錄disabled；首run與日誌均保留，沒有覆寫。

本地整合過程也修復候選級失敗污染其他anchor、cold/warm cache狀態差异、
評估CSV列順序影響指標、ResourceUnavailable後反覆呼叫、顯式backend與宣告
資源/system prompt不一致，以及優化器失敗繞過預算/共享buffer污染等問題。
相應回歸測試均在tests/保存。

## 本機驗證與發行檔

本機Python3.11.14，CPU toy/injected模型與真實PyAV媒體解碼；未在本機跑4B模型。
完整最新測試 **194 passed**，結果保存到
[reports/local-tests.xml](../reports/local-tests.xml)；ruff全數通過。
本機toy demo同樣完成89项、80 cache hits、0失敗；見
[local-final-demo](../runs/local-final-demo/demo_report.json)。

notebooks/01_measurement_walkthrough.ipynb 的全部程式cell已直接執行；保存的notebook
無執行輸出。`uv build` 已成功生成wheel與sdist；configs/studies為可讀StudySpec，
configs/experiments為獨立協議schema。新增研究helper或文件後的本機測試與HPC舊
snapshot測試數量可能不同，請以各run的source hashes和JUnit為準。
完整CLI validate→measure→calibrate→export→network已執行：36項toy分數、4×3矩陣，
0失敗。最終WiC train錨點生成helper與probe網格調整由本機回歸測試驗證，
未為這些不改原生模型forward的變動重跑GPU。

`locks/*-observed.txt` 是實際環境的完整freeze快照，不是乾淨環境經resolver驗證的
通用安裝lock。HPC借用環境含未使用的vLLM與其舊TF依賴；正式全程建立乾淨依賴
環境時需要核對並重做smoke。程式不使用vLLM，計分checkpoint未微調。

## 尚未取得的證據

第二模型、embedding/reranker官方多模態recipe、真實公開資料批量計分、
PMPO本地改寫模型搜尋、所有held-out效度/變化/網路結果與成本scale sweep待下個
session執行。24GB只已證明足夠此demo預算，不保證所有8192-token或16幀配置。

補充合成刺激核對：兩模板平均後，red錨點在紅方形圖為−5.3543、藍圓圖為−9.1192；blue circle錨點分別為−20.4256、−9.4275。同一錨點跨圖變化符合刺激特徵，但藍圓圖內red的原始分數仍略高於較長的blue circle，不能把跨錨點raw排序直接叫語義準確率。這只是兩個合成例子的描述性核對。

最後另補tokenizer/Pillow/可選kernel版本與TF source revision進入backend identity，防止相同版本名下的處理差異命中舊cache；沒有改forward。该metadata更新由本機測試覆蓋，GPU歷史manifest仍保留原snapshot。
