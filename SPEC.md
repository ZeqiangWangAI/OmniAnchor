# OmniAnchor v0.1：工具與測量契約

狀態：實作與 demo 已交付，完整外部效度研究正在執行，尚未全部完成；最新證據見
[研究狀態](docs/RESEARCH_STATUS.md)。這份文件與
[數學推導](DERIVATION_PACKAGE.md)、[完整交接規格](docs/HANDOFF_SPEC.md)、
[Surrey 執行手冊](docs/SURREY_RUNBOOK.md) 共同構成研究規格。
實際測試證據見 [VALIDATION](docs/VALIDATION.md)。合成 demo 不提供構念效度證據。

## 1. 目的與範圍

固定模型、輸入處理、橋接關係與精確錨點字串，把任意材料映射成 N 個條件事件的
log probability。錨點可為價值、情緒、行為或一般概念；不要求成對、互斥、正交或
獨立。N 個分數不作 softmax，提示不包含候選列表。新增或重排錨點不得改變已有
錨點的原始分數。相同字串的別名保留為不同 ID，但須提示其幾何重複加權。

本工具測量模型對材料的概念關聯，不直接觀察材料的「真實內部價值」，亦不把
模型輸出當作人類心理真值。`associated_with`、`expresses_value`、
`endorses_value`、`evokes_emotion`、`means_in_context` 分別建 study；同一集合
只聚合相同關係與語言的改寫。支持英文、中文及使用者自定語言，不自動翻譯、
對齊或宣稱跨語言分數可比。

本次交付：Python API/CLI、嚴格計分與媒體處理、校準、可靠度、聚類/變化/網路、
本地公開資料適配器、bounded 橋接句子集搜尋、測試、Slurm demo、論文實驗規格。
音訊、微調、多 token cache 分支、完整 benchmark 自動調度不在本次執行範圍。

## 2. 不可更改的計分政策

1. 官方 chat template 渲染 user→assistant，`add_generation_prompt=True`、
   `enable_thinking=False`，之後直接追加精確 bridge prefix；保留末尾換行。
2. 對 prefix 和 prefix+anchor 使用同一份凍結媒體；文本 token 序列不 padding、不截斷、
   `add_special_tokens=False`。完整 IDs 必須以 prefix IDs 開頭，否則
   `BoundaryError`。不得分別 tokenize 後拼接以繞過驗證。
3. `token_prefix` 計算完整錨點 token 鏈，預設不計 EOS。
   `turn_terminated` 另附精確 EOT，事件寫入 manifest；兩者不混合。
4. 位置 t 的 token 由 t−1 的 logits 預測；完整詞表 log-softmax 使用 FP32，
   token logp 求和使用雙精度。使用模型原始 logits，無 temperature/top-k/top-p。
5. 模型 `eval()`、inference mode、batch=1、無 padding、`use_cache=False`。
   僅保留必要預測位置的 logits，不能縮減詞表 normalizer。
6. 保存 `raw_logp`、`token_count`、`mean_token_logp`。terminated 事件的
   token_count 包含終止 token；另保存 `anchor_token_count`。mean-token 分數
   只是長度診斷，不是完整事件的 log probability。
7. 錨點字串不 strip、不改大小寫、不做 Unicode 正規化；純空白字串與保留控制 token 拒絕。
   含非空白內容的錨點保留其前導、內部與尾隨空白。
   單一候選錯誤只標記该候選；媒體/前綴失敗標記整個 sample×bridge；
   資源錯誤停止執行並保留已產生的可追蹤結果，不能吞掉 OOM 後反覆嘗試。

## 3. 公共資料契約與 API

實際型別在 `src/omnianchor/types.py`；Pydantic extra fields 禁止。

| 型別 | 欄位與約束 |
|---|---|
| Part | type=text/image/video；text 或本地 path；video 可指定 clip_start/end |
| Sample | id、ordered parts；language/source/group_id/time/pair_id/metadata；可選 target |
| TargetSpan | part_index、start/end（Unicode 字元索引，左閉右開）、lemma |
| Anchor | stable id、精確 surface、language、可選 concept_id |
| Bridge | id、prefix、relation、language；不同關係或語言不能同 study 平均 |
| StudySpec | name、ModelSpec、anchors、bridges、event、ResourceProfile、seed、system_prompt |
| ScoreTable | pandas 長表 + manifest；每 sample×anchor×bridge 一列，包括失敗 |
| CalibrationArtifact | 每 anchor×bridge 的參照統計、有效狀態、參照與設定來源 |
| MeasurementMatrix | float64 values、sample_ids、anchor_ids、variant、manifest |

```python
from omnianchor import measure, score, fit_reference, transform, to_matrix
raw = measure(samples, spec, cache="runs/study/cache")
# reference_raw 必須是獨立允許的 train/external 材料。
calibration = fit_reference(reference_raw)
adjusted = transform(raw, calibration)
matrix = to_matrix(adjusted, variant="reference_z", missing="error")
```

`score(samples, anchors, bridges, backend, ...)` 供明確注入 backend；
`measure(samples, spec, backend=None, ...)` 從 StudySpec 建立 backend。
`transform` 預設同時計算 reference_log_ratio/reference_z，也可用 variant 只計算其中一種；
`to_matrix` 等權平均固定且完整的模板集合。
`missing` 明確可選 error/drop_samples/drop_anchors，輸出覆蓋率；不能填零。

分析：`cluster(matrix,k=...,seed=42)`、`compare_groups`、`semantic_shift`、
`semantic_network(kind='concept'|'sample')`、`analysis.pca`。
`audit_reliability(scores)` 輸出每錨點跨模板 Spearman 與 α 診斷；
`reliability.template_radii` 給有限模板的敏感度半徑。
目前分析結果為可序列化 dict，不假裝已實作所有計畫中的命名 Result 類。

## 4. 校準與可比性

每一 anchor×bridge，以固定參照集計算：

\[
c^{mix}=\operatorname{logsumexp}(s_R)-\log|R|,\quad
\mu=\operatorname{mean}(s_R),\quad \sigma=\operatorname{std}(s_R,ddof=1).
\]

`reference_log_ratio=s-c_mix`；`reference_z=(s-mu)/sigma`。
不得把 mean(logp) 當作 log(mean(p))。零方差/不足兩項只使 reference_z 未定義，
輸出缺失與原因；raw_logp 與 reference_log_ratio 仍可計算。
使用缺失 z 的 to_matrix 預設報錯，除非明確選擇刪除政策。
不得用 epsilon 隱藏不可估計的參照尺度；參照零方差不證明該概念在所有輸入上均無信息。

參照 fit 拒絕已標記 dev/test 材料；若 split 完全未標記且未傳入 split 參數，
目前 API 記錄 unspecified，不自動拒絕。正式研究仍須明確宣告並驗證 train/external，
不能把 unspecified 當成資料隔離已通過；參照與待測材料的實際獨立性由研究協議保證。
這是資料契約檢查，無法偵測被使用者錯標的資料或預訓練污染。
同一參照必須用於比較的群組、模態與時期；不用各組獨立 z-score 假裝對齊。
校準 apply 按 ID 檢查模型/事件/處理資源、錨點精確字串與橋接定義；允許單純重排，
不能以相同 ID 替換定義。固定逐座標平移不改變歐氏幾何，以及跨材料計算的錨點欄
Pearson/Spearman 相關網路；這些量不能用來宣稱基線改進。

## 5. 模型、媒體與資源

主模型：`Qwen/Qwen3.5-4B`，權重 revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
Transformers source commit：`4815a0a6a064214f2d8208c094464a5a6b76ca8d`
（該提交的版本為 `5.18.0.dev0`，不能用任意 `transformers>=5` 代替正式鎖定）。
第二個原生 adapter：`Qwen/Qwen3-VL-4B-Instruct`，需下一 session 鎖定 revision
並完成真實 checkpoint smoke；目前不能聲稱已實測第二模型。
不會載入未列入 adapter 的任意架構。正式模型僅 CUDA/BF16 或明確 FP32。
ToyBackend 和注入小模型只供 CPU 邏輯測試。

| 預算 | 預設 |
|---|---:|
| image min/max pixels | 65,536 / 262,144 |
| video frames / 每幀 max pixels | 8 / 65,536 |
| input_text_tokens | 2,048 |
| anchor_continuation_tokens | 32 |
| total_postprocessor_tokens | 8,192 |
| implicit truncation | false |

官方 processor 保留長寬比 resize。影片使用 PyAV 解碼；保存真實 fps、PTS、
均勻取樣索引與解碼幀 hash。缺失真實時間資訊報錯，不編造 fps。
temporal padding 與已重複採樣幀明確記錄。Qwen video 像素預算涉及時間維，
adapter 按實際 processor 版本處理並檢查展開後 grid，而非盲傳每幀預算。
校驗 pixel tensors、grid、模態 token、parts 順序及總 token 預算。
模型內持久 M-RoPE 狀態在各次參照 forward 前重置。

共享單 token prefill 為 opt-in：首次比較同模型 teacher-forcing，門檻每 token
0.01 nat；不通過即停用。多 token cache 未啟用，因 Qwen3.5 混合注意力的
KV/convolution/recurrent state 不能 shallow copy 或用簡單截短還原。
降低解析度、減幀或截斷不是 OOM 的靜默重試策略，必須形成新設定與測量 ID。

## 6. 追蹤、IO、快取

每列至少含 sample_id/anchor_id/bridge_id/relation/event/raw_logp/token_count/
status/model_id/model_revision/measurement_id。可選 token_ids/token_logps；
不存整個詞表 logits。長表 Parquet，配 `.manifest.json`；矩陣 NPZ 或 Parquet；
設定/校準/診斷 JSON，非有限值以 null 保存，不寫非標準 JSON NaN。

Cache key 包含模型及處理身份、精確輸入/bridge/anchor/event、媒體內容 SHA256、
資源預算。實作保守地也包含樣本 ID/metadata/本地路徑；更名可能失效，
同路徑檔案內容改變必須失效。每候選独立 cache，
只快取成功項。保留編譯與處理 metadata，冷/暖 cache 產生相同分數與成功狀態。
快取並非免驗證的可信數據庫；手改 artifacts 會破壞研究可重現性。

每正式 run 額外保存 source hashes 或 git commit、資料 ID/版本/split/hash、
Python/Torch/CUDA/Transformers/tokenizer/processor、precision、batch=1、
token/pixel/frame 預算、seed、GPU/node/Slurm ID、完整命令、成本與失敗覆蓋率。
未訓練模型，gradient accumulation/training optimizer 記為 not applicable。
demo Slurm script 先複製 source snapshot，再運行，避免工作區更新污染運行中實驗。

## 7. 橋接優化契約

預設關閉；名稱 `reference_guided`。模型/錨點固定，僅同語言/同關係橋接改寫。
API `optimize_bridges(..., score_bridge=callback, operator=callback)` 快取訓練矩陣。
CLI 僅對已計分 train pool 作子集選擇；完整 Qwen 改寫 callback 待下一 session 接入。
研究預設初始8句、選3、2輪各最多4句；總最多16次不同橋接的計分 callback 嘗試，
分數無效或 callback 例外也占預算。規則提前拒絕與重複候選不再次計分。
J=0.8×macroAP+0.2×模板排序一致性；同分比較字面多樣性再 stable IDs。
只納入至少5正5負的錨點；常數/無效可靠度明確處理，不當成滿分。
規則拒絕控制 token、重複、錨點答案洩漏、關係/語言變更；規則不是語義等價證明。
子代生成或計分失敗保留上一個有效集合並給原因。初始固定橋接違反規則直接拒絕；
初始有效計分不足或所有可選子集可靠度未定義時回傳非 ok 狀態，不能宣稱優化成功。
參照/搜尋 conclusion 群組不重疊，
dev 僅評最終候選，test 最後一次；測試不得用作下一輪搜尋訊號。

## 8. 資料、分析及驗收

9 組本地適配器與精確資料注意事項見 `configs/experiments/README.md`。
labels 與輸入分離；VIVA 每樣本候選保留長表，不冒充共同本體矩陣。
E1–E6、模型基線、公開切分與論文圖表的完整執行要求見 HANDOFF_SPEC。

compare_groups/semantic_shift 接受 sampling_units 序列或按 sample ID 的字典；預設
取樣本 group_id（也兼容 metadata.group_id），缺失時回退 sample_id。這個回退假定相應樣本獨立，
不能代替真實來源分組。每次抽中一個來源組，保留該組全部材料；兩側來源集合相同時
共享抽樣次數以保留配對，完全不相交時分別抽樣，部分重疊則明確拒絕。
點估計仍按材料行加權，不自動改成等權來源均值。預設 bootstrap=1000、seed=42；
輸出材料數、有效來源組數及實際重抽樣次數。任何一側少於兩個有效來源組時，
置信區間及 semantic_shift 的可選 permutation p 值未定義。距離的百分位 bootstrap 區間不保證在零變化
邊界或極少來源組時具有名義覆蓋率，不能以「區間不含0」替代符合可交換假設的組置換檢驗。
semantic_shift 逐目標處理恰好兩個時期；它不自動重抽目標詞來估計跨詞 benchmark 指標。

必過：機率 shift、前綴重分詞、Unicode、錨點增刪重排/坏候選隔離、校準統計與
零方差、資料隔離、平移/z 不變性、重複幾何權重、媒體預算/真實 fps、狀態隔離、
快取、行/欄置換網路、均值不變分布變化反例、CLI 依 ID 對齊與 IO roundtrip。
FP32 toy 絕對誤差 1e−6；BF16 加速每 token 0.01 nat（序列乘 token 數）。
通過軟體驗收不等於通過外部效度。完整 benchmark 必須與打亂對照、人工標註及
embedding/reranker 比較，不以可視分群替代證據。
