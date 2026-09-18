# 下一個 session：VLanchor 完整實驗與論文交接

執行更新：本交接的研究協議繼續有效；原始demo交付狀態屬歷史記錄。
正式研究正在進行，最新已完成/待完成證據見[完整研究執行狀態](RESEARCH_STATUS.md)。

**任務邊界：本 session 建立工具、推導、Surrey GPU demo；下一個 session 跑完整研究。**
請先閱讀 [SPEC](../SPEC.md)、[DERIVATION_PACKAGE](../DERIVATION_PACKAGE.md)、
[VALIDATION](VALIDATION.md)。這裡的 E1–E6 是預先指定的研究協議，不是已完成結果。
既有 checkpoint、原始分數、失敗 run 與環境檔必須保留，不覆寫成「成功」。

本輪結果：Surrey作業44314 COMPLETED，Qwen3.5-4B/BF16原生89項計分0失敗；
該GPU snapshot167項測試通過，最終本機194項通過。shared-prefill超過0.01nat
門檻而停用；參照計分、完整logits對照及跨模態後狀態檢查通過。
可直接使用現有strict reference backend開展後續驗證。

## 1. 進場資訊與最小啟動

本機專案 `/Volumes/OutsourceData/VLanchor`。
Surrey SSH alias `surrey-aisurrey`；遠端專案
`/mnt/fast/nobackup/users/zw00924/VLanchor`，透過既有 gateway 設定連線。
本次以 Slurm `debug` 分配 1×RTX A5000 24GB；作業與結果詳见 VALIDATION。
本機 Python 3.11 `.venv` 可跑測試及 toy；正式 4B 推論在已分配 CUDA 節點執行。

```bash
cd /Volumes/OutsourceData/VLanchor
.venv/bin/python -m pytest -q
.venv/bin/python -m vlanchor --help
PYTHONPATH=src .venv/bin/python scripts/run_demo.py --backend toy --output runs/local-check
ssh -o BatchMode=yes surrey-aisurrey 'squeue -u "$USER"'
```

目前共享 home 接近配額，不能在其內下載模型或完整資料。Slurm demo 將模型與
環境放 `/var/tmp/$USER/vlanchor/$SLURM_JOB_ID`，只回傳小結果。這是 node-local
暫存，不能視為持久保存；正式全量資料需先確認專案/parallel scratch 配額與保留期，
記錄路徑後再配置批量任務。不要清理使用者別的專案或終止他人 GPU 作業。
詳見 [Surrey runbook](SURREY_RUNBOOK.md)。

## 2. 已有實作與下一步補齊項

| 能力 | 現有可執行內容 | 全程需要補齊/確認 |
|---|---|---|
| 核心計分 | strict Qwen native、單/多token、prefix/EOT、英文/中文、圖/圖文/影片 | 第二Qwen checkpoint真實重測 |
| 校準與追蹤 | 固定參照log-ratio/z、完整長表、cache、sidecar、缺失覆蓋率 | 大資料分片合併器與不可變run registry |
| 分析 | KMeans/PCA、獨立/配對來源群組bootstrap、energy/centroid shift與群組置換API、概念/樣本網路 | 正式資料上的抽樣/可交換性核查、邊bootstrap、GraphML/圖表批量匯出 |
| 資料 | 9組本地schema adapters、split/hash/缺媒體manifest | 取得真實資料、核對license與公開版本、凍結availability |
| 評估 | AP/MRR/VAD/retrieval、線性probe/grid、network比較 | 實驗driver統一切分、覆蓋率與多方法配對比較 |
| 基線 | 通用text mean embedding、text crossencoder、single-token MLM | 官方E5/Qwen embedding及reranker精確recipe；原生多模態baseline |
| PMPO式優化 | 快取矩陣子集搜尋、16候選上限、失敗回退、score/operator callback | 本地Qwen改寫callback、語義關係人工抽查、消融driver |
| 論文 | 核心方法/命題、RQ與claim-evidence表 | 所有正式結果、置信區間、圖表、結果驅動Discussion |

`configs/experiments/*.yaml` 是資料/協議配置，**不是 StudySpec，也不是自動全程
runner**。六個檔案按資料主題命名，與原定E1–E6的映射寫在其 README。
不要把讀入 YAML 當作「已完成實驗」。下一 session 應在既有 API 上建立少量明確
driver，先 dry-run 32 個 train/dev，再啟動正式任務。

## 3. 固定方法與模型

主模型 Qwen/Qwen3.5-4B：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
TF source `4815a0a6a064214f2d8208c094464a5a6b76ca8d`；實際版本5.18.0.dev0。
Python3.11；模型eval、thinking關閉、BF16、batch1、無padding/cache、原始full-vocab
normalizer。處理預算依SPEC；變更預算必須新measurement ID。

| 角色 | checkpoint | 執行要求 |
|---|---|---|
| 主計分 | Qwen/Qwen3.5-4B | 使用demo環境與固定revision |
| 同量級重測/改寫 | Qwen/Qwen3-VL-4B-Instruct | 先解析immutable revision、原生smoke |
| 多模態embedding | Qwen/Qwen3-VL-Embedding-2B | 按官方pooling、normalize與task instruction；原向量及同N anchor cosine |
| 近鄰相關性baseline | Qwen/Qwen3-VL-Reranker-2B | 官方yes/no相關性recipe，不能用generic mean pooling替代 |
| 文字embedding | intfloat/multilingual-e5-large | 正確query/passage prefix、mask pooling、normalization |
| FMAT | BERT-base、RoBERTa-base | 只重跑預選小部分masked條件，multi-token不假裝單mask |
| 可選獨立family | google/gemma-3-4b-it | 僅圖/文；新增adapter及同樣計分驗證後使用 |

各比較模型revision目前未凍結，下一 session 從官方源解析並保存到 model manifest，
不可默認main。不在這輪增加更大模型或微調。先後載入4B計分/改寫模型以控制顯存。
GPU並行由Slurm分配；不靠殺掉作業爭取記憶體。

## 4. 資料取得與切分凍結

| 資料 | 公開來源 | 角色与注意事項 |
|---|---|---|
| FMAT | [OSF](https://osf.io/5e2hr/) | stored scores先復現統計；不是新的人類標註 |
| ValueEval | [官方資料卡](https://huggingface.co/datasets/webis/Touche23-ValueEval) | 英文20值，多標籤；按conclusion分組 |
| EmoBank | [官方repo](https://github.com/JULIELab/EmoBank) | reader perspective，需reader ratings；emobank.csv是reader/writer混合 |
| Chinese EmoBank | [官方repo](https://github.com/NYCU-NLP/Chinese-EmoBank) | 中文VA；部分csv實為tab分隔，句子/文本分開 |
| OASIS | [原文](https://banaji.sites.fas.harvard.edu/research/publications/articles/2016_Kurdi_BRM.pdf) | 900圖VA；使用stimulus archive內image-level OASIS.csv |
| VIVA | [官方資料](https://huggingface.co/datasets/zhehuderek/VIVA_Benchmark_EMNLP24) | image+action；每樣本正負value短語候選 |
| WiC | [官方網站](https://pilehvar.github.io/wic/) | train建一般錨點及固定參照；同/異義開發 |
| DWUG/SemEval | [官方資源](https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/wugs/) | 真實usage pair人工相關/歷時變化 |
| VATEX | [官方網站](https://eric-xw.github.io/vatex-website/) | 影片/中英文描述；檢索與影片語义能力 |

資料下載明確執行；不把全部原始圖片/影片重新散布到工具包。保存取得日期、release、
原始來源、license、sha256、可取得/缺失ID、重複群組。先建立availability再固定選樣，
不能按模型表现篩圖或影片。OASIS image-level檔在[stimulus archive](https://osf.io/download/uxvpb/)，
不要把另一 participant-level 檔誤餵進image-level adapter。

官方train/dev/test優先保留；沒官方切分才按來源/重複群組hash(seed42)做60/20/20。
hash為期望比例，非精確固定個數。檢查同圖片、場景、conclusion、文檔或目標詞
跨切分，保存 crossings；不能無聲改官方test。若需額外group-disjoint研究，另命名
secondary split，不與官方分數混寫。reference只在train/external fit。
逐檔adapter只能檢查當次載入的材料；ValueEval等資料跨官方檔案的conclusion crossings
必須由driver合併各split manifest後再核查，不能把單檔的空crossings清單當成全庫無洩漏。
EmoBank reader使用`rating_perspective='reader'`並提供`individual_ratings_path`，
按作者規則刪除全1評分、保留有效N>1、按ID均值round(2)，再join官方text/split；
也可用已發布reader.csv作`ratings_path`。被排除的ID及其原因保存在manifest。

資料 preparation 範例（YAML的adapter_kwargs由實際路徑填入）：

```yaml
dataset: valueeval
adapter_kwargs:
  arguments_path: ../data/valueeval/arguments-training.tsv
  labels_path: ../data/valueeval/labels-training.tsv
  split: train
```

```bash
python -m vlanchor prepare --config configs/prepare-valueeval.yaml --output data/prepared/train
python -m vlanchor validate --config configs/study.yaml --samples data/prepared/train/samples.json
python -m vlanchor measure --config configs/study.yaml --samples data/prepared/train/samples.json --output runs/train.parquet --cache runs/cache
```

正式driver將 train 分成不重疊群組：reference、橋接搜尋約256樣本、其餘訓練。
對每split寫出明確的samples.json/labels.csv，不能用全表宣告train繞過防洩漏。

## 5. 錨點與橋接凍結

`values20_en`：ValueEval官方20個label作stable ID/canonical surface；不要憑簡化
翻譯合併power、security、conformity的子類。`affect12_en/zh`：joy/悲傷等12情緒
（見configs/anchors）；兩種語言獨立驗證。
`general128_en`：只由WiC train目標詞計數、頻次降序/字串破同分建立；64/256為
同一完整列表的前綴，凍結其hash後才接觸DWUG評估。不得用DWUG變化標註挑anchors。
已實作`datasets.build_general_anchors(wic_train_bundle, n=128)`，每個WiC pair只計數一次，
按(-頻次, 原始lemma字串)排序；拒絕非WiC、未明確宣告train或詞彙不足的材料。
呼叫者保存來源bundle manifest及生成pack；64/128/256前綴的ID保持一致。
`datasets.general_anchors()`仍是固定24個一般概念的legacy/default fixture銀行，不能冒稱general128。

每關係固定三個橋接；seed pool八個，所有精確換行、空格與Unicode原樣保存。
associated_with測關聯；expresses_value測表達；endorses_value测支持，不能混平均。
reader情緒用evokes_emotion；DWUG用means_in_context且輸入target span。
時間只作分組metadata，不放進DWUG提示。模板語義審查在dev/test之前完成。

## 6. E1–E6完整實驗清單

### E1：計分與既有方法銜接

先在FMAT公開stored scores重現预選職業/名字/關係分析的有符號相關，保存原始欄位
與方向定義，再小樣本BERT/RoBERTa重跑。比較anchor-only與明確命名的完整句PLC。
式logp(Pad)=logp(P)+logp(a|P)+logp(d|Pa)說明後綴項；不要把兩種分數等同。
不取|r|掩蓋方向問題。FMAT歷史年份聯想不代替真實歷時語料。
輸出E1_scores、reproduction_table、signed_correlations、token_alignment_audit。

### E2：外部效度与增量價值

ValueEval（expresses_value主設定）、EmoBank-reader、Chinese EmoBank、OASIS。
比較raw、reference_log_ratio、reference_z、無內容、樣本-標註打亂、原embedding、
同N anchor cosine、reranker。情緒資料驗證情緒概念，不能寫成抽象價值準確度。

直接測量：ValueEval逐錨點macroAP及每樣本內AP分開報；VAD/VA用Spearman。
**affect12到V/A/D的直接投影需在train/dev前確定方向與權重**，目前12個概念
不是天然3個維度。建議新增固定dimension anchors（如pleasant/unpleasant、
aroused/calm、dominant/submissive）作明確命名補充，不任意事後挑12維最大相關。
若主affect12只透過線性模型映射VAD，其結果必須稱預測效度。

固定特徵+probe：依已批准方案及現有API預设，OVR logistic C∈{.01,.1,1,10}；
Ridge α∈{.01,.1,1,10,100}；分別以dev macroAP及dev MSE選擇。
所有表徵同切分/同預算；scaler在train fit，dev選超參，test一次。
參照校準固定移動不能改善逐錨點AP；用每樣本內排序評估跨錨點影響。
輸出direct/probe兩張表、配對method-difference CI、覆蓋率、undefined anchors。
現有macroAP明確排除零正例anchor；每樣本內AP預設要求候選中同時有正負例。
固定評估ID集合並報排除清單，不能讓不同方法用不同的有效分母而不揭露。

### E3：測量敏感度與PMPO式延伸

單橋接/固定三句/優化三句；詞形、短語長度、prefix/EOT；general64/128/256；
重複/無關錨點；兩個4B；raw/log-ratio/z與應有的數學不變性。
昂貴交叉消融最多256個事先hash抽定test材料，不依分數挑選。
固定baseline共同切分，至少包含與預先宣告相同錨點語義的別名重測。

reference_guided：train按conclusion分組抽約256；至少5正5負的anchors納入目標。
另外不重疊train群組fit reference。8seed→3，2輪×4，最多16不同計分嘗試。
每輪從既有3+新增4的35個三元組選；快取矩陣，不因組合數重跑模型。
J=.8V+.2R；同分詞面多樣性再ID。改寫Qwen3-VL-4B只看橋接和關係定義，
不看評估材料/標籤實例；seed42,temp.7,top-p.9,max_new_tokens128。
規則拒絕答案/anchors/controltokens/關係變更；語義等價不能只靠字串規則證明。
比較固定三句、等預算隨機改寫、只效度、只可靠度、α目標。
dev僅評選出的最終集合，不反饋下一輪；test最後一次。
輸出完整搜尋trace、有效anchor覆蓋率、失败回退、held-out效度與可靠度。

### E4：多模態資訊利用

VIVA value任務輸入 **image+該樣本標註的correct action**；action是這個任務的已知
條件，不輸入reason、value label或答案解釋。每樣本公開候選概念名作主評估；
目前adapter去除冒號後解釋，帶解釋長版本需另建明確命名消融。對照同action去圖/錯圖。稱為
action-conditioned value ranking；AP/MRR按樣本候選比較，不拼共同概念矩陣。
換成另一action時不能沿用原action的value標註。另有`load_viva(task='actions')`支援
原始action選擇：輸入只含image，action文字作候選，correct action僅留在評估標籤。

一般圖文分別I、T、IT、錯配IT。固定空輸入參照可計
δ=s(I,T)−s(I,T0)−s(I0,T)+s(I0,T0)，名稱為輸出分數模態交互；
不從此識別內部融合機制。配對樣本共同過預處理方可比較。

VATEX固定可取得官方validation子集目標500video，先availability、固定排序。
video輸入不能含待檢索caption。英/中文caption各自檢索，Recall@1/5/10；
明確區分「至少一個命中率」與「相關項目召回比例」。1/8/16幀改成新處理ID。
如validation還用於調方法，劃出不重疊dev與最終evaluation子集並預先凍結。
輸出配對模態增益、錯配差、影片檢索、缺失/下載失敗清單。

### E5：semantic shift

區分整體概念表徵漂移和目標詞的上下文詞義變化。主實驗DWUG固定target，
使用上下文+span，WiC train產生一般anchors/參照，不按時期重新標準化。
每詞輸出centroid distance與精確分塊V-statistic energy distance。
與人工變化程度排序Spearman；usage-pair相关另報，不把shift結果替代pair驗證。
負對照同一期內拆分、符合可交換性的時期置換、樣本量配平。以目標詞或來源
群組bootstrap，不將tokens/templates作獨立樣本。SemEval trial隨機展示標籤不用。
WiC/DWUG adapter的`group_id=lemma`用於防切分洩漏；它不等於單一目標詞內的
獨立來源單位。單詞內shift CI需顯式傳入真實文檔/來源`sampling_units`；若沒有可識別
單位，保留CI未定義，不把一個lemma cluster偽裝成多個獨立樣本。跨詞變化相關的CI
則以目標詞重抽樣。
輸出每詞距離、群組CI、人工比較、配平/置換/樣本量敏感度。

### E6：semantic network與分群

concept graph節點=anchor，邊=跨材料帶符號Pearson；保存完整矩陣。
ValueEval同批人工多標籤矩陣作參照。比較off-diagonal邊權Spearman、固定密度
邊集合重疊及source-bootstrap穩定性。圖稀疏化只作展示。
常數標註/模型欄的Pearson邊未定義；比較API要求有限矩陣，driver應對兩種網路使用
同一個明確的有效節點遮罩並報移除ID，不能把未定義邊填0。
負對照各錨點欄獨立置換；共同置換樣本行是不變性檢查。

sample graph節點=材料/usage，邊=固定表徵相似性，DWUG人工relatedness驗證。
KMeans的k由使用者/開發協議明確指定；PCA僅展示；模板間ARI/群組穩定性另報。
不把2D可視分離當外部效度。輸出full adjacency、edge CSV、GraphML、bootstrap
邊頻率、same-node-order人工對照、分群參數與模板穩定性。

## 7. 統計、成本與失敗判準

seed42；bootstrap1000、95% CI、BH-FDR q=.05。按文檔/來源群組/target詞抽樣。
同一批樣本的方法差採配對重抽樣；時間置換需可交換；只有一個cluster的CI未定義。
樣本ID/anchorID在評估前顯式join，CSV行欄順序不可決定數值。
模型比較取共同可處理樣本並報每模型及共同覆蓋率；缺失不填0。
公開資料可能進入預訓練，普通train/test不能解決模型訓練污染。

主要經驗通過條件：保留人工標註上的關聯超過打亂對照。若某模態/構念不成立，
限制claim並分析失敗；不以分群圖補足。不預設擊敗embedding；報准确度、可解釋性、
可靠度與成本取捨。校準不變指標相同是驗算，不是負面結果。

效率按N=16/64/128/256記錄：預處理/forward/總時間、峰值GPUallocated/reserved、
樣本數、bridge數、score items、anchor token分布、cache hits。原始參照路徑約
M×B×N次forward；先實測32樣本，再估完整GPU小時與排隊預算。這輪不預承諾時長。

## 8. 下一 session 的執行順序與退出条件

1. 核對本次VALIDATION與既有失敗；本地測試→Qwen第二模型/官方基線32項GPU smoke。
2. 確認scratch容量，取得/核對資料，凍結availability、split、anchors、bridges、revision。
3. 建立分片driver與run registry；32個train/dev端到端，核對人工標註未進prompt。
4. E1+E2固定橋接dev；檢查非有限/退化座標、模態/reader設定、成本，不接觸test調方法。
5. 執行E3 train-only搜尋與消融；dev只評最終候選，凍結所有方法選擇。
6. 凍結manifest後執行E1–E4正式評估及E5/E6；任何設定變動新建run ID。
7. 表格由結果檔生成，不手填數字；配對CI/FDR/覆蓋率和失敗同步報告。
8. 依第9節完成論文與重現包；检查每個claim是否有可定位的實驗或推導證據。

停止條件：缺GPU allocation、無足夠存儲、模型/processor邊界或selected-logits
驗證失敗、資料切分洩漏、零方差被強行填值。加速驗證失敗只停用加速，保留參照。
不自動減幀/降解析度/換模型/換CPU。修復需保留失败run並新建版本，不覆寫证据。

## 9. 論文故事線與必交圖表

暫名 **VLanchor: Probabilistic Anchor Representations for Multimodal Measurement**。
問題是研究者需要任意概念、不同模態、可追蹤的測量工具；核心貢獻是明確條件事件
和可重用分析鏈，而非宣稱讀取真實內部心理。FMAT支持命題測量框架；PMPO把探針
集合視為可優化工具；VLanchor延伸至輸入條件化、任意N座標、多模態與經驗分析。

| 論文段落/RQ | 要證明的事 | 證據 |
|---|---|---|
| Formulation | 事件與向量定義可驗算、邊界清楚 | DERIVATION D1–D10與核心測試 |
| RQ1 validity | 座標對材料概念差異有外部效度 | E2人工標註/無內容/打亂/強基線 |
| RQ2 sensitivity | 模板/詞形/先驗/模型如何影響測量 | E3、不變性及每樣本內排序 |
| RQ3 modality | 視覺與聯合輸入有有效增量信息 | E4去圖/錯圖/檢索 |
| RQ4 applications | 表徵支援可驗證shift/network | E5/E6人工變化與標註網路 |
| Discussion | claim在哪些語言/構念/模型成立 | 共同覆蓋、失敗案例、污染限制與成本 |

必交九類圖表：方法流程；model×modality×language×dataset覆蓋；直接/probe效度；
校準不變性與within-sample排序；固定/優化橋接；模態移除/錯配；shift人工參照；
概念網路穩定邊；N/token長度/模態成本。
paper/manuscript.md 提供可接續的寫作骨架；結果未知處明確待填，不虛構改進數字。

最終重現包至少包含程式版本、lock、取得腳本/來源條款、資料ID/hash/split、所有
study配置、score長表、calibration、矩陣/網路、統計與畫圖腳本、原始日誌、失敗清單、
搜尋trace、run manifest、表格來源mapping、論文與引用。只對授權資料作可分享打包。


## User amendment: 2026-09-11 video demonstration scope

The user reduced video work to small-sample capability demonstrations and requested SciencePlots figures. This amendment supersedes the full VATEX retrieval and full frame-evaluation requirements above. Retain original protocols and outputs as historical records; do not describe a demo as held-out scientific validity. All non-video experiments remain required. See `research/contracts/user-video-scope-amendment-20260911-01.json`. Pending full-video jobs have been withdrawn; already-running development shards may finish with evidence retained.
