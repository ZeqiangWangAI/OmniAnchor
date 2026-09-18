# VLanchor：數學推導與論文命題

狀態：**COHERENT_AFTER_REFRAMING**。可識別的目標是「固定模型條件下，材料使
指定錨點 token 事件更/不容易出現的程度」。材料的模型外真實價值取向仍需外部效度。
以下區分定義、恆等式、條件性命題與經驗設計，不把工程門檻寫成統計定理。

## D1. 對象、前提與符號

固定自回歸模型 M（含 tokenizer、processor、weights 版本）、橋接 b、輸入 x、
精確錨點字串 a_i。P(x,b) 是 user→assistant 官方模板加橋接的完整前綴。
T_M 是包含媒體展開的編碼程序。令 p=T_M(P)、u=T_M(P+a_i)。
假設 u 的前 |p| 個 IDs 與 p 完全相等，且媒體與模態位置相同。
若不成立，所選完整編碼路徑無法分解成宣告前綴加續寫，工具報 BoundaryError；
這不表示錨點的所有可能續寫路徑機率為零。

續寫 c_i=u[|p|:]，長 L_i>0。詞表分布為模型原始 logits 的完整 softmax，
無採樣截斷、無溫度變更。所有 log 使用自然對數，單位 nat。

## D2. 完整 token 事件（鏈式法則）

\[
s_{i,b}(x)=\log p_M(c_i\mid p)=
\sum_{t=1}^{L_i}\log p_M(c_{it}\mid p,c_{i,<t}).\tag{1}
\]

Teacher forcing 輸入 p+c_i；目標序列位置 j 的機率取 logits[j−1]。
候選內容可影響後續候選 token，但不能回流影響已先出的 token（因果遮罩）。
不使用模型預設平均 loss 當 log sequence probability。
式(1) 是一條確定 token 路徑的前綴事件，不是回答任意位置包含該詞，
亦不是所有可解碼為相同字串的 tokenization 路徑機率之和。

若另要求當前 assistant turn 結束：
\[
s^{end}_{i,b}(x)=s_{i,b}(x)+\log p_M(EOT\mid p,c_i)\le s_{i,b}(x).\tag{2}
\]

若 c_i 是 c_j 的 token 前綴，E_j⊆E_i，因此 p(E_j|p)≤p(E_i|p)。
freedom 與 freedom of speech 這類短長錨點，只有在實際 IDs 滿足此前綴關係時，
才有上述必然排序；不能把它解讀成較短概念比較「語義接近」。
其他長度效應仍需另外檢查；m=s/L 是平均 token logp 診斷。

完整句 PLC 與 anchor-only 的關係：令 m 為固定的外部媒體等條件，
P、a、d 為邊界相容的文字 token 串，則
\[
\log p(Pad\mid m)=\log p(P\mid m)+\log p(a\mid P,m)+\log p(d\mid Pa,m).\tag{3}
\]
對不同 a 的比較仍含後綴相容性，除非後綴項相同，不等價於式(1)。

## D3. 從橋接事件到任意錨點座標（定義）

對同一關係下 B 個改寫，\(\bar s_i=|B|^{-1}\sum_b s_{i,b}\)，
\(\phi_{raw}(x)=(\bar s_1,\ldots,\bar s_N)\)。
exp(mean logp) 是幾何平均。橋接句不獨立，不可解讀成 B 次獨立證據相乘。
不同錨點不是互斥完備類別，不作跨 N softmax；否則增加錨點會改變全部座標。

## D4. 兩種參照校準（不同統計量）

固定允許的參照 R={r_k}_{k=1}^K，各座標定義
\[
c^{mix}_{i,b}=\log\left(K^{-1}\sum_k e^{s_{i,b}(r_k)}\right),\qquad
\ell_{i,b}(x)=s_{i,b}(x)-c^{mix}_{i,b}.\tag{4}
\]

這是相對參照輸入混合分布的 log probability ratio。數值計算用 logsumexp。
另一統計量是 logp 的均值 μ 與樣本標準差 σ（ddof=1），
\[
z_{i,b}(x)=(s_{i,b}(x)-\mu_{i,b})/\sigma_{i,b},\quad
\phi_z=|B|^{-1}\sum_b z_b.\tag{5}
\]
σ=0 或 K<2 時 z 未定義；不得以 epsilon 構造虛假的有效座標。
由 Jensen 不等式，c_mix≥mean(s_R)，通常严格大於。
例 s_R=(log .1,log .9)：c_mix=log .5；mean(s_R)=log .3。

**條件性 Bayes 解釋。** 人為構造聯合分布 q(x)p_M(Y|x,b)。設 A_i 是该聯合空間
中定義一致的可測事件；若候選 token 鏈固定，它就是固定的續寫事件。
若編碼因 x 改變，則需明定 A_i={(x,y):y 以前綴 c_i(x,b) 開始}，不能冒稱為
不依賴輸入的共同 token 鏈。要求 q(x)>0 且 p_q(A_i)>0；連續情況的 q(x) 指密度，
條件關係在 q 幾乎處處的意義下成立。此時
\[
\log\frac{p_M(A_i\mid X=x,b)}{\mathbb E_q p_M(A_i\mid X,b)}
=\log\frac{q(x\mid A_i,b)}{q(x)}.\tag{6}
\]
依 Bayes，q(x|A_i,b)=q(x)p_M(A_i|X=x,b)/p_q(A_i|b)，相除即得。
此解釋要求編碼事件在 q 的支持上有定義；BoundaryError 不是概率零。
工具可保存 token IDs 來核查固定 token 鏈前提，僅 token count 相等不足以證明它。
若 q 取有限參照集的經驗分布，新輸入 x 在其支持外時，右側不能直接使用經驗 posterior
比值。式(4) 仍是可計算的參照分數；要援用式(6)，須另指定支持覆蓋 x 的總體 q，
並把參照平均視為對其邊緣概率的估計。式(6) 不涉及人類價值標籤的真實 posterior。

## D5. 模型偏差不可識別性（命題與證明）

若觀察 s_M(x)=v(x)+β_M(x)，任取 g(x)，令 v'=v+g，β'_M=β_M−g，
則 v'+β'_M=s_M。故只觀察模型輸出無法唯一識別 v、β。
對多模型也可同時把同一 g 加入 v 並從全部 β_M 減去；多模型不自動識別真值。
基線差分為
\[
s_M(x)-s_M(x_0)=[v(x)-v(x_0)]+[β_M(x)-β_M(x_0)].\tag{7}
\]
只有輸入無關等額外假设能消除相應偏差部分。參照校準、模型敏感度與外部效度
必須分開報告；校準不是「消除 LLM 偏見」的證明。

## D6. 幾何與評估不變性（恆等式）

令 X'=X−1cᵀ。同一樣本對差分 (x−c)−(y−c)=x−y，因而：

* 歐氏距離、加權歐氏距離、k-means 目標及最優解集合不變。
* 中心化 X' 等於中心化 X，故 PCA 協方差與主子空間不變
  （重根/符號不唯一不屬算法錯誤）。
* 錨點間跨樣本 Pearson/Spearman、共同基線的兩期均值差不變。
* 逐錨點 AP/AUROC/Spearman 不變；跨樣本×錨點 flatten 的 microAP 與
  每樣本內 AP 不在此不變性之內。

若 α_i>0、參照標準差 σ_i>0，且在相同 reference 樣本上，對變換後的分數使用
相同 ddof 重新計算統計量（或等價地變換已有 μ、σ）：
\[
z(α_i s_i+β_i)=(α_i s_i+β_i-α_i μ_i-β_i)/(α_i σ_i)=z(s_i).\tag{8}
\]
故 **當每個 anchor×bridge 的 L_i 在 reference 和評估樣本間固定**，z(s_i/L_i)=z(s_i)，
且 z(s_i−c_i)=z(s_i)。若 token 長度因上下文邊界改變，固定缩放前提不成立；
不能直接套用此式。這裡 L_i 必須是當前 event 的 token_count，terminated 模式包括 EOT。
零參照方差時等式兩側均未定義，不是兩個有效的相同零向量。
每橋接先 z 再平均符合式(5)；先平均再 z 是另一個統計量。
Cosine 對平移不具有不變性，需要獨立檢查參照的影響。

## D7. 加權距離與核（條件性命題）

固定 W=diag(w_i)，w_i≥0，令
\[
d_W(x,y)=\|W^{1/2}(\phi(x)-\phi(y))\|_2,
\quad k_W(x,y)=\phi(x)^T W\phi(y).\tag{9}
\]
任意 γ，\(\sum_{ij}γ_iγ_j k_W(x_i,x_j)=\|\sum_iγ_iW^{1/2}\phi(x_i)\|^2≥0\)，
故為 PSD kernel。非負權重給表徵上的 pseudometric；若所有權重正，對向量是 metric，
但原始材料到向量不保證單射，對材料仍可只是 pseudometric。
重複一個座標相當於增加該方向的平方距離權重，不是增加獨立信息。
固定權重需事先指定；若用標註學習 W，必須納入 train/dev 防洩漏流程。
加權式是數學擴充；v0.1 的距離、聚類、shift 及半徑 API 使用 W=I，未提供任意 W 參數。

## D8. 有限模板敏感度界限（命題）

令 \(r_x=\max_{b\in B}\|\phi_b(x)-\bar\phi(x)\|_W\)。反三角不等式及三角不等式給
\[
\left|\|\phi_b(x)-\phi_b(y)\|_W-\|\bar\phi(x)-\bar\phi(y)\|_W\right|
\le\|(\phi_b(x)-\bar\phi(x))-(\phi_b(y)-\bar\phi(y))\|_W
\le r_x+r_y.\tag{10}
\]
例如平均距離加 r_x+r_y 仍低於閾值 τ，則所有已測模板均落在距離閾值內。
這不是抽樣置信區間，不涵蓋未測新模板，也不證明語義正確。
這個 r_x 界限適用於材料間距離。實作中的概念相關邊界另以「跨材料中心化後的
單位長度錨點欄」計算半徑；欄方差為零時，相關與該界限均未定義。

## D9. 變化與網路統計

本節 x_i、y_j 是已選定 score variant 的材料向量；v0.1 使用 d(u,v)=||u−v||₂。
兩期的 centroid displacement：\(D=\|\bar x-\bar y\|_2\)。
empirical energy distance 使用 V-statistic（包含同組對角0項）：
\[
\widehat E=\frac{2}{nm}\sum_{ij}d(x_i,y_j)
-\frac1{n^2}\sum_{ii'}d(x_i,x_{i'})
-\frac1{m^2}\sum_{jj'}d(y_j,y_{j'}).\tag{11}
\]
固定分塊精確累計，不改成下採樣或 U-statistic。例 X=(-1,1)，Y=(0,0)，
D=0 而 E=1，展示均值不動仍有分布差異。同分布的兩份有限樣本也可得到正距離，
不能把經驗距離大於0直接當成母體變化證據。

實作逐目標、兩個時期，以來源組為單位 bootstrap；sampling_units 可顯式指定，
否則用樣本 group_id，缺失才以 sample_id 代替。每次抽中來源組帶入該組全部材料，
所以估計的是按材料行加權的均值／分布。兩期來源集合相同時共享抽樣次數，來源集合
不相交時各自抽樣；部分重疊拒絕自動推斷。任一側有效來源組少於2時區間和 p 值未定義。
若 group_id 只表示目標詞，不能把同詞多條上下文誤當多個獨立來源；需顯式提供合理的
sampling_units。跨目標詞 benchmark 指標的重抽樣是另一層分析，當前 shift API 不自動實作。

預設1000次、seed=42，回傳2.5%/97.5% bootstrap 分位數。這些重抽樣區間不保證在
零變化邊界或極少來源組時具有95%覆蓋率，不能以距離區間是否含0作顯著性檢驗。
可選 permutation 對不相交來源組整組重新分配時期，對配對來源交換整組的時期資料塊；
需滿足相應可交換假設。非隨機歷時資料不任意解讀成因果變化。

概念網路為同一批材料的錨點欄相關。共同置換樣本行不改變網路；各欄獨立置換
用來破壞材料配對的共變關係，單次有限樣本的相關不必恰為0。
sample network 節點是材料/usage，需用人工 usage relatedness
評估，與 concept network 不同。負相關是有符號邊，不任意取絕對值當正關聯。

## D10. PMPO 式目標（設計選擇，非理論最優）

對每錨點 i，用訓練樣本比較同關係模板的 Spearman，r_i 為所有模板對的平均；
R=(1+median_i r_i)/2，V=macroAP(Y,mean_b Z_b)，J=.8V+.2R。
這裡只包含至少5正5負的 eligible 錨點；任一 eligible 錨點的必要模板相關未定義，
該子集不能取得有效 objective。主協議及 CLI 預設使用 reference_z；Python callback
若返回其他分數，V 中的 Z 必須相應替換並記錄，API 不會自動校準 callback 結果。
0.8/0.2 是預先指定的可消融權重，不由上述推導得出。
Cronbach α 僅沿同錨點的模板軸診斷，負值可存在，常數維度可未定義；
不最大化任意不同概念錨點之間的 α。

## D11. 命題對測試與經驗檢驗的對應

| 性質 | 證據 |
|---|---|
| 式1/2 token shift與完整normalizer | injected logits手算、真實GPU selected/full對照 |
| 事件編譯與媒體一致 | 空格/中文/Unicode/重分詞/MP4/PyAV與tensor/grid測試 |
| 式4/5與缺失政策 | logmeanexp反例、ddof/零方差/fit-apply隔離 |
| 式8/9/10 | affine z、平移、重複維度、有限模板半徑測試 |
| 式11 | block與dense一致、零均值但分布變化反例 |
| 概念有效性 | E2/E4/E5/E6的保留人工標註；目前尚未執行 |

FMAT 以命題填詞提供測量思路與驗證框架；PMPO 提供可優化探針工具觀。
本文件的自回歸事件、任意錨點幾何與參照推導是 VLanchor 的操作定義與分析，
不宣稱復現兩文全部方法或數值。來源：[FMAT](https://psychbruce.github.io/paper/Bao_2024_JPSP_FMAT.pdf)、
[PMPO](https://aclanthology.org/2025.ijcnlp-long.130/)。
