# 可编辑论文流程图

主文件：**OmniAnchor_editable.svg**。微调版画布为 1536 × 1072，保留参考图的 a / b / c 三部分布局、英文内容、公式、表格和连接关系。

- `OmniAnchor_editable.svg`：可独立使用的 SVG，照片已内嵌，无须联网或保持外部素材路径。
- `OmniAnchor_6144px.png`：6144 × 4288 高清预览。
- `OmniAnchor_vector.pdf`：矢量 PDF，正文保留为可选中文字。
- `OmniAnchor_preview.png`：1536 × 1072 快速预览。
- `figure.scene.json`：可修改的元素、样式与坐标源文件。
- `rebuild.py`：仅依赖 Python 标准库的确定性 SVG 生成脚本。
- `render.mjs`：使用 Playwright / Chromium 导出 PNG 与 PDF。
- `assets/`：真实数据集照片、原始矢量图标、许可证与详细来源。
- `validation.json`：可编辑元素计数、文字边界与出画布检查。

## 本轮排版微调

按最新要求，已删除图中底部的 Note 和 Photo 两行；素材来源仍随源文件保存在 SOURCES.md 中。

- 关系标题分两行，统一左边距；wording 卡片间距统一为 12 像素。
- 中间三个模块的顶边、底边与标题基线对齐；表格标题和表头之间增加留白。
- 坐标轴整体下移并略微压缩高度，使顶部轴标签与模块标题分离。
- 底部两路内容采用同一顶线；左侧三卡同高、间距均为 34 像素，流程箭头共用水平基线。
- 较长示例说明改成三行；右侧训练表增加行高，预测框与评价列表的间距拉开。
- 画布只向下增加 48 像素，为验证路线提供纵向空间。核心文案与公式已逐字比对保留。

上一版完整备份位于相邻目录的 `VLanchor_editable_before_spacing.zip`。

## 编辑

可用 Inkscape、Adobe Illustrator 等矢量编辑器打开 SVG。文字没有转曲；公式由可编辑文字及下标组成；方框、表格线、箭头、坐标轴、散点和图标均为原生矢量对象。分组按面板及内容命名，必要时取消分组后选择单个对象。

字体为 Times New Roman。其他系统若缺少此字体，会采用 Times / Liberation Serif 等后备字体，字宽可能略有变化。固定外观请参考 PNG / PDF。

两处照片是可单独替换的内嵌 `<image>` 对象；它们保留真实照片的位图属性，并不宣称照片的像素内容已转为可编辑矢量笔画。

## 数据集示例

两处原示例照片替换为 **OASIS I466 / Lake 12**，数据集记录的摄影者为 **James Wheeler**。原始刺激图为 **500 × 400**，未通过生成式放大添加内容。主面板使用 SVG 视窗居中裁切，验证面板近乎完整显示；源文件未裁切。矢量部分可任意放大，照片细节以真实刺激图原生分辨率为限。

评分、坐标与散点仍是原图的概念示意，未填入或虚构模型分数、人工评分或实验结果。已相应改写原图的“无人工评分 / 非基准样本”照片说明。

原图极小的 continuation 标题括号内容难以可靠辨认，重建时采用与逐词评分公式一致的 `c_{i,t}`。其余核心术语、锚点、公式、方向关系与验证路线均按参考图保留。

## 重建

```sh
python3 rebuild.py
python3 rebuild.py --render-spec figure.scene.json
```

第一条从脚本中的版式生成 SVG 和 scene JSON；第二条从手工编辑后的 scene JSON 生成 SVG。scene 使用明文元素坐标，图片以素材相对路径引用；输出时自动内嵌。

导出位图 / PDF 需要 Node.js、Playwright 和 Chrome：

```sh
node render.mjs
```

若 Playwright 位于单独的依赖目录，可通过 `BUNDLE_NODE_MODULES` 指向该 `node_modules` 目录。

此交付仅新增独立图稿目录；未修改论文正文或项目实验代码。删除这个目录及同名 ZIP 即可撤销。

来源与授权记录见 [SOURCES.md](SOURCES.md)。
