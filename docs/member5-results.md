# Member 5 实施与实验报告

已完成当前可独立执行的 B3 embedding 检索工作：代码、两模型验证集实验、向量缓存、精确索引、固定候选池、复现验证和队友交付包。团队正式评估、BM25 对比、Member 6 接入验收仍待协作。

## 实际完成清单

- [x] 使用 Member 1 已生成并校验的数据，未重新设计数据处理规则。
- [x] 独立 Python 3.12 环境及依赖锁定文件。
- [x] 两个预训练模型使用固定 Hugging Face revision，电影与用户均使用 canonical text。
- [x] 电影和画像向量缓存，输入与模型版本变化时隔离缓存，缓存损坏时报错。
- [x] 本地精确余弦索引、完整已看历史过滤、同分按电影 ID 排序。
- [x] 所有验证用户的 Top 100 预测与 Top 50 候选。
- [x] 比较 Recall@20/50/100、Recall@10、NDCG@10、运行耗时。
- [x] 依据事先声明的验证集 Recall@50（其次 NDCG@10）选定 **MiniLM L6**。
- [x] 4,751 位验证用户与 3,779 位测试用户，每人 50 个固定候选。
- [x] 选定模型的缓存重跑，预测、候选和索引 SHA-256 完全一致。
- [x] 全量候选合法性检查与保存索引重新加载后的分数验证。
- [x] 10 个自动化测试通过；独立审查发现的两个重要问题已用回归测试修复。

## 验证集结果（本地诊断，待 Member 2 正式复核）

所有实验使用同一批 4,751 位用户和 21,333 部候选电影。相关电影定义为验证集评分至少 4 分且在候选目录内。Recall 对每位用户计算再平均；不是“有命中的用户比例”。

| 模型 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | 平均检索毫秒/用户 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM L6 | 0.7926% | 1.3595% | 2.7232% | 4.2224% | 0.020434 | 1.923 |
| MiniLM L12 | 0.7424% | 1.1562% | 2.1688% | 3.6165% | 0.016966 | 2.410 |

验证集全部正例的目录覆盖率为 85.0037%；目录外目标不进入 recall 分母，但明确报告这一覆盖损失。没有运行或查看测试集目标指标，没有用测试标签选择模型。

候选数量比较：20、50、100 的 recall 如上。扩大池子可以提高候选覆盖，但交给 Member 6 的文件保持设计文档约定的 **50 部**；没有自行改变团队接口。这是两个模型中的选择，不能称为全局最优或证明 embedding 优于 BM25。

## 运行信息与限制

- 选定模型：`sentence-transformers/all-MiniLM-L6-v2`。
- 固定 revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
- 设备：CPU；本环境未提供可用 MPS。预训练推理，无额外微调，无付费推理 API。
- 选定模型初次验证运行：电影编码 61.86 秒，用户编码 64.82 秒，全流程 166.53 秒。
- 检索延迟只包含相似度计算、过滤和排序，不包含用户文字编码。编码耗时另列，不能当成在线端到端延迟。
- 256-token 默认长度导致验证集 **4749 / 4,751** 份画像截断；电影文档截断 **1 / 21,333**。测试画像截断 3779 / 3,779。
- 当前召回偏低，结果应如实保留。截断可能影响效果，但本实验没有证明它是低分的唯一原因。后续可在验证集上开展长输入编码实验，不能根据最终测试成绩反复选方案。
- 两模型使用共同的 canonical 文本和静态 genome 标签，其数据限制与 Member 1 文档一致。
- 不同硬件、线程数、batch 设置可能产生浮点差异；缓存身份没有囊括每一种硬件细节。固定文件及 SHA-256 是团队比较依据，不承诺跨机器重新编码逐字节一致。

## 交给 Member 6 的文件

目录：`member5/handoff/`（相对于本报告所在 outputs 目录）。

- `validation_candidates_top50.jsonl`：4,751 行。
- `test_candidates_top50.jsonl`：3,779 行。
- `validation_predictions_top100.jsonl`、`test_predictions_top100.jsonl`：B3 原始排名。
- `movie_documents.jsonl`：按 movie_id 连接取得标准电影文本。
- `index.npz`：电影向量及其 ID 映射。
- `validation_run.json`、`test_run.json`：配置、输入输出校验、版本与耗时。
- `selection.json`：模型选择规则与验证集结果。
- `manifest.json`：交付包文件校验值。

使用固定候选文件进行 B3/P1 比较，不让 P1 自行换一批候选。文件采用设计文档的标准 JSONL 格式，不需要自定义转换。实际消费验收仍由 Member 6 完成。

## 代码与复现

实现位于桌面项目的 `member5-embedding` 分支，尚未推送或合并。主要入口：

```bash
cd /Users/hxh/Desktop/movie-rec-agent
source .venv/bin/activate
export HF_HOME="$PWD/models/hf-cache"
python -m unittest discover -s tests -v
python -m src.retrieval.run_embedding \
  --data data/processed/ml25m-global-v1 \
  --config configs/embedding-selected.json \
  --split validation \
  --cache models/embedding-cache \
  --output outputs/predictions/member5-new-validation
```

完成目录不会被覆盖，重跑需要使用新输出目录。完整使用方法见项目 `src/retrieval/README.md`。新机器先用 Python 3.10+ 建立环境并安装 `requirements-embedding.txt`，再生成 Member 1 数据。

## 剩余协作项

- [ ] Member 2 使用正式评估器复核、确定统一 reranker 子集、生成置信区间和官方结果表。
- [ ] Member 4 提供 BM25 同用户、同划分的预测，才能实际比较 BM25 与 embedding。
- [ ] 团队确认本次模型配置提案；Member 6 验证候选文件接入并完成 P1 比较。

审查小项保留：测试文件的直接执行入口位于部分测试类之前；本文使用的 discovery 命令执行全部 10 个测试。代码已提供明确保存索引加载接口，避免重复归一化改变分数；诊断评估核对标签、画像和电影文本的原始 manifest 哈希。

模型资料：[MiniLM L6](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[MiniLM L12](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)。
