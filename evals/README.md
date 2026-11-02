# 评测入口

当前实现 E0 输入审计：读取 LongMemEval 格式数据，采用字段白名单隔离历史、查询和评分标签，并生成带文件哈希及代码状态的报告。附带的三条记录全部是本项目手写合成样例，不是公开基准子集。

~~~powershell
python -m uv run hmph eval --suite evals/fixtures/longmemeval-smoke.json --output artifacts/evals/smoke.json
~~~

该命令不调用模型、不计算记忆准确率。报告必须保持 quality_status=not_run、quality_metrics=null，不能把输入审计通过当作记忆测试通过。

E0 当前只支持不超过 32 MiB 的小套件；完整公开数据的流式处理在后续实现，不要把此入口当成已经能运行完整排行榜的工具。

模型侧仅接收 ModelInput：中性来源编号、原始说话方/内容、正常历史时间，以及问题和请求时间。answer、has_answer、question_type、answer_session_ids 和带题型含义的原始标识不进入该对象。日期目前保持原始字符串，尚未测试时序归一化。

E0 的 32 MiB 限制继续适用；新增 E1 流式读取器支持不超过 512 MiB 的 LongMemEval JSON 数组，生成过程不读取 GoldAnswer。实验协议见 [评测设计](../docs/design/evaluation.md)。数据和输出放在 Git 忽略的 artifacts，真实用户对话不自动进入评测。

## E1 原文基线与语义检索实验

数据来源为 [LongMemEval 官方说明](https://github.com/xiaowu0162/LongMemEval#data)及 [cleaned 数据集](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)。本地使用的 LongMemEval-S revision 为 98d7416c24c778c2fee6e6f3006e7a073259d48f，源文件约 277 MB，数据卡标注 MIT。下载 URL、SHA-256、许可及条件保存在 artifacts/datasets/longmemeval/manifest.json；传入 --manifest 时会核对文件哈希。

~~~powershell
python -m uv run hmph eval-run --suite artifacts/datasets/longmemeval/longmemeval_s_cleaned.json --manifest artifacts/datasets/longmemeval/manifest.json --condition cleaned-S-development --output artifacts/evals/dev.json --limit 5 --seed 17 --judge
~~~

eval-run 默认使用 DeepSeek 本地配置，会产生收费调用；--judge 增加独立评分请求。reader 与 judge 同型号，评分规则为项目固定二元规则，不是官方 GPT-4o 成绩。回答侧只接收正常问题、日期及中性来源片段；答案、题型和拒答标签留在独立评分侧。按 seed 与 case ID 的哈希选样，不根据答案挑题。

可运行的 profile：recent_window、lexical_raw、source_aware_raw、可选 hybrid_raw、full_history。近期窗口和词法片段都最多 12 个片段、默认 12,000 UTF-8 字节读答输入。原文按 400 字符片段、80 字符重叠切分；词法采用英文词/CJK 双字与 BM25。source_aware_raw 从相同 BM25 前 80 候选开始，每条原消息只占一个来源单元，在最佳命中周围展开最多 800 字符，保留精确原文偏移并按时间呈现；同样最多 12 个来源及相同总预算，不能给相邻重叠片段重复记功。full_history 放不下时记录 unsupported，不调用模型、不报告虚假的零分或完整历史结果。它们分别是明确命名的基线，不冒充带摘要的 B0 或完整 H-memory。

无需供应商调用的全量来源检索检查：

~~~powershell
python -m uv run hmph eval-run --suite artifacts/datasets/longmemeval/longmemeval_s_cleaned.json --manifest artifacts/datasets/longmemeval/manifest.json --condition cleaned-S-retrieval-development --output artifacts/evals/retrieval.json --profiles recent_window lexical_raw --limit 500 --retrieval-only
~~~

同预算来源去重读答开发对照：

~~~powershell
python -m uv run hmph eval-run --suite artifacts/datasets/longmemeval/longmemeval_s_cleaned.json --manifest artifacts/datasets/longmemeval/manifest.json --condition cleaned-S-development --output artifacts/evals/source-aware-dev.json --profiles lexical_raw source_aware_raw --limit 20 --seed 31 --budget-bytes 12000 --judge
~~~

可选 hybrid_raw 使用本地多语言 MiniLM 与 BM25 的 reciprocal rank fusion。模型约 0.22 GB，首次下载至被忽略的 artifacts/models，正文不会发往 embedding 服务：

~~~powershell
python -m uv sync --extra embeddings --frozen
python -m uv run --extra embeddings hmph eval-run --suite artifacts/datasets/longmemeval/longmemeval_s_cleaned.json --manifest artifacts/datasets/longmemeval/manifest.json --condition cleaned-S-development --output artifacts/evals/hybrid.json --profiles lexical_raw hybrid_raw --limit 5 --seed 17 --judge
~~~

FastEmbed 0.9.1 / ONNX CPU 路径已实测；报告包含模型 snapshot 文件哈希与配置。当前每题重新构建公开语料的向量，不保存个人数据向量缓存，也未接入在线产品。向量试验尚未显示小样本质量收益，不能称为已完成的增强记忆。

报告保留每题状态、失败、unsupported、来源召回、预测、独立评分、token 用量与耗时。未知用量不会写成已知零；价格版本和货币成本尚未提供。报告在收费调用前及题后检查点保存，用原子替换和 Windows 短暂占用重试，检索模式限频写盘；只有 run_status=completed 表示完整结束。

官方 cleaned-S 中有 13 题含重复的 filler session ID，内容相同而日期不同。adapter v2 保留每个出现位置并生成不同中性 ID，不删除记录；若金标准引用重复 ID 则拒绝有歧义的输入。本轮重复项均非金标准会话。

实际结果及阶段判断见 [实现反思](../docs/implementation-assessment.md)。五题和后续二十题均为开发样本，不能再作为留出集；500 题来源召回不代表回答正确率。外部作者代码的合成接通已完成，版本与复跑方法见 [原生参照记录](../docs/research/native-references.md)；公开数据上的外部参照比较、第二类数据集、完整记忆构建与非劣性验收仍待执行。
