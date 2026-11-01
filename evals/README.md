# 评测入口

当前实现 E0 输入审计：读取 LongMemEval 格式数据，采用字段白名单隔离历史、查询和评分标签，并生成带文件哈希及代码状态的报告。附带的三条记录全部是本项目手写合成样例，不是公开基准子集。

~~~powershell
python -m uv run hmph eval --suite evals/fixtures/longmemeval-smoke.json --output artifacts/evals/smoke.json
~~~

该命令不调用模型、不计算记忆准确率。报告必须保持 quality_status=not_run、quality_metrics=null，不能把输入审计通过当作记忆测试通过。

E0 当前只支持不超过 32 MiB 的小套件；完整公开数据的流式处理在后续实现，不要把此入口当成已经能运行完整排行榜的工具。

模型侧仅接收 ModelInput：中性来源编号、原始说话方/内容、正常历史时间，以及问题和请求时间。answer、has_answer、question_type、answer_session_ids 和带题型含义的原始标识不进入该对象。日期目前保持原始字符串，尚未测试时序归一化。

公开数据下载、许可归档、实际记忆构建、读答模型及裁判在后续 E1/E2 实现。实验协议见 [评测设计](../docs/design/evaluation.md)。原始数据和输出放在 Git 忽略的 artifacts 目录，避免意外提交真实用户内容或大型数据。
