# DeepSafe当前方案运行说明

运行入口为tools/run.py，组件为QwenEvidenceModel、QwenEvidenceDataset、QwenEvidenceEvaluator、QwenEvidenceMetric。只使用置信度筛选与单阶段独立证据复核，旧两阶段和兼容入口已删除。

框架README和唯一依赖清单均位于[ValueQA_Classification目录](../ValueQA_Classification/README.md)。在DeepSafe根目录执行`python -m pip install -r ValueQA_Classification/requirements.txt`即可安装初判/API复核依赖。可选vLLM由setup_vllm.sh安装到独立Linux环境。

## 当前明确参数

| 参数 | 值 |
|---|---|
| high / medium送审 | 预测类别概率 < 0.9 |
| low送审 | 预测类别概率 < 0.7 |
| API模型 | deepseek-flash |
| API地址 | https://api.deepseek.com |
| API并发 | 5 |
| 输出预算 / 超时 | 4096 tokens / 120秒 |
| 接受方式 | experimental，通过校验的改判会被接受 |

一条入选样本仅一次模型复核；要求提取、答案核验、证据引用和修复诊断在同一次调用完成。跳过标记为review_status=skipped，调用失败为terminal_failed；均保留初判。模型看不到初判、置信度和参考标签。

## Windows

设置LLM_REVIEW_API_KEY，选择configs/eval_tasks/valueqa_classification_api.yaml。已有1533条完整初判时用valueqa_classification_cached_1533.yaml，不加载Qwen：

```powershell
conda activate Laya
python tools/run.py configs/eval_tasks/valueqa_classification_cached_1533.yaml --evaluator.initial_predictions path/to/initial_predictions.jsonl
```

## Linux

选择valueqa_classification_api_linux.yaml，并明确设置QWEN_BASE_PATH；LoRA已位于仓库models/qwen35_4b_lora/best_adapter，配置通过相对路径加载。QWEN_LORA_PATH仅在使用其他adapter目录时覆盖。离线vLLM选择valueqa_classification_vllm.yaml并设置REVIEW_MODEL_PATH，必要时用VLLM_PYTHON指定独立环境。使用4bit NF4、BF16及训练时初判协议。Windows和Linux统一安装ValueQA_Classification/requirements.txt，其中已包含bitsandbytes。

## 恢复和数据

每个明确任务配置使用独立state_dir。evaluator.limit只限制本次未完成样本，0表示全部。创建.pause暂停，删除后原命令续跑。成功或skipped不重复，失败不自动重试。更换协议/源码后使用新目录，导入冻结初判，禁止修改旧身份绕过约束。

原始输入字段全部保留，target_comment为null表示没有目标评论。参考注释按各自field校验字符位置，不将材料或题目注释强制当成答案片段。复核等级不是片段定位标签。

首次缓存运行须将path/to/initial_predictions.jsonl替换为匹配输入的冻结初判文件，不再自动读取某个本机实验目录。仓库不包含results中的缓存；已有同配置checkpoint可直接续跑。600条旧初判的评论占位与当前null输入有差异，不能通过改写输入哈希伪装成完全匹配的缓存。该次实验差异及未经独立核实的标签已在报告披露。1533条现有初判可按完全匹配输入直接导入。

配置必须显式指定：独立入口--config，DeepSafe evaluator.config_path。配置只保存密钥环境变量名，密钥不写文件。新配置清楚标记experimental，不声明独立校准已完成。

[规则说明](单阶段置信度复核_20261012.md) · [完整600与1533实验结果](../experiments/single_review_deepseek_newkey_600_1533_20261012/report.md)。

LoRA权重使用Git LFS管理，GitHub克隆后执行git lfs pull。相对路径按配置文件目录解析，不按当前终端目录解析。服务器压缩包同时包含ValueQA_Classification代码目录和models中的推理adapter，不包含基础模型、训练checkpoint或API密钥。[发布说明](GitHub发布说明.md)。
