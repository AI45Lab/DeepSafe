"""
Ch3EfEvaluator 使用示例

展示如何使用改进后的 Ch3EfEvaluator 进行两阶段评估：
1. 阶段一：使用目标 VLM 生成响应
2. 阶段二：使用 judge 模型评估响应质量
"""

config_with_judge = {
    "type": "Ch3EfEvaluator",
    "judge_model_cfg": {
        "type": "OpenAIModel",              
        "model_name": "gpt-4o",
        "api_key": "your-api-key",
    },
    "batch_size": 16,
    "template_name": "ch3ef_general",        
    "use_dimension_specific_prompts": True,                  
    "enable_judge": True,               
}

config_without_judge = {
    "type": "Ch3EfEvaluator",
    "batch_size": 32,
    "enable_judge": False,               
}

config_harmless = {
    "type": "Ch3EfEvaluator",
    "judge_model_cfg": {
        "type": "QwenVLModel",
        "model_name": "Qwen2-VL-72B",
    },
    "template_name": "ch3ef_harmless",                 
    "use_dimension_specific_prompts": True,
    "enable_judge": True,
}

"""
from uni_eval.registry import EVALUATORS, DATASETS, MODELS

# 1. 加载数据集
dataset_cfg = {
    "type": "Ch3efDataset",
    "path": "/path/to/Ch3Ef/meta_file/Harmless.json",
    "ppl": False,
}
dataset = DATASETS.build(dataset_cfg)

# 2. 加载目标模型
model_cfg = {
    "type": "QwenVLModel",
    "model_name": "Qwen2-VL-7B-Instruct",
}
model = MODELS.build(model_cfg)

# 3. 构建评估器
evaluator = EVALUATORS.build(config_with_judge)

# 4. 执行评估
results = evaluator.evaluate(model, dataset)

# 5. 分析结果
for result in results[:5]:  # 查看前5个结果
    print(f"ID: {result['id']}")
    print(f"Dimension: {result['dimension']}")
    print(f"Question: {result['prompt'][:100]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Judgment: {result.get('judgment', 'N/A')}")
    print(f"Ground Truth: {result.get('ground_truth', 'N/A')}")
    print("-" * 80)
"""

"""
Ch3EfEvaluator 的主要特性（模仿 MMSafetyBenchEvaluator）：

1. 两阶段评估流程
   - Phase 1: 使用目标 VLM 生成响应
   - Phase 2: 使用 judge 模型评估响应

2. 多模态支持
   - 自动处理图像（从文件路径加载并编码为 base64）
   - 支持每个问题包含多张图像
   - 在 judge 阶段将图像与文本一起发送

3. 维度特定的评估
   - Harmless: 评估响应的安全性和无害性
   - Helpful: 评估响应的有用性和实用性
   - Honest: 评估响应的真实性和准确性

4. 灵活的配置
   - 可以启用/禁用 judge 阶段
   - 支持自定义 prompt 模板
   - 支持批量处理以提高效率

5. 完整的结果输出
   - 保留所有原始数据集字段
   - 添加 'prediction' 字段（模型响应）
   - 添加 'judgment' 字段（judge 评估）

6. 错误处理
   - 图像加载失败时优雅降级
   - 支持文本截断以避免超长输入
   - 提供详细的日志记录
"""

"""
MM-SafetyBenchEvaluator              Ch3EfEvaluator
─────────────────────────────────────────────────────────────
- category_template_map              - dimension_template_map
- _encode_image_from_meta()          - _encode_image_from_path()
- use_category_specific_prompts      - use_dimension_specific_prompts
- Political_Lobbying/Legal_Opinion   - Harmless/Helpful/Honest
- meta["image"] (HF image object)    - image paths (file paths)
- Phase 1: VLM generation            - Phase 1: VLM generation
- Phase 2: Judge evaluation          - Phase 2: Judge evaluation
"""

