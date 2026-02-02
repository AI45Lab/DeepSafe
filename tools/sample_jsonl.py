import json
import random
import os

num_samples = 10

input_path = "/mnt/shared-storage-user/zhangbo1/Fake-Alignment-main/safety.jsonl"
output_path = f"/mnt/shared-storage-user/zhangbo1/Fake-Alignment-main/safety_sample_{num_samples}.jsonl"

input_path = "/mnt/shared-storage-user/zhangbo1/oaieval_sandbagging/evals/registry/data/sandbagging/samples-all.jsonl"
output_path = f"/mnt/shared-storage-user/zhangbo1/oaieval_sandbagging/evals/registry/data/sandbagging/samples-all_sample_{num_samples}.jsonl"

if os.path.exists(input_path):
    print(f"Loading from {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    total = len(data)
    print(f"Total items: {total}")
    
    sample = random.sample(data, min(num_samples, total))
    print(f"Sampled {len(sample)} items.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved sample to {output_path}")
else:
    print(f"File not found: {input_path}")

