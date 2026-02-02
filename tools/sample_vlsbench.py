import json
import random
import os

num_samples = 10
input_path = "/mnt/shared-storage-user/zhangbo1/datasets/vlsbench/data.json"
output_path = f"/mnt/shared-storage-user/zhangbo1/datasets/vlsbench/data_sample_{num_samples}.json"

if os.path.exists(input_path):
    print(f"Loading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"Total items: {total}")
    
    sample = random.sample(data, min(num_samples, total))
    print(f"Sampled {len(sample)} items.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved sample to {output_path}")
else:
    print(f"File not found: {input_path}")

