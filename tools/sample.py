                                            
import json
import random
import os

def sample_json(input_path, output_path, num_samples):
    with open(input_path, 'r') as f:
        data = json.load(f)
    data = random.sample(data, num_samples)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def sample_jsonl(input_path, output_path, num_samples):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    data = random.sample(data, num_samples)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    input_path = "/root/zhangbo1/Fake-Alignment-main/safety.jsonl"
    output_path = "/root/zhangbo1/Fake-Alignment-main/safety_sample_30.jsonl"
    num_samples = 30
    sample_jsonl(input_path, output_path, num_samples)