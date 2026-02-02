
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from uni_eval.datasets.salad_bench import SaladDataset
import inspect
print(f"Imported SaladDataset from: {inspect.getfile(SaladDataset)}")

def test_dataset(path, name):
    print(f"--- Testing {name} ---")
    try:
        dataset = SaladDataset(path=path)

        print(f"Dataset Path: {dataset.path}")
        print(f"Loaded {len(dataset)} items.")
        
        if len(dataset) > 0:
            item = dataset[0]
            print(f"First item ID: {item.get('id')}")
            print(f"Prompt: {str(item.get('prompt'))[:50]}...")
            print(f"Meta Task Type: {item.get('meta', {}).get('task_type')}")
            if name == "MCQ":
                if item['meta']['task_type'] == 'mcq':
                    print("SUCCESS: Detected MCQ mode correctly.")
                else:
                    print(f"FAILURE: Expected 'mcq' but got '{item['meta']['task_type']}'")
            elif name == "QA":
                if item['meta']['task_type'] == 'qa':
                    print("SUCCESS: Detected QA mode correctly.")
                else:
                    print(f"FAILURE: Expected 'qa' but got '{item['meta']['task_type']}'")
        else:
            print("WARNING: Dataset is empty. Please check if the file exists and is not empty.")
            if os.path.exists(path):
                print(f"File exists. Size: {os.path.getsize(path)} bytes.")
            else:
                print("File DOES NOT exist.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mcq_path = "/mnt/shared-storage-user/zhangbo1/datasets/Salad-Data/mcq_set_sample_100.json"
    qa_path = "/mnt/shared-storage-user/zhangbo1/datasets/Salad-Data/base_set_sample_100.json"
    
    test_dataset(mcq_path, "MCQ")
    print("\n")
    test_dataset(qa_path, "QA")
