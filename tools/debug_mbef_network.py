import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEBUG_NETWORK")

sys.path.append("/mnt/shared-storage-user/zhangbo1/MBEF")

try:
    from uni_eval.models.api import APIModel
except ImportError as e:
    logger.error(f"Failed to import APIModel: {e}")
    sys.exit(1)

def test_connection():
    print("\n" + "="*50)
    print("DEBUGGING MBEF APIModel Connectivity")
    print("="*50)

    config = {
        "model_name": "gpt-4o",
        "api_base": "http://35.220.164.252:3888/v1/",
        "api_key": "OPENAI_API_KEY",                     
        "http_proxy": "HTTP_PROXY",
        "timeout": 10.0,
        "concurrency": 1
    }

    print(f"\n[1] Configuration:")
    print(f"    API Base:   {config['api_base']}")
    print(f"    Proxy:      {config['http_proxy']}")
    print(f"    Timeout:    {config['timeout']}")

    print(f"\n[2] Environment Variables (Before Init):")
    print(f"    http_proxy:  {os.environ.get('http_proxy', 'Not Set')}")
    print(f"    https_proxy: {os.environ.get('https_proxy', 'Not Set')}")
    print(f"    no_proxy:    {os.environ.get('no_proxy', 'Not Set')}")

    print(f"\n[3] Initializing APIModel...")
    try:
        model = APIModel(**config)
        print("    --> Initialization Success!")
    except Exception as e:
        print(f"    --> Initialization FAILED: {e}")
        return

    print(f"\n[4] Environment Variables (After Init - Modified by APIModel):")
    print(f"    http_proxy:  {os.environ.get('http_proxy', 'Not Set')}")
    print(f"    https_proxy: {os.environ.get('https_proxy', 'Not Set')}")
    print(f"    no_proxy:    {os.environ.get('no_proxy', 'Not Set')}")

    print(f"\n[5] Testing Generation (Calling GPT-4o)...")
    prompt = "Hello, are you online? Reply with 'Yes'."
    try:
                                                           
        responses = model.generate([prompt])
        print(f"    --> Generation Success!")
        print(f"    --> Response: {responses[0]}")
    except Exception as e:
        print(f"    --> Generation FAILED!")
        print(f"    --> Error Type: {type(e).__name__}")
        print(f"    --> Error Details: {e}")

        if "407" in str(e):
            print("\n[!] HINT: 407 Proxy Authentication Required. Check username/password.")
        elif "ConnectTimeout" in str(e) or "ReadTimeout" in str(e):
            print("\n[!] HINT: Connection Timed Out. Proxy might be unreachable or too slow.")

if __name__ == "__main__":
    test_connection()

