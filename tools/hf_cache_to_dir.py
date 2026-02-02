                                       
import json
import os
import shutil
import pathlib
import argparse
from tqdm import tqdm
import sys

def handle_symlink(src: pathlib.Path, dst: pathlib.Path, use_hardlink: bool) -> None:
    """处理符号链接，将其转换为实际文件/目录"""
    try:
        target = pathlib.Path(os.readlink(src))
                
        if not target.is_absolute():
            target = src.parent / target
        
        if not target.exists():
            print(f"⚠️ 警告：符号链接 {src} 指向不存在的目标 {target}，已跳过")
            return

        if target.is_symlink():
            handle_symlink(target, dst, use_hardlink)
            return

        if target.is_file():
            if dst.exists():
                dst.unlink()
            if use_hardlink:
                try:
                    os.link(target, dst)
                    return
                except OSError:
                    pass
            shutil.copy2(target, dst)

        elif target.is_dir():
            if dst.exists():
                if dst.is_dir():
                            
                    for item in dst.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                else:
                    dst.unlink()
            shutil.copytree(target, dst, dirs_exist_ok=True, symlinks=False)
            
    except Exception as e:
        print(f"⚠️ 处理符号链接 {src} 时出错：{str(e)}，已跳过")

def unhub(cache_dir: str, out_dir: str, revision: str = "main", use_hardlink: bool = True, 
          model_type: str = "auto", overwrite: bool = False, delete_source: bool = False):
    """
    把 HuggingFace 缓存目录转成普通模型目录（完全离线）
    cache_dir : 原始缓存顶层，例如 ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct
    out_dir   : 想要生成的标准目录
    revision  : 分支名，默认 main
    use_hardlink : 是否使用硬链（节省空间），默认 True
    model_type: 模型类型 (auto, text, image, audio)，用于文件检查
    overwrite : 是否自动覆盖目标目录，默认 False
    delete_source: 转换成功后是否删除原始缓存目录，默认 False
    """
    cache_path = pathlib.Path(cache_dir).resolve()
    out_path = pathlib.Path(out_dir).resolve()

    if not cache_path.exists():
        raise FileNotFoundError(f"缓存目录不存在：{cache_path}")

    ref_file = cache_path / "refs" / revision
    if not ref_file.exists():
        raise FileNotFoundError(f"{ref_file} 不存在，确认缓存完整/版本号正确")
    commit_hash = ref_file.read_text().strip()

    snapshot = cache_path / "snapshots" / commit_hash
    if not snapshot.exists():
        raise FileNotFoundError(f"快照目录不存在：{snapshot}")

    required_files_map = {
        "auto": ["config.json"],
        "text": ["config.json", "tokenizer_config.json", "vocab.txt"],
        "image": ["config.json", "preprocessor_config.json"],
        "audio": ["config.json", "preprocessor_config.json"]
    }

    if model_type not in required_files_map:
        raise ValueError(f"不支持的模型类型：{model_type}，可选类型：{list(required_files_map.keys())}")
    
    required_files = required_files_map[model_type]
    missing = [f for f in required_files if not (snapshot / f).exists()]
    if missing:
                   
        print(f"⚠️ 警告：快照目录缺少推荐文件：{missing}，可能影响模型使用")

    if out_path.exists():
        if any(out_path.iterdir()):
            if not overwrite:
                          
                confirm = input(f"目标目录 {out_path} 非空。\n"
                               f"是否删除并重建此目录？这将清除所有现有内容！[y/N] ")
                if confirm.lower() != "y":
                    print("❌ 操作已取消")
                    return
                     
            try:
                for item in out_path.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            except Exception as e:
                raise RuntimeError(f"清除目标目录内容时出错：{str(e)}")

    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(snapshot.iterdir(), key=lambda x: x.name)

    for src in tqdm(files, desc="转换中"):
        try:
            dst = out_path / src.name
            
            if src.is_symlink():
                                
                handle_symlink(src, dst, use_hardlink)
                
            elif src.is_dir():
                              
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
                
            else:
                        
                if dst.exists():
                    dst.unlink()

                if use_hardlink:
                    try:
                        os.link(src, dst)
                        continue
                    except OSError:                
                        pass
                
                shutil.copy2(src, dst)
                
        except Exception as e:
            print(f"⚠️ 处理 {src} 时出错：{str(e)}，已跳过")

    print(f"✅ 已生成标准目录：{out_path.absolute()}")

    if delete_source:
        print(f"正在删除原始缓存目录：{cache_path}...")
        try:
            shutil.rmtree(cache_path)
            print("✅ 原始目录已删除")
        except Exception as e:
            print(f"⚠️ 删除原始目录失败：{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 HuggingFace 缓存转换为标准模型目录")
    parser.add_argument("cache_dir", help="原始缓存目录路径")
    parser.add_argument("out_dir", help="目标模型目录路径")
    parser.add_argument("--revision", default="main", help="分支名（默认 main）")
    parser.add_argument("--no-hardlink", action="store_true", help="不使用硬链，强制复制")
    parser.add_argument("--model-type", default="auto", 
                      choices=["auto", "text", "image", "audio"],
                      help="模型类型，用于文件检查（默认 auto）")
    parser.add_argument("--overwrite", action="store_true", 
                      help="自动覆盖目标目录，不提示确认")
    parser.add_argument("--delete-original", action="store_true",
                      help="转换成功后删除原始缓存目录")
    
    args = parser.parse_args()

    try:
        unhub(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            revision=args.revision,
            use_hardlink=not args.no_hardlink,
            model_type=args.model_type,
            overwrite=args.overwrite,
            delete_source=args.delete_original
        )
    except Exception as e:
        print(f"❌ 操作失败：{str(e)}", file=sys.stderr)
        sys.exit(1)
