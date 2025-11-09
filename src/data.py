import os
from datasets import load_dataset
import sentencepiece as spm
from pathlib import Path
import tempfile
import contextlib
import sys

def load_or_download_iwslt(local_arrow_dir='/root/autodl-tmp/large-model/data/iwslt2017-en-de', hf_name='iwslt2017', config='iwslt2017-en-de'):
    try:
        local_path = Path(local_arrow_dir)
        split_map = {}
        
        # 遍历三个子文件夹（train / validation / test）
        for split_name in ['train', 'validation', 'test']:
            split_dir = local_path / split_name
            if split_dir.exists():
                arrow_files = list(split_dir.glob('*.arrow'))
                if len(arrow_files) > 0:
                    split_map[split_name] = str(arrow_files[0])
        
        if len(split_map) > 0:
            print(f"✅ Found local dataset splits: {list(split_map.keys())}")
            ds = load_dataset('arrow', data_files=split_map)
            return ds

        else:
            print("⚠️ No .arrow files found locally, will download dataset.")
    except Exception as e:
        print("⚠️ Local load failed, will download dataset instead:", e)

    # 如果本地加载失败，则从 Hugging Face 下载
    ds = load_dataset(hf_name, config)
    return ds

# 自定义静音上下文管理器：屏蔽C++层stderr输出
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

# 基于给定的语料文件训练SentencePiece模型
def train_sentencepiece(corpus_files, model_prefix='spm', vocab_size=8000, model_type='bpe', model_dir='/root/autodl-tmp/large-model/src/sentence'):
    if model_dir is None:
        model_dir = tempfile.mkdtemp()
    else:
        # 确保自定义目录存在
        os.makedirs(model_dir, exist_ok=True)

    cp = ','.join(corpus_files)
    model_prefix_path = os.path.join(model_dir, model_prefix)
    
    print(f"🧩 Training SentencePiece model: {model_prefix_path}.model (vocab={vocab_size})")
    
    # ✅ 使用 suppress_stdout_stderr() 屏蔽所有底层日志
    with suppress_stdout_stderr():
        spm.SentencePieceTrainer.Train(
            input=cp,
            model_prefix=model_prefix_path,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

    print(f"✅ SentencePiece model saved to {model_prefix_path}.model")
    
    return model_prefix_path + '.model', model_prefix_path + '.vocab'

# 从dataset中抽取源/目标句子并写成两个平行文本文件
def build_corpus_files(dataset, src_lang='en', tgt_lang='de', out_dir='/root/autodl-tmp/large-model/src/tokenizer_corpus', max_samples=None):
    Path(out_dir).mkdir(exist_ok=True)
    src_path = os.path.join(out_dir, 'src.txt')
    tgt_path = os.path.join(out_dir, 'tgt.txt')
    n = 0
    with open(src_path, 'w', encoding='utf-8') as sf, open(tgt_path, 'w', encoding='utf-8') as tf:
        for ex in dataset:
            if max_samples and n >= max_samples:
                break
            src = ex['translation'][src_lang].strip()
            tgt = ex['translation'][tgt_lang].strip()
            if src and tgt:
                sf.write(src + '\n')
                tf.write(tgt + '\n')
                n += 1
    return src_path, tgt_path
