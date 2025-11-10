import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from pathlib import Path
from model import TransformerModel, generate_square_subsequent_mask
from data import load_or_download_iwslt, build_corpus_files, train_sentencepiece
import evaluate
from tqdm import tqdm
import sacrebleu
import numpy as np


# ---------------- 固定随机种子 ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 随机种子设置为: {seed}")


# ---------------- 数据集封装 ----------------
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_sp, tgt_sp, max_len=128):
        self.ds = hf_dataset
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]['translation']
        src = item['en']
        tgt = item['de']
        src_ids = self.src_sp.encode(src, out_type=int)
        tgt_ids = self.tgt_sp.encode(tgt, out_type=int)
        src_ids = src_ids[:self.max_len-2]
        tgt_ids = tgt_ids[:self.max_len-2]
        src_ids = [2] + src_ids + [3]  # 添加 BOS 和 EOS
        tgt_ids = [2] + tgt_ids + [3]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), src, tgt


# ---------------- DataLoader 拼接 ----------------
def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch, src_texts, tgt_texts = zip(*batch)
    max_src = max(len(x) for x in src_batch)
    max_tgt = max(len(x) for x in tgt_batch)
    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    for i, s in enumerate(src_batch):
        src_padded[i, :s.size(0)] = s
    for i, t in enumerate(tgt_batch):
        tgt_padded[i, :t.size(0)] = t
    return src_padded, tgt_padded, src_texts, tgt_texts


# ---------------- 翻译生成函数 ----------------
@torch.no_grad()
def translate_sentence(model, src_text, src_sp, tgt_sp, device, max_len=128, pad_id=0):
    """翻译单个句子"""
    model.eval()
    
    # 编码源文本
    src_ids = src_sp.encode(src_text, out_type=int)
    src_ids = [2] + src_ids[:max_len-2] + [3]  # 添加 BOS 和 EOS
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_key_padding_mask = (src_tensor == pad_id)
    
    # 编码
    memory = model.encode(src_tensor, src_key_padding_mask=src_key_padding_mask)
    
    # 自回归生成
    ys = torch.tensor([[2]], dtype=torch.long, device=device)  # BOS
    
    for step in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        tgt_key_padding_mask = (ys == pad_id)
        
        out = model.decode(ys, memory, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = model.generator(out[:, -1:])
        next_token = logits.argmax(dim=-1)
        
        ys = torch.cat([ys, next_token], dim=1)
        
        # 遇到 EOS 停止
        if next_token.item() == 3:
            break
    
    # 解码为文本
    pred_ids = ys[0].cpu().tolist()
    pred_ids = [id for id in pred_ids if id not in [0, 2, 3]]  # 移除特殊标记
    
    if not pred_ids:
        pred_text = ""
    else:
        pred_text = tgt_sp.decode(pred_ids)
    
    return pred_text


# ---------------- 测试函数 ----------------
def test_model(model, test_loader, src_sp, tgt_sp, device, num_samples=50, max_len=128):
    """测试模型并输出翻译结果"""
    model.eval()
    
    print(f"🧪 开始测试，随机抽取 {num_samples} 个样本...")
    print("=" * 100)
    
    # 随机选择样本
    all_indices = list(range(len(test_loader.dataset)))
    selected_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    
    all_hypotheses = []
    all_references = []
    all_sources = []
    
    for idx in tqdm(selected_indices, desc="测试进度"):
        try:
            # 获取样本
            src_tensor, tgt_tensor, src_text, ref_text = test_loader.dataset[idx]
            src_tensor = src_tensor.unsqueeze(0).to(device)
            
            # 生成翻译
            pred_text = translate_sentence(model, src_text, src_sp, tgt_sp, device, max_len)
            
            # 收集结果
            all_sources.append(src_text)
            all_references.append(ref_text)
            all_hypotheses.append(pred_text)
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            continue
    
    # 输出部分翻译示例
    print("\n📊 翻译示例:")
    print("=" * 100)
    for i in range(min(10, len(all_sources))):
        print(f"\n样本 {i+1}:")
        print(f"源文 (EN): {all_sources[i]}")
        print(f"参考译文 (DE): {all_references[i]}")
        print(f"模型译文 (DE): {all_hypotheses[i]}")
        print("-" * 80)
    
    # 计算 BLEU 分数
    if all_hypotheses:
        try:
            bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
            print(f"\n🎯 测试结果 (基于 {len(all_hypotheses)} 个样本):")
            print(f"BLEU 分数: {bleu.score:.2f}")
            print(f"详细统计:")
            print(f"  - 精确度: {bleu.precisions}")
            print(f"  - 长度比率: {bleu.ratio:.2f}")
            print(f"  - 翻译长度: {bleu.sys_len}, 参考长度: {bleu.ref_len}")
        except Exception as e:
            print(f"计算 BLEU 时出错: {e}")
    else:
        print("❌ 没有生成有效的翻译结果")
    
    return all_sources, all_references, all_hypotheses


# ---------------- 主函数 ----------------
def main():
    parser = argparse.ArgumentParser(description="测试训练好的翻译模型")
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/large-model/src/checkpoints/PE4/model_epoch10.pt', 
                       help='训练好的模型路径')
    parser.add_argument('--src_sp_model', type=str, default='/root/autodl-tmp/large-model/src/sentence/spm_src.model',
                       help='源语言分词器路径')
    parser.add_argument('--tgt_sp_model', type=str, default='/root/autodl-tmp/large-model/src/sentence/spm_tgt.model',
                       help='目标语言分词器路径')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='随机测试的样本数量')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_file', type=str, default='/root/autodl-tmp/large-model/results/test_results.csv',
                       help='结果保存文件路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  设备: {device}')
    
    # 加载分词器
    print("📖 加载分词器...")
    src_sp = spm.SentencePieceProcessor(model_file=args.src_sp_model)
    tgt_sp = spm.SentencePieceProcessor(model_file=args.tgt_sp_model)
    
    # 加载模型
    print("🤖 加载模型...")
    model = TransformerModel(
        len(src_sp), len(tgt_sp), 
        d_model=512, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, dropout=0.1, max_len=128
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ 模型加载成功: {args.model_path}")
    
    # 加载测试数据
    print("📚 加载测试数据...")
    ds = load_or_download_iwslt()
    test_data = ds['test']
    
    # 创建测试数据集
    test_dataset = TranslationDataset(test_data, src_sp, tgt_sp)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"📊 测试集大小: {len(test_dataset)}")
    
    # 进行测试
    sources, references, hypotheses = test_model(
        model, test_loader, src_sp, tgt_sp, device, args.num_samples
    )
    
    # 保存结果到文件（如果指定了输出文件）
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("源文,参考译文,模型译文\n")
            for src, ref, hyp in zip(sources, references, hypotheses):
                f.write(f'"{src}","{ref}","{hyp}"\n')
        print(f"💾 结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()