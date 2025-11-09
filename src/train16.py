import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from pathlib import Path
from model import TransformerModel, generate_square_subsequent_mask
from data import load_or_download_iwslt, build_corpus_files, train_sentencepiece
import evaluate  # 用于BLEU计算
from tqdm import tqdm
import matplotlib.pyplot as plt
import sacrebleu
import random
import numpy as np


# ---------------- 固定随机种子函数 ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed set to {seed}")
    

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
        src_ids = [2] + src_ids + [3]
        tgt_ids = [2] + tgt_ids + [3]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ---------------- DataLoader 拼接 ----------------
def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)
    max_src = max(len(x) for x in src_batch)
    max_tgt = max(len(x) for x in tgt_batch)
    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    for i, s in enumerate(src_batch):
        src_padded[i, :s.size(0)] = s
    for i, t in enumerate(tgt_batch):
        tgt_padded[i, :t.size(0)] = t
    return src_padded, tgt_padded

# ---------------- 掩码生成 ----------------
def make_masks(tgt, pad_id=0):
    tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
    tgt_key_padding_mask = (tgt == pad_id)
    return tgt_mask, tgt_key_padding_mask

# ---------------- 训练一个epoch ----------------
def train_epoch(model, dataloader, optimizer, criterion, device, pad_id=0):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask, tgt_kp = make_masks(tgt_input, pad_id)
        src_kp = (src == pad_id)

        optimizer.zero_grad()
        out = model(src, tgt_input, src_key_padding_mask=src_kp,
                    tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_kp,
                    memory_key_padding_mask=src_kp)

        loss = criterion(out.reshape(-1, out.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_lm(model, dataloader, criterion, device, pad_id=0):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask, tgt_kp = make_masks(tgt_input, pad_id)
        src_kp = (src == pad_id)

        out = model(src, tgt_input, src_key_padding_mask=src_kp,
                    tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_kp,
                    memory_key_padding_mask=src_kp)

        loss = criterion(out.reshape(-1, out.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# ---------------- 计算 BLEU ----------------
@torch.no_grad()
def compute_bleu_fixed(model, dataloader, src_sp, tgt_sp, device, max_len=128, pad_id=0, num_samples=200):
    """修复版本的 BLEU 计算"""
    model.eval()
    all_hypotheses = []
    all_references = []
    
    print("开始计算 BLEU 分数...")
    
    count = 0
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="BLEU计算进度")):
        if count >= num_samples:
            break
            
        src = src.to(device)
        tgt = tgt.to(device)
        
        for i in range(src.size(0)):
            if count >= num_samples:
                break
                
            try:
                # 处理源序列
                src_seq = src[i:i+1]
                src_kp = (src_seq == pad_id)
                
                # 编码
                memory = model.encode(src_seq, src_key_padding_mask=src_kp)
                
                # 自回归生成
                ys = torch.tensor([[2]], dtype=torch.long, device=device)  # BOS
                
                for step in range(max_len - 1):
                    tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
                    tgt_kp = (ys == pad_id)
                    
                    out = model.decode(ys, memory, tgt_mask, tgt_key_padding_mask=tgt_kp)
                    logits = model.generator(out[:, -1:])
                    next_token = logits.argmax(dim=-1)
                    
                    ys = torch.cat([ys, next_token], dim=1)
                    
                    # 遇到 EOS 停止
                    if next_token.item() == 3:
                        break
                
                # 转换为文本
                pred_ids = ys[0].cpu().tolist()
                ref_ids = tgt[i].cpu().tolist()
                
                # 清理特殊标记
                pred_ids = [id for id in pred_ids if id not in [0, 2, 3]]
                ref_ids = [id for id in ref_ids if id not in [0, 2, 3]]
                
                # 防止空序列
                if not pred_ids:
                    pred_ids = [1]
                if not ref_ids:
                    ref_ids = [1]
                
                # 解码为文本
                pred_text = tgt_sp.decode(pred_ids)
                ref_text = tgt_sp.decode(ref_ids)
                
                # 添加到列表
                all_hypotheses.append(pred_text)
                all_references.append(ref_text)
                
                count += 1
                
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
    
    print(f"成功处理 {len(all_hypotheses)} 个样本")
    
    if not all_hypotheses:
        print("警告: 没有生成任何有效的翻译!")
        return 0.0
    
    try:
        # 使用 sacrebleu 计算 BLEU
        bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
        print(f"BLEU 计算成功!")
        print(f"BLEU 分数: {bleu.score:.2f}")
        print(f"详细结果: {bleu}")
        return bleu.score
    except Exception as e:
        print(f"计算 BLEU 时出错: {e}")
        return 0.0

# ---------------- 主函数 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/large-model/src/checkpoints/PE16')
    parser.add_argument('--result_dir', type=str, default='/root/autodl-tmp/large-model/results/PE16', 
                        help="Directory to save loss/BLEU curves")
    parser.add_argument('--max_samples_tokenizer', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")  
    args = parser.parse_args()
    
    # ✅ 固定随机种子
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    # ✅ 确保保存路径存在
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载数据集（包含 train/validation/test）
    ds = load_or_download_iwslt()
    train_data = ds['train']
    valid_data = ds['validation'] if 'validation' in ds else None
    test_data = ds['test'] if 'test' in ds else None

    # 2. SentencePiece 分词器训练
    src_corpus, tgt_corpus = build_corpus_files(train_data, max_samples=args.max_samples_tokenizer)
    src_model, _ = train_sentencepiece([src_corpus], model_prefix='spm_src', vocab_size=8000)
    tgt_model, _ = train_sentencepiece([tgt_corpus], model_prefix='spm_tgt', vocab_size=8000)
    src_sp = spm.SentencePieceProcessor(model_file=src_model)
    tgt_sp = spm.SentencePieceProcessor(model_file=tgt_model)

    # 3. DataLoader
    train_dataset = TranslationDataset(train_data, src_sp, tgt_sp)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_loader = None
    if valid_data is not None:
        valid_dataset = TranslationDataset(valid_data, src_sp, tgt_sp)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. 模型与优化器
    model = TransformerModel(len(src_sp), len(tgt_sp), d_model=512, nhead=16,
                             num_encoder_layers=4, num_decoder_layers=4,
                             dim_feedforward=1024, dropout=0.1, max_len=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    
    train_losses = []
    val_losses = []
    bleu_scores = []
    
    # 5. 训练循环
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_lm(model, valid_loader, criterion, device) if valid_loader else 0.0
        elapsed = time.time() - start
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        

        print(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s')

        if valid_loader:
            bleu = compute_bleu_fixed(model, valid_loader, src_sp, tgt_sp, device)
            bleu_scores.append(bleu)
            print(f'Validation BLEU: {bleu:.2f}')

        torch.save(model.state_dict(), Path(args.save_dir) / f'model_epoch{epoch}.pt')
        

    print("✅ Training complete!")
    
# ---------------- 绘制曲线 ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    if valid_loader:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(args.result_dir) / 'loss_curve.png')  # ✅ 保存为图片
    plt.show()

    if bleu_scores:
        plt.figure(figsize=(8, 5))
        plt.plot(bleu_scores, label='Validation BLEU', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('Validation BLEU Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(args.result_dir) / 'bleu_curve.png')
        plt.show()

if __name__ == '__main__':
    main()