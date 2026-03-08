#!/usr/bin/env python3
"""
TmpAi Lite - 単体動作確認スクリプト
（外部依存なしで完結）
"""
from tmpai_lite import TmpAiLite, TmpAiLiteTokenizer, QuantizationConfig
from tmpai_lite.utils.memory import get_memory_usage, format_memory_size


def main():
    print("=" * 60)
    print("TmpAi Lite - 単体動作確認")
    print("=" * 60)
    
    # 1. トークナイザー初期化（内蔵）
    print("\n=== トークナイザー初期化 ===")
    tokenizer = TmpAiLiteTokenizer()
    print(f"語彙サイズ: {tokenizer.vocab_size}")
    print(f"特殊トークン:")
    print(f"  PAD: {tokenizer.pad_token_id}")
    print(f"  BOS: {tokenizer.bos_token_id}")
    print(f"  EOS: {tokenizer.eos_token_id}")
    print(f"  UNK: {tokenizer.unk_token_id}")
    print(f"  MASK: {tokenizer.mask_token_id}")
    
    # テストエンコード/デコード
    test_texts = [
        "こんにちは、世界！Hello World!",
        "This is a test sentence.",
        "TmpAi Liteは軽量モデルです。",
        "日本語とEnglishの混合テスト。",
    ]
    
    print("\n=== エンコード/デコードテスト ===")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"原文: {text}")
        print(f"トークン数: {len(tokens)}")
        print(f"トークンID: {tokens[:10]}..." if len(tokens) > 10 else f"トークンID: {tokens}")
        print(f"復元: {decoded}")
        print()
    
    # 2. モデル初期化（通常）
    print("=== モデル初期化（通常） ===")
    model = TmpAiLite()
    info = model.get_model_info()
    print(f"モデルタイプ: {info['model_type']}")
    print(f"語彙サイズ: {info['vocab_size']}")
    print(f"埋め込み次元: {info['embed_dim']}")
    print(f"レイヤー数: {info['num_layers']}")
    print(f"最大シーケンス長: {info['max_seq_len']}")
    print(f"総パラメータ: {info['parameter_count_str']}")
    print(f"モデルサイズ（推定）: {info['model_size_gb']:.2f} GB (float32)")
    
    # メモリ使用量確認
    mem_stats = get_model_memory(model)
    print(f"メモリ使用量: {mem_stats['total_size']}")
    
    # 3. テキスト生成テスト
    print("\n=== テキスト生成テスト ===")
    
    # 入力テキストをトークン化
    input_text = "Hello, this is"
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    print(f"入力: {input_text}")
    print(f"入力トークン: {input_ids}")
    
    # 生成
    import torch
    input_tensor = torch.tensor([input_ids])
    
    print("\n生成中...")
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    
    # デコード
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(f"生成結果: {output_text}")
    
    # 4. 量子化設定のデモ
    print("\n=== 4bit量子化設定デモ ===")
    config = QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
    )
    print(f"量子化有効: {config.load_in_4bit}")
    print(f"計算精度: {config.bnb_4bit_compute_dtype}")
    print(f"量子化タイプ: {config.bnb_4bit_quant_type}")
    
    # 5. モデル保存・読込テスト
    print("\n=== モデル保存・読込テスト ===")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model")
        tokenizer_path = os.path.join(tmpdir, "tokenizer")
        
        # 保存
        print(f"モデルを保存: {model_path}")
        model.save_pretrained(model_path)
        
        print(f"トークナイザーを保存: {tokenizer_path}")
        tokenizer.save_pretrained(tokenizer_path)
        
        # 読込
        print(f"モデルを読込...")
        loaded_model = TmpAiLite.load_pretrained(model_path)
        loaded_info = loaded_model.get_model_info()
        print(f"読込成功: {loaded_info['parameter_count_str']} パラメータ")
        
        print(f"トークナイザーを読込...")
        loaded_tokenizer = TmpAiLiteTokenizer()
        loaded_tokenizer.load_pretrained(tokenizer_path)
        print(f"読込成功: {loaded_tokenizer.vocab_size} 語彙")
        
        # 読込したモデルでテスト
        test_decode = loaded_tokenizer.decode(loaded_tokenizer.encode("Test"))
        print(f"読込テスト: {test_decode}")
    
    print("\n" + "=" * 60)
    print("TmpAi Lite 単体動作確認完了！")
    print("=" * 60)


def get_model_memory(model):
    """Helper to get model memory stats."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total = param_size + buffer_size
    
    class MemStats:
        def __init__(self, total, total_str):
            self.total_bytes = total
            self.total_size = total_str
    
    return MemStats(total, format_memory_size(total))


if __name__ == '__main__':
    main()
