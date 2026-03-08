"""
TmpAi Lite Tokenizer

Built-in BPE (Byte Pair Encoding) tokenizer.
No external dependencies - completely self-contained.
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional
import unicodedata


class TmpAiLiteTokenizer:
    """
    BPE Tokenizer for TmpAi Lite.
    
    Features:
    - BPE encoding/decoding
    - Vocabulary size: 32000
    - Japanese and English support
    - Special tokens: <pad>, <s>, </s>, <unk>, <mask>
    """
    
    SPECIAL_TOKENS = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '<unk>': 3,
        '<mask>': 4,
    }
    
    def __init__(
        self,
        vocab_size: int = 32000,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            vocab_file: Path to vocabulary JSON file (optional)
            merges_file: Path to BPE merges file (optional)
        """
        self.vocab_size = vocab_size
        self._vocab: Dict[str, int] = {}
        self._inv_vocab: Dict[int, str] = {}
        self._merges: List[Tuple[str, str]] = []
        
        # Load or create vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocab()
        
        # Load or create merges
        if merges_file and os.path.exists(merges_file):
            self.load_merges(merges_file)
        else:
            self._build_default_merges()
    
    def _build_default_vocab(self):
        """Build default vocabulary with special tokens and basic characters."""
        # Start with special tokens
        self._vocab = dict(self.SPECIAL_TOKENS)
        
        # Add basic ASCII characters
        for i in range(256):
            char = chr(i)
            if char.isprintable() or char.isspace():
                token = f'<byte_{i}>'
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)
        
        # Add common English words and subwords
        english_tokens = [
            # Common words
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
            'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
            'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
            'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
            'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us', 'is', 'was', 'are', 'were', 'been', 'has',
            'had', 'did', 'does', 'doing', 'done',
            # Common prefixes/suffixes
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment',
            'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ize', 'ise',
            # Common Japanese kana and basic kanji
            'あ', 'い', 'う', 'え', 'お',
            'か', 'き', 'く', 'け', 'こ',
            'さ', 'し', 'す', 'せ', 'そ',
            'た', 'ち', 'つ', 'て', 'と',
            'な', 'に', 'ぬ', 'ね', 'の',
            'は', 'ひ', 'ふ', 'へ', 'ほ',
            'ま', 'み', 'む', 'め', 'も',
            'や', 'ゆ', 'よ',
            'ら', 'り', 'る', 'れ', 'ろ',
            'わ', 'を', 'ん',
            'が', 'ぎ', 'ぐ', 'げ', 'ご',
            'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
            'だ', 'ぢ', 'づ', 'で', 'ど',
            'ば', 'び', 'ぶ', 'べ', 'ぼ',
            'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ',
            # Hiragana combinations
            'きゃ', 'きゅ', 'きょ',
            'しゃ', 'しゅ', 'しょ',
            'ちゃ', 'ちゅ', 'ちょ',
            'にゃ', 'にゅ', 'にょ',
            'ひゃ', 'ひゅ', 'ひょ',
            'みゃ', 'みゅ', 'みょ',
            'りゃ', 'りゅ', 'りょ',
            'ぎゃ', 'ぎゅ', 'ぎょ',
            'じゃ', 'じゅ', 'じょ',
            'びゃ', 'びゅ', 'びょ',
            'ぴゃ', 'ぴゅ', 'ぴょ',
            # Common Japanese words
            'です', 'ます', 'した', 'する', 'ある', 'いる', 'なる', 'れる',
            'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
            'ここ', 'そこ', 'あそこ', 'どこ',
            '私', '僕', '俺', 'あなた', '彼', '彼女', 'たち',
            '日本', '語', '言葉', '文', '本', '事', '時', '人',
            '年', '月', '日', '週', '時間', '分', '秒',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '百', '千', '万', '円',
            '大', '小', '高', '低', '新', '古', '長', '短',
            '上', '下', '中', '外', '前', '後', '左', '右',
            '行く', '来る', '見る', '聞く', '言う', '話す', '読む', '書く',
            '食べる', '飲む', '買う', '売る', '待つ', '持つ', '死ぬ', '遊ぶ',
            '良い', '悪い', '大きい', '小さい', '多い', '少ない', '新しい', '古い',
            '楽しい', '難しい', '易しい', '忙しい', '嬉しい', '悲しい',
            # Punctuation
            '。', '、', '！', '？', '…', '・', '「', '」', '『', '』',
            '（', '）', '［', '］', '｛', '｝',
            # Numbers and symbols
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}',
            '"', "'", '`', '@', '#', '$', '%', '^', '&', '*', '+', '=', '/', '\\',
            '<', '>', '|', '~',
            # Whitespace
            ' ', '\n', '\t',
        ]
        
        for token in english_tokens:
            if token not in self._vocab and len(self._vocab) < self.vocab_size:
                self._vocab[token] = len(self._vocab)
        
        # Add common character bigrams
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for c1 in chars:
            for c2 in chars:
                bigram = c1 + c2
                if bigram not in self._vocab and len(self._vocab) < self.vocab_size:
                    self._vocab[bigram] = len(self._vocab)
        
        # Build inverse vocabulary
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
    
    def _build_default_merges(self):
        """Build default BPE merge rules."""
        # Common BPE merges for English
        common_pairs = [
            ('t', 'h'), ('h', 'e'), ('e', 'r'), ('i', 'n'), ('n', 'g'),
            ('e', 'd'), ('a', 'n'), ('o', 'n'), ('e', 'n'), ('a', 't'),
            ('i', 'o'), ('o', 'n'), ('s', 't'), ('e', 's'), ('e', 'n'),
            ('r', 'e'), ('a', 'r'), ('o', 'r'), ('l', 'y'), ('i', 'c'),
            ('a', 'l'), ('i', 't'), ('v', 'e'), ('f', 'o'), ('w', 'a'),
            ('h', 'a'), ('w', 'h'), ('o', 'u'), ('o', 'w'), ('i', 's'),
            ('a', 's'), ('i', 'd'), ('e', 'l'), ('a', 'c'), ('o', 'm'),
            ('p', 'r'), ('c', 'o'), ('m', 'e'), ('d', 'i'), ('p', 'e'),
            ('s', 'e'), ('u', 'n'), ('m', 'a'), ('b', 'e'), ('o', 't'),
            # Japanese merges
            ('で', 'す'), ('ま', 'す'), ('し', 'た'), ('こ', 'の'),
            ('あ', 'る'), ('い', 'る'), ('な', 'る'), ('れ', 'る'),
            ('日', '本'), ('で', 'き'), ('行', 'く'), ('来', 'る'),
            ('見', 'る'), ('食', 'べ'), ('大', 'き'), ('小', 'さ'),
            ('良', 'い'), ('新', 'し'), ('難', 'し'), ('楽', 'し'),
        ]
        
        self._merges = common_pairs[:min(len(common_pairs), self.vocab_size // 10)]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization."""
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into initial word tokens."""
        # Simple word segmentation
        # For Japanese, use character-based approach
        # For English, split on whitespace and punctuation
        tokens = []
        current_word = ''
        
        for char in text:
            if char.isspace():
                if current_word:
                    tokens.append(current_word)
                    current_word = ''
                tokens.append(char)
            elif char in '。、！？…・「」『』（）［］｛｝':
                if current_word:
                    tokens.append(current_word)
                    current_word = ''
                tokens.append(char)
            elif char.isascii() and not char.isalnum():
                if current_word:
                    tokens.append(current_word)
                    current_word = ''
                tokens.append(char)
            else:
                # Check if switching between CJK and Latin
                if current_word:
                    last_char = current_word[-1]
                    is_last_cjk = '\u4e00' <= last_char <= '\u9fff' or '\u3040' <= last_char <= '\u309f' or '\u30a0' <= last_char <= '\u30ff'
                    is_curr_cjk = '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'
                    if is_last_cjk != is_curr_cjk:
                        tokens.append(current_word)
                        current_word = ''
                current_word += char
        
        if current_word:
            tokens.append(current_word)
        
        return tokens
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE to a single word."""
        if word in self._vocab:
            return [word]
        
        # Start with characters
        word_tokens = list(word)
        
        # Apply merges
        for merge_left, merge_right in self._merges:
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == merge_left and word_tokens[i + 1] == merge_right:
                    merged = merge_left + merge_right
                    if merged in self._vocab:
                        new_tokens.append(merged)
                        i += 2
                        continue
                new_tokens.append(word_tokens[i])
                i += 1
            word_tokens = new_tokens
        
        return word_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add <s> and </s> tokens
        
        Returns:
            List of token IDs
        """
        text = self._preprocess_text(text)
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.SPECIAL_TOKENS['<s>'])
        
        # Split into words
        words = self._get_word_tokens(text)
        
        for word in words:
            # Try to find in vocabulary directly
            if word in self._vocab:
                token_ids.append(self._vocab[word])
            else:
                # Apply BPE
                bpe_tokens = self._apply_bpe(word)
                for token in bpe_tokens:
                    if token in self._vocab:
                        token_ids.append(self._vocab[token])
                    else:
                        # Encode as bytes
                        for byte in token.encode('utf-8', errors='ignore'):
                            byte_token = f'<byte_{byte}>'
                            if byte_token in self._vocab:
                                token_ids.append(self._vocab[byte_token])
                            else:
                                token_ids.append(self.SPECIAL_TOKENS['<unk>'])
        
        if add_special_tokens:
            token_ids.append(self.SPECIAL_TOKENS['</s>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self._inv_vocab:
                token = self._inv_vocab[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                
                # Handle byte tokens
                if token.startswith('<byte_') and token.endswith('>'):
                    try:
                        byte_val = int(token[6:-1])
                        tokens.append(chr(byte_val))
                    except ValueError:
                        tokens.append('')
                else:
                    tokens.append(token)
        
        # Join tokens - handle Japanese properly (no spaces)
        result = ''
        for i, token in enumerate(tokens):
            if i > 0:
                # Don't add space before Japanese characters or punctuation
                prev_token = tokens[i - 1]
                if not (token[0] in '。、！？…・「」『』（）［］｛｝' or 
                        '\u4e00' <= token[0] <= '\u9fff' or
                        '\u3040' <= token[0] <= '\u309f' or
                        '\u30a0' <= token[0] <= '\u30ff' or
                        prev_token[-1] in '([{\'"-_'):
                    if not (prev_token[-1] in '。、！？…・「」『』（）［］｛｝'):
                        if token[0] not in ')]}\'"-_.,;:!?':
                            result += ' '
            result += token
        
        return result
    
    def save_pretrained(self, path: str):
        """Save tokenizer to a directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(path, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges_path = os.path.join(path, 'merges.txt')
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write('#version: 0.2\n')
            for left, right in self._merges:
                f.write(f'{left} {right}\n')
    
    def load_pretrained(self, path: str):
        """Load tokenizer from a directory."""
        vocab_path = os.path.join(path, 'vocab.json')
        if os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        
        merges_path = os.path.join(path, 'merges.txt')
        if os.path.exists(merges_path):
            self.load_merges(merges_path)
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
    
    def load_merges(self, merges_file: str):
        """Load BPE merges from text file."""
        self._merges = []
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    self._merges.append((parts[0], parts[1]))
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.SPECIAL_TOKENS['<pad>']
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.SPECIAL_TOKENS['</s>']
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.SPECIAL_TOKENS['<s>']
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.SPECIAL_TOKENS['<unk>']
    
    @property
    def mask_token_id(self) -> int:
        """Mask token ID."""
        return self.SPECIAL_TOKENS['<mask>']
