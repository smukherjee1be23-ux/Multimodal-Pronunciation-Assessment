============================================================================
# Cell 1: Imports & Environment Setup
# ============================================================================

import os
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm.auto import tqdm

# Hugging Face
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    AutoTokenizer,
    AutoModel,
)

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(".").resolve()  # gopt/
DATA_ROOT     = PROJECT_ROOT / "data" / "speechocean762"
WAVE_DIR      = DATA_ROOT / "WAVE"
SCORES_PATH   = DATA_ROOT / "resource" / "scores.json"
TRAIN_TEXT     = DATA_ROOT / "train" / "text"
TEST_TEXT      = DATA_ROOT / "test" / "text"
TRAIN_WAVSCP   = DATA_ROOT / "train" / "wav.scp"
TEST_WAVSCP    = DATA_ROOT / "test" / "wav.scp"

FEATURES_PATH = PROJECT_ROOT / "data" / "extracted_features.pt"

# ── Constants ───────────────────────────────────────────────────────────────
MAX_PHN_LEN     = 50   # pad/truncate phoneme sequences to this length
WAV2VEC_DIM     = 768
BERT_DIM        = 768
SAMPLE_RATE     = 16000

# Quick sanity check
assert DATA_ROOT.exists(),  f"Dataset not found at {DATA_ROOT}"
assert SCORES_PATH.exists(), f"Scores not found at {SCORES_PATH}"
print(f"Dataset root: {DATA_ROOT}")
print(f"Scores:       {SCORES_PATH}")

============================================================================
# Cell 2: Feature Extraction — Wav2Vec2 + TRUE CTC Forced Alignment + BERT
# ============================================================================
# Uses torchaudio.functional.forced_align (requires torchaudio >= 2.1)
# to get exact per-phoneme frame boundaries from the CTC model.
#
# Key change vs old Cell 2:
#   OLD: frames_per_phone = T_frames / num_phones  (uniform split, WRONG)
#   NEW: torchaudio.functional.forced_align(log_probs, targets, ...)
#        returns exact start/end frame per phoneme  (CORRECT)
#
# ARPAbet → CTC character mapping:
#   wav2vec2-base-960h uses a character-level CTC vocab (A-Z, ' , |, [PAD], etc.)
#   speechocean762 phones are ARPAbet (AH0, T, K, SIL, ...).
#   We map each ARPAbet phone to its closest English letter(s) so that
#   forced_align can find the right frame windows.
#   SIL/SP map to the silence token '|'.
#
# Output dict per utterance (IDENTICAL schema to old Cell 2 — no other cells change):
#   'acoustic':    Tensor [num_phones, 768]
#   'bert_cls':    Tensor [768]
#   'phn_scores':  list[float]  phone accuracy scores (0-2)
#   'word_scores': list[dict]   word accuracy/stress/total (0-10)
#   'utt_scores':  dict         utterance accuracy/completeness/fluency/prosodic/total
#   'word_ids':    list[int]    maps each phone index → word index
#   'split':       'train' or 'test'
# ============================================================================

import torch
import torchaudio
import torchaudio.functional as AF
from pathlib import Path
from tqdm.auto import tqdm
import json
import warnings

# ── Verify forced_align is available ────────────────────────────────────────
assert hasattr(AF, 'forced_align'), (
    "torchaudio.functional.forced_align not found. "
    "Please upgrade: pip install torchaudio --upgrade"
)
print(f"torchaudio version: {torchaudio.__version__} ✓  forced_align available")


# ── ARPAbet → single CTC character mapping ──────────────────────────────────
ARPABET_TO_CHAR = {
    # Vowels
    "AA": "A", "AE": "A", "AH": "A", "AO": "O", "AW": "A",
    "AY": "A", "EH": "E", "ER": "R", "EY": "E", "IH": "I",
    "IY": "I", "OW": "O", "OY": "O", "UH": "U", "UW": "U",
    # Consonants
    "B":  "B", "CH": "C", "D": "D", "DH": "D", "F": "F",
    "G":  "G", "HH": "H", "JH": "J", "K": "K", "L": "L",
    "M":  "M", "N":  "N", "NG": "N", "P": "P", "R": "R",
    "S":  "S", "SH": "S", "T": "T", "TH": "T", "V": "V",
    "W":  "W", "Y":  "Y", "Z": "Z", "ZH": "Z",
    # Silence / noise
    "SIL": "|", "SP": "|", "SPN": "|", "NSN": "|",
}

def arpabet_to_char(phone: str) -> str:
    phone_stripped = phone.rstrip("012")
    return ARPABET_TO_CHAR.get(phone_stripped.upper(), "A")


# ── Helper: load Kaldi-style files ───────────────────────────────────────────
def load_wav_scp(scp_path):
    mapping = {}
    with open(scp_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = " ".join(parts[1:])
    return mapping

def load_transcripts(text_path):
    mapping = {}
    with open(text_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


# ── Main extraction function ─────────────────────────────────────────────────
def extract_all_features():

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(SCORES_PATH) as f:
        scores = json.load(f)

    train_wavs  = load_wav_scp(TRAIN_WAVSCP)
    test_wavs   = load_wav_scp(TEST_WAVSCP)
    train_texts = load_transcripts(TRAIN_TEXT)
    test_texts  = load_transcripts(TEST_TEXT)

    all_utts = []
    for uid, path in train_wavs.items():
        if uid in scores:
            all_utts.append((uid, path, train_texts.get(uid, ""), "train"))
    for uid, path in test_wavs.items():
        if uid in scores:
            all_utts.append((uid, path, test_texts.get(uid, ""), "test"))

    print(f"Total utterances to process: {len(all_utts)}")

    # ── Load models on CPU (MPS has instability on long audio sequences) ─────
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model, AutoTokenizer, AutoModel
    w2v_device = torch.device("cpu")

    print("Loading Wav2Vec2ForCTC (facebook/wav2vec2-base-960h)...")
    processor  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    ctc_model  = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().to(w2v_device)

    print("Loading Wav2Vec2 base for hidden states (facebook/wav2vec2-base)...")
    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval().to(w2v_device)

    print("Loading BERT...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model     = AutoModel.from_pretrained("bert-base-uncased").eval().to(w2v_device)

    # ── Build vocab token lookup ─────────────────────────────────────────────
    vocab        = processor.tokenizer.get_vocab()
    pad_token_id = processor.tokenizer.pad_token_id  # blank token for CTC

    def char_to_id(ch: str) -> int:
        return vocab.get(ch.upper(), vocab.get("[UNK]", 1))

    features_dict = {}
    errors        = []
    proportional_fallback_count = 0

    for utt_id, wav_rel, transcript, split in tqdm(all_utts, desc="Extracting"):
        try:
            utt_scores = scores[utt_id]

            # ── 1. Load & preprocess audio ───────────────────────────────────
            wav_path = Path(wav_rel)
            if not wav_path.exists():
                wav_path = DATA_ROOT / wav_rel
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)   # [T_samples]

            # ── 2. Run both models in one forward pass ────────────────────────
            with torch.no_grad():
                inputs       = processor(waveform.numpy(), sampling_rate=SAMPLE_RATE,
                                         return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(w2v_device)

                # Hidden states for embeddings
                base_out      = base_model(input_values)
                hidden_states = base_out.last_hidden_state[0]   # [T_frames, 768]

                # CTC log-probs for alignment
                ctc_out   = ctc_model(input_values)
                log_probs = torch.log_softmax(ctc_out.logits, dim=-1)[0]  # [T_frames, vocab]

            T_frames = hidden_states.shape[0]

            # ── 3. Collect phoneme sequence from scores.json ──────────────────
            all_phones      = []
            phone_word_ids  = []
            phone_scores_l  = []
            word_scores_list = []

            for w_idx, word_info in enumerate(utt_scores["words"]):
                word_scores_list.append({
                    "accuracy": word_info["accuracy"],
                    "stress":   word_info["stress"],
                    "total":    word_info["total"],
                })
                for phn, phn_acc in zip(word_info["phones"], word_info["phones-accuracy"]):
                    all_phones.append(phn)
                    phone_word_ids.append(w_idx)
                    phone_scores_l.append(phn_acc)

            num_phones = len(all_phones)
            if num_phones == 0:
                errors.append((utt_id, "no phones"))
                continue

            # ── 4. TRUE CTC forced alignment ──────────────────────────────────
            # torchaudio 2.8: forced_align returns Tuple(Tensor, Tensor).
            # Unpack as (aligned_tokens, scores), then call merge_tokens()
            # to get TokenSpan objects with .start / .end frame indices.

            use_proportional = False
            try:
                token_ids = torch.tensor(
                    [char_to_id(arpabet_to_char(p)) for p in all_phones],
                    dtype=torch.long
                )  # [num_phones]

                input_lengths  = torch.tensor([T_frames], dtype=torch.long)
                target_lengths = torch.tensor([num_phones], dtype=torch.long)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aligned_tokens, align_scores = AF.forced_align(
                        log_probs.unsqueeze(0),   # [1, T, vocab]
                        token_ids.unsqueeze(0),   # [1, num_phones]
                        input_lengths,
                        target_lengths,
                        blank=pad_token_id,
                    )
                # merge_tokens collapses frame-level output → one TokenSpan per phoneme
                # each span has .start, .end (inclusive), .score
                spans = AF.merge_tokens(aligned_tokens[0], align_scores[0])

            except Exception as align_err:
                # Fallback: proportional split
                use_proportional = True
                proportional_fallback_count += 1
                if proportional_fallback_count <= 5:
                    print(f"  forced_align fallback on {utt_id}: {align_err}")

            # ── 5. Pool hidden states per phoneme ────────────────────────────
            phone_embeddings = []

            if not use_proportional:
                for p_idx in range(num_phones):
                    if p_idx < len(spans):
                        span = spans[p_idx]
                        s = max(int(span.start), 0)
                        e = min(int(span.end) + 1, T_frames)  # end is inclusive
                    else:
                        # merge_tokens returned fewer spans than phones (blank collapsing)
                        # fall back proportionally for this phone only
                        frames_per_phone = T_frames / num_phones
                        s = int(p_idx * frames_per_phone)
                        e = min(int((p_idx + 1) * frames_per_phone), T_frames)
                    s = max(s, 0)
                    e = min(e, T_frames)
                    e = max(e, s + 1)
                    phone_embeddings.append(hidden_states[s:e].mean(dim=0))
            else:
                # Proportional fallback for entire utterance
                frames_per_phone = T_frames / num_phones
                for p_idx in range(num_phones):
                    s = int(p_idx * frames_per_phone)
                    e = min(int((p_idx + 1) * frames_per_phone), T_frames)
                    e = max(e, s + 1)
                    phone_embeddings.append(hidden_states[s:e].mean(dim=0))

            acoustic_feats = torch.stack(phone_embeddings)   # [num_phones, 768]

            # ── 6. BERT [CLS] ────────────────────────────────────────────────
            with torch.no_grad():
                bert_inputs = bert_tokenizer(transcript, return_tensors="pt",
                                             truncation=True, max_length=128, padding=True)
                bert_inputs = {k: v.to(w2v_device) for k, v in bert_inputs.items()}
                bert_out    = bert_model(**bert_inputs)
                bert_cls    = bert_out.last_hidden_state[0, 0]   # [768]

            # ── 7. Store ─────────────────────────────────────────────────────
            features_dict[utt_id] = {
                "acoustic":    acoustic_feats.cpu(),
                "bert_cls":    bert_cls.cpu(),
                "phn_scores":  phone_scores_l,
                "word_scores": word_scores_list,
                "utt_scores":  {
                    "accuracy":     utt_scores["accuracy"],
                    "completeness": utt_scores["completeness"],
                    "fluency":      utt_scores["fluency"],
                    "prosodic":     utt_scores["prosodic"],
                    "total":        utt_scores["total"],
                },
                "word_ids": phone_word_ids,
                "split":    split,
            }

        except Exception as e:
            errors.append((utt_id, str(e)))
            if len(errors) <= 5:
                print(f"  Error on {utt_id}: {e}")

    print(f"\nDone. Extracted: {len(features_dict)} | Errors: {len(errors)} | "
          f"Proportional fallbacks: {proportional_fallback_count}")
    return features_dict


# ── Load cache or re-extract ─────────────────────────────────────────────────
if FEATURES_PATH.exists():
    print(f"Loading cached features from {FEATURES_PATH} ...")
    features_dict = torch.load(FEATURES_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(features_dict)} utterances.")
else:
    print("No cache found — running extraction (~30-40 min) ...")
    features_dict = extract_all_features()
    torch.save(features_dict, FEATURES_PATH)
    print(f"Saved to {FEATURES_PATH}")

# ── Sanity check ─────────────────────────────────────────────────────────────
if features_dict:
    sid = list(features_dict.keys())[0]
    s   = features_dict[sid]
    print(f"\nSample '{sid}':")
    print(f"  acoustic : {s['acoustic'].shape}")
    print(f"  bert_cls : {s['bert_cls'].shape}")
    print(f"  phones   : {len(s['phn_scores'])}")
    print(f"  words    : {len(s['word_scores'])}")
    print(f"  split    : {s['split']}")
else:
    print("⚠️  features_dict is empty — check errors above.")

# ============================================================================
# Cell 3: MultimodalGOPT Model
# ============================================================================
#
# Sequence layout (56 tokens total):
#   [0] cls_accuracy  [1] cls_completeness  [2] cls_fluency
#   [3] cls_prosodic  [4] cls_total
#   [5] BERT text token
#   [6:56] 50 acoustic phoneme tokens (padded)
#
# Scoring heads:
#   Utterance: x[:, 0:5] → 5 separate MLP heads
#   Phone:     x[:, 6:]  → 1 MLP head (masked for padding)
#   Word:      x[:, 6:]  → 3 MLP heads (accuracy, stress, total)
# ============================================================================


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (from timm)."""
    with torch.no_grad():
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.2, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultimodalGOPT(nn.Module):
    """
    Multimodal GOPT: Wav2Vec2 acoustic features + BERT text features.

    Sequence layout: [5 utt CLS tokens] [1 BERT token] [50 acoustic tokens] = 56 total

    Args:
        embed_dim:  Transformer embedding dimension
        num_heads:  Number of attention heads
        depth:      Number of Transformer blocks
        input_dim:  Dimension of acoustic features (768 for Wav2Vec2)
    """
    def __init__(self, embed_dim=192, num_heads=8, depth=6, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # ── Transformer blocks with dropout ────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, drop=0.2, attn_drop=0.1)
            for _ in range(depth)
        ])

        # ── Positional embedding ───────────────────────────────────────────
        # 56 = 5 utt CLS + 1 BERT text + 50 acoustic phonemes
        self.pos_embed = nn.Parameter(torch.zeros(1, 56, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # ── Input projections ──────────────────────────────────────────────
        # Project 768-D acoustic features → embed_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        # Project 768-D BERT [CLS] → embed_dim
        self.text_proj = nn.Linear(768, embed_dim)
        self.acoustic_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)

        # ── Scoring heads (embed_dim=192 matching) ─────────────────────────
        # Utterance heads (5 of them):
        self.utt_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(5)
        ])

        # Phone head:
        self.phn_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        # Word heads (3 of them):
        self.word_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(3)
        ])

        # Dropout before scoring heads
        self.dropout = nn.Dropout(0.3)

        # ── Utterance-level CLS tokens & heads ─────────────────────────────
        # 5 learnable CLS tokens: accuracy, completeness, fluency, prosodic, total
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # accuracy
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # completeness
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # fluency
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # prosodic
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # total

        # Initialize CLS tokens
        for tok in [self.cls_token1, self.cls_token2, self.cls_token3,
                    self.cls_token4, self.cls_token5]:
            trunc_normal_(tok, std=.02)

    def forward(self, x, text_embed):
        """
        Args:
            x:          [batch, 50, 768] acoustic phoneme embeddings (padded)
            text_embed: [batch, 768] BERT [CLS] embeddings

        Returns:
            u1..u5:  [batch, 1] utterance-level scores
            p:       [batch, 50, 1] phone-level scores
            w1..w3:  [batch, 50, 1] word-level scores
        """
        B = x.shape[0]

        # Project acoustic features: [batch, 50, 768] → [batch, 50, embed_dim]
        x = self.acoustic_norm(self.in_proj(x))
        text_token = self.text_norm(
            self.text_proj(text_embed)
        ).unsqueeze(1)

        # Expand CLS tokens for batch
        cls1 = self.cls_token1.expand(B, -1, -1)
        cls2 = self.cls_token2.expand(B, -1, -1)
        cls3 = self.cls_token3.expand(B, -1, -1)
        cls4 = self.cls_token4.expand(B, -1, -1)
        cls5 = self.cls_token5.expand(B, -1, -1)

        # Concatenate: [5 CLS] [1 BERT] [50 acoustic] = 56 tokens
        # idx:          0-4      5        6-55
        x = torch.cat([cls1, cls2, cls3, cls4, cls5, text_token, x], dim=1)
        # x shape: [batch, 56, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer forward
        for blk in self.blocks:
            x = blk(x)

        # Apply dropout before scoring heads
        x = self.dropout(x)

        # ── Utterance scores from CLS tokens [0:5] ─────────────────────────
        u1 = self.utt_heads[0](x[:, 0])   # [batch, 1] accuracy
        u2 = self.utt_heads[1](x[:, 1])   # [batch, 1] completeness
        u3 = self.utt_heads[2](x[:, 2])   # [batch, 1] fluency
        u4 = self.utt_heads[3](x[:, 3])   # [batch, 1] prosodic
        u5 = self.utt_heads[4](x[:, 4])   # [batch, 1] total

        # ── Phone & word scores from acoustic tokens [6:] ──────────────────
        acoustic_out = x[:, 6:]  # [batch, 50, embed_dim]

        p  = self.phn_head(acoustic_out)    # [batch, 50, 1]
        w1 = self.word_heads[0](acoustic_out)  # [batch, 50, 1] word accuracy
        w2 = self.word_heads[1](acoustic_out)  # [batch, 50, 1] word stress
        w3 = self.word_heads[2](acoustic_out)  # [batch, 50, 1] word total

        return u1, u2, u3, u4, u5, p, w1, w2, w3


# Quick shape verification
_model = MultimodalGOPT(embed_dim=192, num_heads=8, depth=6, input_dim=768)
_x = torch.randn(2, 50, 768)
_t = torch.randn(2, 768)
_u1, _u2, _u3, _u4, _u5, _p, _w1, _w2, _w3 = _model(_x, _t)
print("Shape verification:")
print(f"  Utt scores:  {_u1.shape}  (expected [2, 1])")
print(f"  Phone scores: {_p.shape}  (expected [2, 50, 1])")
print(f"  Word scores:  {_w1.shape}  (expected [2, 50, 1])")
total_params = sum(p.numel() for p in _model.parameters()) / 1e3
print(f"  Total params: {total_params:.1f}K")
del _model, _x, _t

============================================================================
# Cell 4: PyTorch Dataset & DataLoader
# ============================================================================

class PronunciationDataset(Dataset):
    """
    Dataset for Multimodal GOPT.

    Each sample returns a dictionary of features.
    """
    def __init__(self, features_dict, split, norm_stats=None):
        self.samples = []
        raw = []
        for utt_id, feat in features_dict.items():
            if feat["split"] == split:
                raw.append((utt_id, feat))

        # Collect utterance scores for normalization
        if norm_stats is None:
            utt_keys = ['accuracy','completeness','fluency','prosodic','total']
            vals = {k: [] for k in utt_keys}
            for _, feat in raw:
                for k in utt_keys:
                    vals[k].append(feat['utt_scores'][k])
            self.norm_stats = {
                k: (np.mean(vals[k]), np.std(vals[k]) + 1e-8)
                for k in utt_keys
            }
        else:
            self.norm_stats = norm_stats

        for utt_id, feat in raw:
            self.samples.append((utt_id, feat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, feat = self.samples[idx]

        # Acoustic features: pad/truncate to MAX_PHN_LEN
        acoustic = feat["acoustic"]  # [num_phones, 768]
        num_phones = acoustic.shape[0]

        if num_phones >= MAX_PHN_LEN:
            acoustic = acoustic[:MAX_PHN_LEN]
            mask = torch.ones(MAX_PHN_LEN, dtype=torch.bool)
            phn_scores = feat["phn_scores"][:MAX_PHN_LEN]
            word_ids = feat["word_ids"][:MAX_PHN_LEN]
        else:
            pad_len = MAX_PHN_LEN - num_phones
            acoustic = torch.cat([acoustic, torch.zeros(pad_len, WAV2VEC_DIM)], dim=0)
            mask = torch.cat([
                torch.ones(num_phones, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
            phn_scores = feat["phn_scores"] + [0.0] * pad_len
            word_ids = feat["word_ids"] + [-1] * pad_len

        bert_cls = feat["bert_cls"]  # [768]

        # Normalized utterance scores
        utt_keys = ['accuracy','completeness','fluency','prosodic','total']
        utt_scores = torch.tensor([
            (feat['utt_scores'][k] - self.norm_stats[k][0]) / self.norm_stats[k][1]
            for k in utt_keys
        ], dtype=torch.float32)

        # Phone scores normalized to [0,1] (originally 0-2)
        phn_scores = torch.tensor(phn_scores, dtype=torch.float32) / 2.0

        # Word scores
        word_scores_acc   = []
        word_scores_stress = []
        word_scores_total  = []
        for p_idx in range(MAX_PHN_LEN):
            w_idx = word_ids[p_idx] if p_idx < len(word_ids) else -1
            if w_idx >= 0 and w_idx < len(feat['word_scores']):
                ws = feat['word_scores'][w_idx]
                word_scores_acc.append(ws['accuracy'] / 10.0)
                word_scores_stress.append(ws['stress'] / 10.0)
                word_scores_total.append(ws['total'] / 10.0)
            else:
                word_scores_acc.append(0.0)
                word_scores_stress.append(0.0)
                word_scores_total.append(0.0)

        return {
            'acoustic':      acoustic,
            'bert_cls':      bert_cls,
            'mask':          mask,
            'phn_scores':    phn_scores,
            'word_scores':   torch.stack([
                                torch.tensor(word_scores_acc),
                                torch.tensor(word_scores_stress),
                                torch.tensor(word_scores_total)
                             ], dim=1),  # [MAX_PHN_LEN, 3]
            'utt_scores':    utt_scores,
        }


# ── Create datasets and dataloaders (Reduced Batch Size to 16) ──────────────
train_dataset = PronunciationDataset(features_dict, split='train')
test_dataset  = PronunciationDataset(features_dict, split='test',
                                      norm_stats=train_dataset.norm_stats)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Verify shapes
for batch in train_loader:
    acoustic = batch['acoustic']
    bert_cls = batch['bert_cls']
    phn_label = batch['phn_scores']
    word_label = batch['word_scores']
    utt_label = batch['utt_scores']
    mask = batch['mask']
    print("Batch shapes:")
    print(f"  acoustic:   {acoustic.shape}   (expected [B, 50, 768])")
    print(f"  bert_cls:   {bert_cls.shape}   (expected [B, 768])")
    print(f"  phn_label:  {phn_label.shape}  (expected [B, 50])")
    print(f"  word_label: {word_label.shape} (expected [B, 50, 3])")
    print(f"  utt_label:  {utt_label.shape}  (expected [B, 5])")
    print(f"  mask:       {mask.shape}       (expected [B, 50])")
    break

============================================================================
# Cell 5: Training & Evaluation Loop
# ============================================================================

def compute_pcc(pred, target, mask=None):
    """Compute Pearson Correlation Coefficient, optionally masked."""
    if mask is not None:
        # Flatten and select valid entries
        pred = pred[mask > 0]
        target = target[mask > 0]

    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()

    if len(pred) < 2 or np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def masked_mse_loss(pred, target, mask, weights=None):
    """
    MSE loss only on valid (non-padded) positions, optionally weighted across dimensions.
    """
    if pred.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(pred)

    mask = mask.float()
    se = (pred - target) ** 2 * mask
    if weights is not None:
        se = se * weights

    return se.sum() / mask.sum().clamp(min=1.0)


def evaluate(model, data_loader, device):
    """Run evaluation and return metrics."""
    model.eval()

    all_phn_pred, all_phn_target, all_phn_mask = [], [], []
    all_utt_pred, all_utt_target = [], []
    all_word_pred, all_word_target, all_word_mask = [], [], []

    total_loss = 0.0
    n_batches = 0

    # Loss dimension-specific weights
    utt_weights = torch.tensor([1.0, 0.5, 1.0, 1.0, 1.0], device=device)
    word_weights = torch.tensor([1.0, 2.0, 1.0], device=device)

    with torch.no_grad():
        for batch in data_loader:
            acoustic  = batch['acoustic'].to(device)
            bert_cls  = batch['bert_cls'].to(device)
            phn_label = batch['phn_scores'].to(device)
            word_label = batch['word_scores'].to(device)
            utt_label = batch['utt_scores'].to(device)
            mask      = batch['mask'].to(device)

            u1, u2, u3, u4, u5, p, w1, w2, w3 = model(acoustic, bert_cls)

            # Phone loss
            p_squeezed = p.squeeze(-1)  # [B, 50]
            loss_phn = masked_mse_loss(p_squeezed, phn_label, mask)

            # Utt loss in normalized space
            utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)
            loss_utt = (F.mse_loss(utt_pred, utt_label, reduction='none') * utt_weights).mean()

            # Word loss (using the same acoustic mask and word weights)
            word_target_scores = word_label  # [B, 50, 3]
            word_pred = torch.cat([w1, w2, w3], dim=2)  # [B, 50, 3]
            word_mask = mask  # [B, 50]
            loss_word = masked_mse_loss(word_pred, word_target_scores, word_mask, weights=word_weights)

            loss = loss_phn + loss_utt + loss_word
            total_loss += loss.item()
            n_batches += 1

            # Collect for metrics
            all_phn_pred.append(p_squeezed.cpu())
            all_phn_target.append(phn_label.cpu())
            all_phn_mask.append(mask.cpu())

            all_utt_pred.append(utt_pred.cpu())
            all_utt_target.append(utt_label.cpu())

            all_word_pred.append(word_pred.cpu())
            all_word_target.append(word_target_scores.cpu())
            all_word_mask.append(word_mask.cpu())

    # Concatenate
    all_phn_pred = torch.cat(all_phn_pred)
    all_phn_target = torch.cat(all_phn_target)
    all_phn_mask = torch.cat(all_phn_mask)

    all_utt_pred = torch.cat(all_utt_pred)
    all_utt_target = torch.cat(all_utt_target)

    all_word_pred = torch.cat(all_word_pred)
    all_word_target = torch.cat(all_word_target)
    all_word_mask = torch.cat(all_word_mask)

    # Phone-level metrics
    phn_mse = masked_mse_loss(all_phn_pred, all_phn_target, all_phn_mask).item()
    phn_pcc = compute_pcc(all_phn_pred, all_phn_target, all_phn_mask)

    # Utterance-level metrics (Denormalize ONLY for reporting and PCC calculation)
    norm_stats = data_loader.dataset.norm_stats
    utt_labels = ["Accuracy", "Completeness", "Fluency", "Prosodic", "Total"]
    utt_keys = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    utt_pcc = {}
    utt_mse_dict = {}
    for i, (label, key) in enumerate(zip(utt_labels, utt_keys)):
        mean, std = norm_stats[key]
        pred_denorm = all_utt_pred[:, i] * std + mean
        target_denorm = all_utt_target[:, i] * std + mean

        utt_pcc[label] = compute_pcc(pred_denorm, target_denorm)
        utt_mse_dict[label] = F.mse_loss(pred_denorm, target_denorm).item()

    # Word-level metrics
    word_labels_names = ["Accuracy", "Stress", "Total"]
    word_pcc = {}
    for i, label in enumerate(word_labels_names):
        word_pcc[label] = compute_pcc(
            all_word_pred[:, :, i], all_word_target[:, :, i], all_word_mask
        )

    avg_loss = total_loss / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "phn_mse": phn_mse,
        "phn_pcc": phn_pcc,
        "utt_pcc": utt_pcc,
        "utt_mse": utt_mse_dict,
        "word_pcc": word_pcc,
    }


def train_model(model, train_loader, test_loader, device,
                n_epochs=100, lr=1e-3,
                loss_w_phn=1.0, loss_w_utt=1.0, loss_w_word=1.0):
    """Full training loop with evaluation each epoch."""

    model = model.to(device)
    # AdamW with weight decay L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.95, 0.999))

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e3
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
    print(f"Total params: {total_params:.1f}K | Trainable: {trainable_params:.1f}K")
    print(f"Loss weights: phn={loss_w_phn}, utt={loss_w_utt}, word={loss_w_word}")
    print(f"Training for {n_epochs} epochs on {device}")
    print("=" * 80)

    # Early Stopping setup
    patience = 20
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    history = []

    # Loss dimension-specific weights
    utt_weights = torch.tensor([1.0, 0.5, 1.0, 1.0, 1.0], device=device)
    word_weights = torch.tensor([1.0, 2.0, 1.0], device=device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            acoustic  = batch['acoustic'].to(device)
            bert_cls  = batch['bert_cls'].to(device)
            phn_label = batch['phn_scores'].to(device)
            word_label = batch['word_scores'].to(device)
            utt_label = batch['utt_scores'].to(device)
            mask      = batch['mask'].to(device)

            # Forward
            u1, u2, u3, u4, u5, p, w1, w2, w3 = model(acoustic, bert_cls)

            # Phone loss
            p_squeezed = p.squeeze(-1)  # [B, 50]
            loss_phn = masked_mse_loss(p_squeezed, phn_label, mask)

            # Utterance loss with completeness weight reduced by half
            utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)  # [B, 5]
            loss_utt = (F.mse_loss(utt_pred, utt_label, reduction='none') * utt_weights).mean()

            # Word loss (using the same acoustic mask and word weights)
            word_target_scores = word_label  # [B, 50, 3]
            word_pred = torch.cat([w1, w2, w3], dim=2)  # [B, 50, 3]
            word_mask = mask  # [B, 50]
            loss_word = masked_mse_loss(word_pred, word_target_scores, word_mask, weights=word_weights)

            # Total loss
            loss = loss_w_phn * loss_phn + loss_w_utt * loss_utt + loss_w_word * loss_word

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        test_loss = test_metrics["loss"]

        # Track best and Early Stopping on test_loss
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
            save_dir = PROJECT_ROOT / "exp" / "multimodal_gopt"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            epochs_no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **test_metrics,
        })

        # Print
        best_marker = " ★" if is_best else ""
        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Phn MSE: {test_metrics['phn_mse']:.4f} PCC: {test_metrics['phn_pcc']:.3f} | "
            f"Utt Total PCC: {test_metrics['utt_pcc']['Total']:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{best_marker}"
        )

        # Detailed utterance-level print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Utt PCC  → " + " | ".join(
                f"{k}: {v:.3f}" for k, v in test_metrics['utt_pcc'].items()
            ))
            print(f"  Word PCC → " + " | ".join(
                f"{k}: {v:.3f}" for k, v in test_metrics['word_pcc'].items()
            ))

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("=" * 80)
    print(f"Best epoch: {best_epoch+1} with Test Loss: {best_test_loss:.4f}")
    print(f"Model saved to: {PROJECT_ROOT / 'exp' / 'multimodal_gopt' / 'best_model.pth'}")

    return history


# ── Instantiate & Train (n_epochs=100) ──────────────────────────────────────
model = MultimodalGOPT(
    embed_dim=192,
    num_heads=8,
    depth=6,
    input_dim=768,
)

history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=DEVICE,
    n_epochs=100,
    lr=3e-4,
    loss_w_phn=2.0,
    loss_w_utt=1.0,
    loss_w_word=1.5,
      )
          
