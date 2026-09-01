"""
models.py
---------
Preserves the original ContextRNN (LSTM + attention) and ContextNCF
(neural collaborative filtering) architectures exactly as designed by
the seniors' code, plus a SAFE loader that never crashes the app if
the trained weight files (rnn_model_trained2.pth, ncf_model_trained2.pth,
metadata2.pth) or the dataset are missing.

If loading fails for any reason, load_models() returns
(None, None, None, reason) and the recommender falls back to a
content-based-only scoring mode (see modules/recommender.py), which is
clearly labeled in the UI rather than silently pretending the deep
models ran.
"""

import os
import torch
import torch.nn as nn
from .config import MODEL_DIR


class ContextRNN(nn.Module):
    def __init__(self, num_songs, num_genres, num_vibes, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.song_emb = nn.Embedding(num_songs, embed_dim)
        self.genre_emb = nn.Embedding(num_genres, 8)
        self.vibe_emb = nn.Embedding(num_vibes, 8)
        self.context_fc = nn.Linear(6, 16)
        self.lstm = nn.LSTM(160, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, num_songs)

    def forward(self, seq, genre, vibe, ctx):
        s = self.song_emb(seq)
        g = self.genre_emb(genre).unsqueeze(1).repeat(1, seq.size(1), 1)
        v = self.vibe_emb(vibe).unsqueeze(1).repeat(1, seq.size(1), 1)
        c = self.context_fc(ctx).unsqueeze(1).repeat(1, seq.size(1), 1)
        x = torch.cat([s, g, v, c], dim=2)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)


class ContextNCF(nn.Module):
    def __init__(self, num_users, num_songs, num_genres, num_vibes):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 16)
        self.song_emb = nn.Embedding(num_songs, 32)
        self.genre_emb = nn.Embedding(num_genres, 8)
        self.vibe_emb = nn.Embedding(num_vibes, 8)
        self.mood_emb = nn.Embedding(6, 4)
        total_input = 16 + 32 + 8 + 8 + 4 + 1 + 1 + 1 + 1 + 1 + 1
        self.fc1 = nn.Linear(total_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, u, s, g, v, m, st, h, t, d, w, pop):
        x = torch.cat([self.user_emb(u), self.song_emb(s), self.genre_emb(g),
                        self.vibe_emb(v), self.mood_emb(m), st, h, t, d, w, pop], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def load_models():
    """Returns (metadata, rnn_model, ncf_model, error_reason).
    error_reason is None on success; otherwise the other three are None
    and the caller must fall back to content-based-only scoring."""
    meta_path = os.path.join(MODEL_DIR, "metadata2.pth")
    rnn_path = os.path.join(MODEL_DIR, "rnn_model_trained2.pth")
    ncf_path = os.path.join(MODEL_DIR, "ncf_model_trained2.pth")

    if not (os.path.exists(meta_path) and os.path.exists(rnn_path) and os.path.exists(ncf_path)):
        missing = [p for p in [meta_path, rnn_path, ncf_path] if not os.path.exists(p)]
        return None, None, None, f"Model file(s) not found: {missing}. Place trained weights in the project root to enable RNN/NCF scoring."

    try:
        metadata = torch.load(meta_path, map_location="cpu")
        rnn_model = ContextRNN(
            num_songs=metadata["num_songs"],
            num_genres=metadata["num_genres"],
            num_vibes=metadata["num_vibes"],
        )
        ncf_model = ContextNCF(
            num_users=metadata["num_users"],
            num_songs=metadata["num_songs"],
            num_genres=metadata["num_genres"],
            num_vibes=metadata["num_vibes"],
        )
        rnn_model.load_state_dict(torch.load(rnn_path, map_location="cpu"), strict=True)
        ncf_model.load_state_dict(torch.load(ncf_path, map_location="cpu"), strict=True)
        rnn_model.eval()
        ncf_model.eval()
        return metadata, rnn_model, ncf_model, None
    except Exception as e:
        return None, None, None, f"Failed to load trained models: {e}"
