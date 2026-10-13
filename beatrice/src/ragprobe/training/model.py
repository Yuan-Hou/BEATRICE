import torch
import torch.nn as nn


RAG_SIGNAL_CLASSES = (
    "Attention Bias",
    "Logic Drift",
    "Hierarchy Misalignment",
    "Over-correction",
    "Others",
    "Not Regression",
)


class GlobalHeadSelector(nn.Module):
    """
    Global head gate + head dropout.

    Input:
        x: [B, num_heads, T]

    Output:
        x: [B, num_heads, T]
    """
    def __init__(self, num_heads, head_dropout=0.1, use_gate=True):
        super().__init__()
        self.use_gate = use_gate

        if use_gate:
            # 2 * sigmoid(0) = 1, so initialization roughly preserves input magnitude
            self.head_logits = nn.Parameter(torch.zeros(1, num_heads, 1))
        else:
            self.register_parameter("head_logits", None)

        self.dropout = nn.Dropout1d(head_dropout) if head_dropout > 0 else nn.Identity()

    def forward(self, x):
        if self.use_gate:
            head_weights = 2.0 * torch.sigmoid(self.head_logits)
            x = x * head_weights

        # Dropout1d drops whole channels/heads, suitable for [B, C, T]
        x = self.dropout(x)
        return x

    def get_head_weights(self):
        if self.head_logits is None:
            return None
        return (2.0 * torch.sigmoid(self.head_logits)).squeeze()


class ConvBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RAGProbeCNN(nn.Module):
    def __init__(
        self,
        embed_weights,
        num_attn_channels=128,
        prob_channels=3,
        hidden_dim=64,
        noise_std=0.05,
        dropout=0.15,
        head_dropout=0.1,
        use_head_gate=True,
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
    ):
        super().__init__()

        self.num_attn_channels = num_attn_channels
        self.prob_channels = prob_channels
        self.noise_std = noise_std
        self.num_rag_classes = num_rag_classes
        self.rag_signal_classes = RAG_SIGNAL_CLASSES[:num_rag_classes]

        num_embeddings, embedding_dim = embed_weights.shape

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(embed_weights)
        self.embedding.weight.requires_grad = False

        self.head_selector = GlobalHeadSelector(
            num_heads=num_attn_channels,
            head_dropout=head_dropout,
            use_gate=use_head_gate,
        )

        self.attn_conv = ConvBranch(
            in_channels=num_attn_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.emb_conv = ConvBranch(
            in_channels=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.prob_conv = ConvBranch(
            in_channels=prob_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.attn_gate_layer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.emb_gate_layer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        self.post_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        class_pool_dim = hidden_dim * 2
        self.rag_class_head = nn.Sequential(
            nn.LayerNorm(class_pool_dim * 3),
            nn.Linear(class_pool_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_rag_classes),
        )
        self.aux_binary_head = nn.Sequential(
            nn.LayerNorm(class_pool_dim * 3),
            nn.Linear(class_pool_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2),
        )

    @staticmethod
    def _masked_pool(x, pad_mask=None):
        if pad_mask is None:
            x_max = torch.max(x, dim=-1).values
            x_mean = torch.mean(x, dim=-1)
            return torch.cat([x_max, x_mean], dim=-1)

        valid_mask = ~pad_mask
        x_for_max = x.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        x_max = torch.max(x_for_max, dim=-1).values
        x_max = torch.nan_to_num(x_max, neginf=0.0)

        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(x.dtype)
        x_sum = x.masked_fill(pad_mask.unsqueeze(1), 0.0).sum(dim=-1)
        x_mean = x_sum / valid_counts
        return torch.cat([x_max, x_mean], dim=-1)

    def forward(self, features, token_ids, pad_mask=None, return_stats=False, return_multitask=False):
        """
        features:  [B, C, T]
        token_ids: [B, T]
        pad_mask:  [B, T], True marks padding
        """
        if self.training and self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std

        expected_min_channels = self.num_attn_channels + self.prob_channels
        if features.size(1) < expected_min_channels:
            raise ValueError(
                f"features has {features.size(1)} channels, but model expects at least "
                f"{expected_min_channels} = {self.num_attn_channels} attention channels "
                f"+ {self.prob_channels} probability/stat channels."
            )

        attn_input = features[:, :self.num_attn_channels, :]
        prob_input = features[
            :,
            self.num_attn_channels:self.num_attn_channels + self.prob_channels,
            :
        ]

        attn_selected = self.head_selector(attn_input)

        attn_feat = self.attn_conv(attn_selected)
        prob_feat = self.prob_conv(prob_input)

        emb = self.embedding(token_ids).transpose(1, 2)
        emb_feat = self.emb_conv(emb)

        # The auxiliary 6-way RAG signal classifier uses the three feature
        # families directly. It deliberately bypasses the global head gate.
        class_attn_feat = self.attn_conv(attn_input)
        class_features = torch.cat(
            [
                self._masked_pool(class_attn_feat, pad_mask),
                self._masked_pool(emb_feat, pad_mask),
                self._masked_pool(prob_feat, pad_mask),
            ],
            dim=-1,
        )
        rag_class_logits = self.rag_class_head(class_features)
        aux_binary_logits = self.aux_binary_head(class_features)

        gate_a = torch.sigmoid(self.attn_gate_layer(attn_feat))
        gate_e = torch.sigmoid(self.emb_gate_layer(emb_feat))

        fused = (1 - prob_feat) * gate_a * gate_e
        x = self.post_fusion(fused)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        x_max, _ = torch.max(x, dim=-1)
        x_max = torch.nan_to_num(x_max, neginf=0.0)

        logits = self.fc(x_max)

        if return_multitask:
            output = {
                "binary_logits": logits,
                "rag_class_logits": rag_class_logits,
                "aux_binary_logits": aux_binary_logits,
            }
            if not return_stats:
                return output
        elif not return_stats:
            return logits

        head_weights = self.head_selector.get_head_weights()
        if head_weights is not None:
            head_weights = head_weights.detach().cpu().numpy()

        stats = {
            "head_weights": head_weights,
            "gate_a": gate_a.mean(dim=1).detach().cpu().numpy(),
            "gate_e": gate_e.mean(dim=1).detach().cpu().numpy(),
            "fused": fused.mean(dim=1).detach().cpu().numpy(),
            "rag_signal_classes": self.rag_signal_classes,
        }

        if return_multitask:
            return output, stats

        return logits, stats


class MaskedConvBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout=0.1, groups=8):
        super().__init__()
        norm_groups = max(1, min(int(groups), int(hidden_dim)))
        while hidden_dim % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.norm1 = nn.GroupNorm(norm_groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, valid_mask):
        mask = valid_mask.unsqueeze(1).to(dtype=x.dtype)
        x = x * mask
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x * mask
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x * mask


class SegmentFeatureEncoder(nn.Module):
    def __init__(self, embedding, embedding_dim, num_attn_channels, prob_channels, hidden_dim, dropout=0.1, use_attention_branch=True):
        super().__init__()
        self.embedding = embedding
        self.num_attn_channels = num_attn_channels
        self.prob_channels = prob_channels
        self.use_attention_branch = bool(use_attention_branch)
        self.attn_conv = MaskedConvBranch(num_attn_channels, hidden_dim, dropout=dropout) if self.use_attention_branch else None
        self.prob_conv = MaskedConvBranch(prob_channels, hidden_dim, dropout=dropout)
        self.emb_conv = MaskedConvBranch(embedding_dim, hidden_dim, dropout=dropout)
        pooled_dim = hidden_dim * 2 * (3 if self.use_attention_branch else 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def masked_pool(x, valid_mask):
        valid_mask = valid_mask.bool()
        x_for_max = x.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        x_max = torch.max(x_for_max, dim=-1).values
        x_max = torch.nan_to_num(x_max, neginf=0.0)
        counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
        x_mean = x.masked_fill(~valid_mask.unsqueeze(1), 0.0).sum(dim=-1) / counts
        return torch.cat([x_max, x_mean], dim=-1)

    def forward(self, features, token_ids, valid_mask):
        prob_input = features[:, self.num_attn_channels:self.num_attn_channels + self.prob_channels, :]
        emb_input = self.embedding(token_ids).transpose(1, 2)

        prob_feat = self.prob_conv(prob_input, valid_mask)
        emb_feat = self.emb_conv(emb_input, valid_mask)
        pooled_parts = []
        if self.use_attention_branch:
            attn_input = features[:, :self.num_attn_channels, :]
            attn_feat = self.attn_conv(attn_input, valid_mask)
            pooled_parts.append(self.masked_pool(attn_feat, valid_mask))
        pooled_parts.extend([
            self.masked_pool(prob_feat, valid_mask),
            self.masked_pool(emb_feat, valid_mask),
        ])
        pooled = torch.cat(pooled_parts, dim=-1)
        return self.proj(pooled)


class QuestionAnswerDualProbeCNN(nn.Module):
    """Question-only probe for v4 features: question segment + direct-answer segment."""
    def __init__(
        self,
        embed_weights,
        num_attn_channels=128,
        prob_channels=9,
        hidden_dim=128,
        noise_std=0.03,
        dropout=0.1,
        head_dropout=0.0,
        use_head_gate=False,
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
        use_attention_branch=True,
    ):
        super().__init__()
        self.num_attn_channels = num_attn_channels
        self.prob_channels = prob_channels
        self.noise_std = noise_std
        self.num_rag_classes = num_rag_classes
        self.rag_signal_classes = RAG_SIGNAL_CLASSES[:num_rag_classes]
        self.use_attention_branch = bool(use_attention_branch)
        self.question_mask_channel = num_attn_channels + prob_channels
        self.answer_mask_channel = num_attn_channels + prob_channels + 1

        num_embeddings, embedding_dim = embed_weights.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(embed_weights)
        self.embedding.weight.requires_grad = False

        self.question_encoder = SegmentFeatureEncoder(
            self.embedding, embedding_dim, num_attn_channels, prob_channels, hidden_dim, dropout=dropout, use_attention_branch=self.use_attention_branch
        )
        self.answer_encoder = SegmentFeatureEncoder(
            self.embedding, embedding_dim, num_attn_channels, prob_channels, hidden_dim, dropout=dropout, use_attention_branch=self.use_attention_branch
        )
        self.empty_answer = nn.Parameter(torch.zeros(hidden_dim * 2))

        fusion_dim = hidden_dim * 2 * 5 + 5
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.rag_class_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_rag_classes),
        )
        self.aux_binary_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2),
        )

    def _segment_masks(self, features, pad_mask):
        if features.size(1) <= self.answer_mask_channel:
            raise ValueError(
                f'QuestionAnswerDualProbeCNN expects mask channels at indices '
                f'{self.question_mask_channel}/{self.answer_mask_channel}, got C={features.size(1)}'
            )
        question_mask = features[:, self.question_mask_channel, :] > 0.5
        answer_mask = features[:, self.answer_mask_channel, :] > 0.5
        if pad_mask is not None:
            question_mask = question_mask & (~pad_mask)
            answer_mask = answer_mask & (~pad_mask)
        return question_mask, answer_mask

    def _length_stats(self, features, question_mask, answer_mask):
        q_len = question_mask.sum(dim=-1, keepdim=True).to(dtype=features.dtype)
        a_len = answer_mask.sum(dim=-1, keepdim=True).to(dtype=features.dtype)
        q_frac = q_len / max(float(features.size(-1)), 1.0)
        a_frac = a_len / 32.0
        answer_empty = (a_len <= 0).to(dtype=features.dtype)
        surprisal_idx = self.num_attn_channels + 2
        if features.size(1) > surprisal_idx:
            surprisal = features[:, surprisal_idx, :]
            denom = a_len.clamp_min(1.0)
            ans_mean_surprisal = surprisal.masked_fill(~answer_mask, 0.0).sum(dim=-1, keepdim=True) / denom
            ans_max_surprisal = surprisal.masked_fill(~answer_mask, float('-inf')).max(dim=-1, keepdim=True).values
            ans_max_surprisal = torch.nan_to_num(ans_max_surprisal, neginf=0.0)
        else:
            ans_mean_surprisal = torch.zeros_like(q_frac)
            ans_max_surprisal = torch.zeros_like(q_frac)
        return torch.cat([q_frac, a_frac, answer_empty, ans_mean_surprisal, ans_max_surprisal], dim=-1)

    def forward(self, features, token_ids, pad_mask=None, return_stats=False, return_multitask=False):
        expected_min_channels = self.num_attn_channels + self.prob_channels + 2
        if features.size(1) < expected_min_channels:
            raise ValueError(
                f'features has {features.size(1)} channels, but dual question model expects at least {expected_min_channels}'
            )
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(features[:, :self.num_attn_channels + self.prob_channels, :]) * self.noise_std
            features = torch.cat([features[:, :self.num_attn_channels + self.prob_channels, :] + noise,
                                  features[:, self.num_attn_channels + self.prob_channels:, :]], dim=1)

        question_mask, answer_mask = self._segment_masks(features, pad_mask)
        q_repr = self.question_encoder(features, token_ids, question_mask)
        a_repr = self.answer_encoder(features, token_ids, answer_mask)
        empty = (answer_mask.sum(dim=-1) == 0).unsqueeze(-1)
        a_repr = torch.where(empty, self.empty_answer.unsqueeze(0).expand_as(a_repr), a_repr)

        stats_vec = self._length_stats(features, question_mask, answer_mask)
        fusion_input = torch.cat([q_repr, a_repr, a_repr - q_repr, torch.abs(a_repr - q_repr), q_repr * a_repr, stats_vec], dim=-1)
        fused = self.fusion(fusion_input)
        logits = self.fc(fused)
        rag_class_logits = self.rag_class_head(fusion_input)
        aux_binary_logits = self.aux_binary_head(fusion_input)

        if return_multitask:
            output = {"binary_logits": logits, "rag_class_logits": rag_class_logits, "aux_binary_logits": aux_binary_logits}
            if not return_stats:
                return output
        elif not return_stats:
            return logits

        stats = {
            "head_weights": None,
            "question_len": question_mask.sum(dim=-1).detach().cpu().numpy(),
            "answer_len": answer_mask.sum(dim=-1).detach().cpu().numpy(),
            "use_attention_branch": self.use_attention_branch,
            "rag_signal_classes": self.rag_signal_classes,
        }
        if return_multitask:
            return output, stats
        return logits, stats


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1, kernel_size=3, groups=8):
        super().__init__()
        padding = kernel_size // 2
        norm_groups = max(1, min(int(groups), int(channels)))
        while channels % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x, mask):
        mask_f = mask.unsqueeze(1).to(dtype=x.dtype)
        y = self.net(x * mask_f)
        return self.act((x + y) * mask_f)


class TokenResNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_blocks=3, dropout=0.1, conv_kernel_size=3,
                 pool_first="off"):
        super().__init__()
        # pool_first: the transform/pool ORDER ablation. "off" (default) = ours,
        # transform tokens then pool. "maxmean"/"mean" = pool the raw composed
        # token features FIRST (max+mean concat or mean only), then run the SAME
        # stem+blocks on the pooled vector as a length-1 sequence (k=3 convs see
        # zero padding, i.e. degenerate to pointwise): identical capacity, only
        # the order of aggregation changes. Output dim is unchanged ([B, 2H]).
        self.pool_first = str(pool_first or "off")
        stem_in = in_channels * 2 if self.pool_first == "maxmean" else in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(stem_in, hidden_dim, kernel_size=1),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        # conv_kernel_size=1 turns every block conv pointwise (per-token MLP + pooling):
        # the temporal-receptive-field ablation. Everything else (dims, norm, act,
        # dropout, pooling, [B, hidden_dim, T] interface) is unchanged.
        self.blocks = nn.ModuleList([
            ResidualConvBlock1D(hidden_dim, dropout=dropout, kernel_size=int(conv_kernel_size))
            for _ in range(int(num_blocks))
        ])

    @staticmethod
    def masked_pool(x, mask):
        mask = mask.bool()
        x_max = x.masked_fill(~mask.unsqueeze(1), float("-inf")).max(dim=-1).values
        x_max = torch.nan_to_num(x_max, neginf=0.0)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
        x_mean = x.masked_fill(~mask.unsqueeze(1), 0.0).sum(dim=-1) / counts
        return torch.cat([x_max, x_mean], dim=-1)

    def encode_tokens(self, x, mask):
        mask_f = mask.unsqueeze(1).to(dtype=x.dtype)
        x = self.stem(x * mask_f) * mask_f
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, x, mask):
        if self.pool_first != "off":
            v = self.masked_pool(x, mask)                    # [B, 2C] (max ‖ mean)
            if self.pool_first == "mean":
                v = v[:, v.shape[1] // 2:]                   # mean half only -> [B, C]
            t = v.unsqueeze(-1)                              # length-1 "sequence"
            ones = torch.ones(t.shape[0], 1, dtype=torch.bool, device=t.device)
            return self.masked_pool(self.encode_tokens(t, ones), ones)
        return self.masked_pool(self.encode_tokens(x, mask), mask)


class QuestionAnswerResNetProbe(nn.Module):
    """Question-only correctness probe using sidecar question/answer features."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        dropout=0.1,
        noise_std=0.02,
        use_token_embedding=True,
        use_last_hidden=True,
        conv_kernel_size=3,
        encoder_pool_first="off",
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        self.use_token_embedding = bool(use_token_embedding)
        self.use_last_hidden = bool(use_last_hidden)
        self.conv_kernel_size = int(conv_kernel_size)
        self.encoder_pool_first = str(encoder_pool_first or "off")

        if self.use_token_embedding:
            num_embeddings, embedding_dim = embed_weights.shape
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.copy_(embed_weights)
            self.embedding.weight.requires_grad = False
            self.embed_mlp = nn.Sequential(
                nn.Linear(embedding_dim, self.embed_proj_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_proj_dim),
            )
        else:
            self.embedding = None
            self.embed_mlp = None
        self.hidden_mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.hidden_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_proj_dim),
        )

        in_channels = self.numeric_channels + self.attention_channels + self.embed_proj_dim + self.hidden_proj_dim
        self.question_encoder = TokenResNetEncoder(in_channels, hidden_dim=hidden_dim, num_blocks=num_res_blocks, dropout=dropout, conv_kernel_size=self.conv_kernel_size, pool_first=self.encoder_pool_first)
        self.answer_encoder = TokenResNetEncoder(in_channels, hidden_dim=hidden_dim, num_blocks=num_res_blocks, dropout=dropout, conv_kernel_size=self.conv_kernel_size, pool_first=self.encoder_pool_first)
        pooled_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(pooled_dim * 2),
            nn.Linear(pooled_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _compose(self, numeric, attn, token_ids, last_hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            last_hidden = last_hidden + torch.randn_like(last_hidden) * self.noise_std
        if self.use_token_embedding:
            emb = self.embed_mlp(self.embedding(token_ids)).transpose(1, 2)
        else:
            emb = numeric.new_zeros((numeric.size(0), self.embed_proj_dim, numeric.size(-1)))
        hidden = self.hidden_mlp(last_hidden).transpose(1, 2)
        if not self.use_last_hidden:
            hidden = torch.zeros_like(hidden)
        if attn.size(1) == 0:
            attn = numeric.new_zeros((numeric.size(0), self.attention_channels, numeric.size(-1)))
        return torch.cat([numeric, attn, emb, hidden], dim=1)

    def encode(self, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask):
        q = self._compose(q_numeric, q_attn, q_token_ids, q_hidden)
        a = self._compose(a_numeric, a_attn, a_token_ids, a_hidden)
        q_vec = self.question_encoder(q, q_mask)
        a_vec = self.answer_encoder(a, a_mask)
        x = torch.cat([q_vec, a_vec], dim=-1)
        # Reuse the pretrained MLP as part of the encoder, dropping only the final logit layer.
        return self.fusion[:-1](x)

    def forward(self, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask):
        return self.fusion[-1](self.encode(q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask))

class ContextQuestionAnswerResNetProbe(nn.Module):
    """With-context correctness/helpfulness probe using context/question/answer sidecar features."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        dropout=0.1,
        noise_std=0.02,
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
        use_token_embedding=True,
        use_last_hidden=True,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        self.use_token_embedding = bool(use_token_embedding)
        self.use_last_hidden = bool(use_last_hidden)
        self.conv_kernel_size = int(conv_kernel_size)
        self.num_rag_classes = int(num_rag_classes)
        self.rag_signal_classes = RAG_SIGNAL_CLASSES[:num_rag_classes]

        if self.use_token_embedding:
            num_embeddings, embedding_dim = embed_weights.shape
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.copy_(embed_weights)
            self.embedding.weight.requires_grad = False
            self.embed_mlp = nn.Sequential(
                nn.Linear(embedding_dim, self.embed_proj_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_proj_dim),
            )
        else:
            self.embedding = None
            self.embed_mlp = None
        self.hidden_mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.hidden_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_proj_dim),
        )

        in_channels = self.numeric_channels + self.attention_channels + self.embed_proj_dim + self.hidden_proj_dim
        self.context_encoder = TokenResNetEncoder(in_channels, hidden_dim=hidden_dim, num_blocks=num_res_blocks, dropout=dropout, conv_kernel_size=self.conv_kernel_size)
        self.question_encoder = TokenResNetEncoder(in_channels, hidden_dim=hidden_dim, num_blocks=num_res_blocks, dropout=dropout, conv_kernel_size=self.conv_kernel_size)
        self.answer_encoder = TokenResNetEncoder(in_channels, hidden_dim=hidden_dim, num_blocks=num_res_blocks, dropout=dropout, conv_kernel_size=self.conv_kernel_size)
        pooled_dim = hidden_dim * 2
        fusion_dim = pooled_dim * 6
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim * 2, 1)
        self.aux_binary_head = nn.Linear(hidden_dim * 2, 2)
        self.rag_class_head = nn.Linear(hidden_dim * 2, num_rag_classes)

    def _compose(self, numeric, attn, token_ids, last_hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            last_hidden = last_hidden + torch.randn_like(last_hidden) * self.noise_std
        if self.use_token_embedding:
            emb = self.embed_mlp(self.embedding(token_ids)).transpose(1, 2)
        else:
            emb = numeric.new_zeros((numeric.size(0), self.embed_proj_dim, numeric.size(-1)))
        hidden = self.hidden_mlp(last_hidden).transpose(1, 2)
        if not self.use_last_hidden:
            hidden = torch.zeros_like(hidden)
        if attn.size(1) == 0:
            attn = numeric.new_zeros((numeric.size(0), self.attention_channels, numeric.size(-1)))
        return torch.cat([numeric, attn, emb, hidden], dim=1)

    def forward(
        self,
        c_numeric,
        c_attn,
        c_token_ids,
        c_hidden,
        c_mask,
        q_numeric,
        q_attn,
        q_token_ids,
        q_hidden,
        q_mask,
        a_numeric,
        a_attn,
        a_token_ids,
        a_hidden,
        a_mask,
        return_multitask=False,
    ):
        fused = self.encode(
            c_numeric, c_attn, c_token_ids, c_hidden, c_mask,
            q_numeric, q_attn, q_token_ids, q_hidden, q_mask,
            a_numeric, a_attn, a_token_ids, a_hidden, a_mask,
        )
        binary_logits = self.binary_head(fused)
        if not return_multitask:
            return binary_logits
        return {
            "binary_logits": binary_logits,
            "aux_binary_logits": self.aux_binary_head(fused),
            "rag_class_logits": self.rag_class_head(fused),
        }

    def encode(
        self,
        c_numeric,
        c_attn,
        c_token_ids,
        c_hidden,
        c_mask,
        q_numeric,
        q_attn,
        q_token_ids,
        q_hidden,
        q_mask,
        a_numeric,
        a_attn,
        a_token_ids,
        a_hidden,
        a_mask,
    ):
        c = self._compose(c_numeric, c_attn, c_token_ids, c_hidden)
        q = self._compose(q_numeric, q_attn, q_token_ids, q_hidden)
        a = self._compose(a_numeric, a_attn, a_token_ids, a_hidden)
        c_vec = self.context_encoder(c, c_mask)
        q_vec = self.question_encoder(q, q_mask)
        a_vec = self.answer_encoder(a, a_mask)
        fusion_input = torch.cat([
            c_vec,
            q_vec,
            a_vec,
            q_vec - c_vec,
            a_vec - q_vec,
            a_vec - c_vec,
        ], dim=-1)
        return self.fusion(fusion_input)






class _OptionalTokenBranch(nn.Module):
    def __init__(self, in_channels, out_dim, encoder_hidden_dim, num_blocks=2, dropout=0.1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        if self.in_channels > 0:
            self.encoder = TokenResNetEncoder(
                self.in_channels,
                hidden_dim=int(encoder_hidden_dim),
                num_blocks=int(num_blocks),
                dropout=float(dropout),
            )
            self.project = nn.Sequential(
                nn.LayerNorm(int(encoder_hidden_dim) * 2),
                nn.Linear(int(encoder_hidden_dim) * 2, self.out_dim),
                nn.GELU(),
            )
        else:
            self.encoder = None
            self.project = None

    def forward(self, x, mask):
        if self.encoder is None or x is None or x.size(1) == 0:
            return None
        return self.project(self.encoder(x, mask))


class ModalityGatedTokenEncoder(nn.Module):
    """Token encoder with hidden-state backbone and gated auxiliary modalities."""

    def __init__(
        self,
        numeric_channels,
        attention_channels,
        embed_weights,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.prob_feature_channels = max(0, min(int(prob_feature_channels), self.numeric_channels))
        self.residual_channels = max(0, self.numeric_channels - self.prob_feature_channels)
        self.use_token_embedding = bool(use_token_embedding)
        self.use_probability_features = bool(use_probability_features)
        self.use_residual_features = bool(use_residual_features)
        self.use_attention_features = bool(use_attention_features)
        self.use_last_hidden = bool(use_last_hidden)
        self.modality_dropout = float(modality_dropout)
        self.output_dim = self.hidden_dim * 2

        num_embeddings, embedding_dim = embed_weights.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(embed_weights)
        self.embedding.weight.requires_grad = False
        self.embed_mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.embed_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_proj_dim),
        )
        self.hidden_mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.hidden_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_proj_dim),
        )
        self.hidden_branch = _OptionalTokenBranch(
            self.hidden_proj_dim,
            self.output_dim,
            encoder_hidden_dim=self.hidden_dim,
            num_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.prob_branch = _OptionalTokenBranch(
            self.prob_feature_channels if self.use_probability_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        self.residual_branch = _OptionalTokenBranch(
            self.residual_channels if self.use_residual_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        self.attention_branch = _OptionalTokenBranch(
            self.attention_channels if self.use_attention_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        self.embedding_branch = _OptionalTokenBranch(
            self.embed_proj_dim if self.use_token_embedding else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        aux_count = int(self.prob_branch.in_channels > 0) + int(self.residual_branch.in_channels > 0) + int(self.attention_branch.in_channels > 0) + int(self.embedding_branch.in_channels > 0)
        self.aux_count = aux_count
        self.gate = nn.Sequential(
            nn.LayerNorm(self.output_dim * max(aux_count + 1, 1)),
            nn.Linear(self.output_dim * max(aux_count + 1, 1), max(self.output_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.output_dim // 2, 32), max(aux_count, 1)),
        )
        self.out_norm = nn.LayerNorm(self.output_dim)

    def _split_numeric(self, numeric):
        prob = numeric[:, :self.prob_feature_channels, :] if self.prob_feature_channels > 0 else numeric[:, :0, :]
        residual = numeric[:, self.prob_feature_channels:, :] if self.residual_channels > 0 else numeric[:, :0, :]
        return prob, residual

    def forward(self, numeric, attn, token_ids, last_hidden, mask):
        hidden = self.hidden_mlp(last_hidden).transpose(1, 2)
        if not self.use_last_hidden:
            hidden = torch.zeros_like(hidden)
        hidden_vec = self.hidden_branch(hidden, mask)
        if hidden_vec is None:
            hidden_vec = numeric.new_zeros((numeric.size(0), self.output_dim))

        prob, residual = self._split_numeric(numeric)
        aux_vecs = []
        prob_vec = self.prob_branch(prob, mask)
        if prob_vec is not None:
            aux_vecs.append(prob_vec)
        residual_vec = self.residual_branch(residual, mask)
        if residual_vec is not None:
            aux_vecs.append(residual_vec)
        if attn.size(1) == 0:
            attn = numeric.new_zeros((numeric.size(0), self.attention_channels, numeric.size(-1)))
        attn_vec = self.attention_branch(attn, mask)
        if attn_vec is not None:
            aux_vecs.append(attn_vec)
        emb = self.embedding(token_ids)
        emb = self.embed_mlp(emb).transpose(1, 2)
        emb_vec = self.embedding_branch(emb, mask)
        if emb_vec is not None:
            aux_vecs.append(emb_vec)

        if not aux_vecs:
            return self.out_norm(hidden_vec)
        gate_input = torch.cat([hidden_vec] + aux_vecs, dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))[:, :len(aux_vecs)]
        if self.training and self.modality_dropout > 0:
            keep = (torch.rand_like(gates) >= self.modality_dropout).to(gates.dtype)
            gates = gates * keep
        aux_sum = torch.stack([gates[:, i:i + 1] * aux_vecs[i] for i in range(len(aux_vecs))], dim=0).sum(dim=0)
        return self.out_norm(hidden_vec + aux_sum / max(len(aux_vecs), 1) ** 0.5)


class QuestionAnswerModalityGatedProbe(nn.Module):
    """Question-only probe with hidden backbone and gated auxiliary feature branches."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        noise_std=0.02,
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        common = dict(
            numeric_channels=numeric_channels,
            attention_channels=attention_channels,
            embed_weights=embed_weights,
            hidden_dim=hidden_dim,
            aux_hidden_dim=self.aux_hidden_dim,
            embed_proj_dim=embed_proj_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            hidden_proj_dim=self.hidden_proj_dim,
            num_res_blocks=num_res_blocks,
            aux_num_res_blocks=aux_num_res_blocks,
            dropout=dropout,
            prob_feature_channels=prob_feature_channels,
            use_token_embedding=use_token_embedding,
            use_probability_features=use_probability_features,
            use_residual_features=use_residual_features,
            use_attention_features=use_attention_features,
            use_last_hidden=use_last_hidden,
            modality_dropout=modality_dropout,
        )
        self.question_encoder = ModalityGatedTokenEncoder(**common)
        self.answer_encoder = ModalityGatedTokenEncoder(**common)
        enc_dim = self.question_encoder.output_dim
        fusion_dim = enc_dim * 4
        self.fusion_body = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim, 1)

    def _maybe_noise(self, numeric, attn, hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            hidden = hidden + torch.randn_like(hidden) * self.noise_std
        return numeric, attn, hidden

    def encode(self, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask):
        q_numeric, q_attn, q_hidden = self._maybe_noise(q_numeric, q_attn, q_hidden)
        a_numeric, a_attn, a_hidden = self._maybe_noise(a_numeric, a_attn, a_hidden)
        q_vec = self.question_encoder(q_numeric, q_attn, q_token_ids, q_hidden, q_mask)
        a_vec = self.answer_encoder(a_numeric, a_attn, a_token_ids, a_hidden, a_mask)
        return self.fusion_body(torch.cat([q_vec, a_vec, a_vec - q_vec, torch.abs(a_vec - q_vec)], dim=-1))

    def forward(self, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask):
        return self.binary_head(self.encode(q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask))


class ContextQuestionAnswerModalityGatedProbe(nn.Module):
    """With-context probe with separate context/question/answer gated modality encoders."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        noise_std=0.02,
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        self.num_rag_classes = int(num_rag_classes)
        self.rag_signal_classes = RAG_SIGNAL_CLASSES[:num_rag_classes]
        common = dict(
            numeric_channels=numeric_channels,
            attention_channels=attention_channels,
            embed_weights=embed_weights,
            hidden_dim=hidden_dim,
            aux_hidden_dim=self.aux_hidden_dim,
            embed_proj_dim=embed_proj_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            hidden_proj_dim=self.hidden_proj_dim,
            num_res_blocks=num_res_blocks,
            aux_num_res_blocks=aux_num_res_blocks,
            dropout=dropout,
            prob_feature_channels=prob_feature_channels,
            use_token_embedding=use_token_embedding,
            use_probability_features=use_probability_features,
            use_residual_features=use_residual_features,
            use_attention_features=use_attention_features,
            use_last_hidden=use_last_hidden,
            modality_dropout=modality_dropout,
        )
        self.context_encoder = ModalityGatedTokenEncoder(**common)
        self.question_encoder = ModalityGatedTokenEncoder(**common)
        self.answer_encoder = ModalityGatedTokenEncoder(**common)
        enc_dim = self.context_encoder.output_dim
        fusion_dim = enc_dim * 6
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim * 2, 1)
        self.aux_binary_head = nn.Linear(hidden_dim * 2, 2)
        self.rag_class_head = nn.Linear(hidden_dim * 2, num_rag_classes)

    def _maybe_noise(self, numeric, attn, hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            hidden = hidden + torch.randn_like(hidden) * self.noise_std
        return numeric, attn, hidden

    def encode(self, c_numeric, c_attn, c_token_ids, c_hidden, c_mask, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask):
        c_numeric, c_attn, c_hidden = self._maybe_noise(c_numeric, c_attn, c_hidden)
        q_numeric, q_attn, q_hidden = self._maybe_noise(q_numeric, q_attn, q_hidden)
        a_numeric, a_attn, a_hidden = self._maybe_noise(a_numeric, a_attn, a_hidden)
        c_vec = self.context_encoder(c_numeric, c_attn, c_token_ids, c_hidden, c_mask)
        q_vec = self.question_encoder(q_numeric, q_attn, q_token_ids, q_hidden, q_mask)
        a_vec = self.answer_encoder(a_numeric, a_attn, a_token_ids, a_hidden, a_mask)
        return self.fusion(torch.cat([c_vec, q_vec, a_vec, q_vec - c_vec, a_vec - q_vec, a_vec - c_vec], dim=-1))

    def forward(self, c_numeric, c_attn, c_token_ids, c_hidden, c_mask, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask, return_multitask=False):
        fused = self.encode(c_numeric, c_attn, c_token_ids, c_hidden, c_mask, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask)
        binary_logits = self.binary_head(fused)
        if not return_multitask:
            return binary_logits
        return {
            "binary_logits": binary_logits,
            "aux_binary_logits": self.aux_binary_head(fused),
            "rag_class_logits": self.rag_class_head(fused),
        }




class ModalityEnsembleSegmentEncoder(nn.Module):
    """Segment encoder that keeps each modality as an independent evidence branch."""

    def __init__(
        self,
        numeric_channels,
        attention_channels,
        embed_weights,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.0,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.prob_feature_channels = max(0, min(int(prob_feature_channels), self.numeric_channels))
        self.residual_channels = max(0, self.numeric_channels - self.prob_feature_channels)
        self.use_token_embedding = bool(use_token_embedding)
        self.use_probability_features = bool(use_probability_features)
        self.use_residual_features = bool(use_residual_features)
        self.use_attention_features = bool(use_attention_features)
        self.use_last_hidden = bool(use_last_hidden)
        self.output_dim = self.hidden_dim * 2

        self.hidden_mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.hidden_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_proj_dim),
        )
        self.hidden_branch = _OptionalTokenBranch(
            self.hidden_proj_dim if self.use_last_hidden else 0,
            self.output_dim,
            encoder_hidden_dim=self.hidden_dim,
            num_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.prob_branch = _OptionalTokenBranch(
            self.prob_feature_channels if self.use_probability_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        self.residual_branch = _OptionalTokenBranch(
            self.residual_channels if self.use_residual_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        self.attention_branch = _OptionalTokenBranch(
            self.attention_channels if self.use_attention_features else 0,
            self.output_dim,
            encoder_hidden_dim=self.aux_hidden_dim,
            num_blocks=aux_num_res_blocks,
            dropout=dropout,
        )
        if self.use_token_embedding:
            num_embeddings, embedding_dim = embed_weights.shape
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.copy_(embed_weights)
            self.embedding.weight.requires_grad = False
            self.embed_mlp = nn.Sequential(
                nn.Linear(embedding_dim, self.embed_proj_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_proj_dim),
            )
            self.embedding_branch = _OptionalTokenBranch(
                self.embed_proj_dim,
                self.output_dim,
                encoder_hidden_dim=self.aux_hidden_dim,
                num_blocks=aux_num_res_blocks,
                dropout=dropout,
            )
        else:
            self.embedding = None
            self.embed_mlp = None
            self.embedding_branch = _OptionalTokenBranch(0, self.output_dim, self.aux_hidden_dim)
        self.modality_names = ["hidden"]
        if self.use_probability_features and self.prob_feature_channels > 0:
            self.modality_names.append("probability")
        if self.use_residual_features and self.residual_channels > 0:
            self.modality_names.append("residual")
        if self.use_attention_features and self.attention_channels > 0:
            self.modality_names.append("attention")
        if self.use_token_embedding:
            self.modality_names.append("embedding")

    def _split_numeric(self, numeric):
        prob = numeric[:, :self.prob_feature_channels, :] if self.prob_feature_channels > 0 else numeric[:, :0, :]
        residual = numeric[:, self.prob_feature_channels:, :] if self.residual_channels > 0 else numeric[:, :0, :]
        return prob, residual

    def forward(self, numeric, attn, token_ids, last_hidden, mask):
        outputs = {}
        if self.use_last_hidden:
            hidden = self.hidden_mlp(last_hidden).transpose(1, 2)
            outputs["hidden"] = self.hidden_branch(hidden, mask)
        prob, residual = self._split_numeric(numeric)
        prob_vec = self.prob_branch(prob, mask)
        if prob_vec is not None:
            outputs["probability"] = prob_vec
        residual_vec = self.residual_branch(residual, mask)
        if residual_vec is not None:
            outputs["residual"] = residual_vec
        if attn.size(1) == 0:
            attn = numeric.new_zeros((numeric.size(0), self.attention_channels, numeric.size(-1)))
        attn_vec = self.attention_branch(attn, mask)
        if attn_vec is not None:
            outputs["attention"] = attn_vec
        if self.use_token_embedding:
            emb = self.embed_mlp(self.embedding(token_ids)).transpose(1, 2)
            emb_vec = self.embedding_branch(emb, mask)
            if emb_vec is not None:
                outputs["embedding"] = emb_vec
        return outputs


class _ModalityEvidenceFusion(nn.Module):
    def __init__(self, modality_names, pair_dim, hidden_dim=128, dropout=0.1, modality_dropout=0.0):
        super().__init__()
        self.modality_names = list(modality_names)
        self.hidden_dim = int(hidden_dim)
        self.modality_dropout = float(modality_dropout)
        self.branch_bodies = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for name in self.modality_names
        })
        self.branch_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in self.modality_names})
        num_modalities = len(self.modality_names)
        num_pairwise = num_modalities * (num_modalities - 1) // 2
        final_dim = hidden_dim * num_modalities + num_modalities + num_pairwise
        self.final = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _drop_modalities(self, branch_vecs, logits):
        if not self.training or self.modality_dropout <= 0 or len(branch_vecs) <= 1:
            return branch_vecs, logits

        batch_size = logits.size(0)
        num_modalities = logits.size(1)
        keep = torch.rand(batch_size, num_modalities, device=logits.device) >= self.modality_dropout
        dropped_all = ~keep.any(dim=1)
        if dropped_all.any():
            restore = torch.randint(num_modalities, (int(dropped_all.sum().item()),), device=logits.device)
            keep[dropped_all] = False
            keep[dropped_all, restore] = True

        scale = 1.0 / max(1.0 - self.modality_dropout, 1e-6)
        keep_f = keep.to(logits.dtype) * scale
        dropped_vecs = [vec * keep_f[:, i:i + 1] for i, vec in enumerate(branch_vecs)]
        dropped_logits = logits * keep_f
        return dropped_vecs, dropped_logits

    def forward(self, pair_features):
        branch_vecs = []
        branch_logits = []
        for name in self.modality_names:
            vec = self.branch_bodies[name](pair_features[name])
            logit = self.branch_heads[name](vec)
            branch_vecs.append(vec)
            branch_logits.append(logit)
        logits = torch.cat(branch_logits, dim=-1)
        final_vecs, final_logits = self._drop_modalities(branch_vecs, logits)
        disagreements = []
        for i in range(final_logits.size(1)):
            for j in range(i + 1, final_logits.size(1)):
                disagreements.append(torch.abs(final_logits[:, i:i + 1] - final_logits[:, j:j + 1]))
        if disagreements:
            disagreement = torch.cat(disagreements, dim=-1)
            final_input = torch.cat(final_vecs + [final_logits, disagreement], dim=-1)
        else:
            final_input = torch.cat(final_vecs + [final_logits], dim=-1)
        return self.final(final_input), logits


class QuestionAnswerModalityEnsembleProbe(nn.Module):
    """Question-only evidence ensemble with directly supervised modality branches."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        noise_std=0.02,
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.0,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        common = dict(
            numeric_channels=numeric_channels,
            attention_channels=attention_channels,
            embed_weights=embed_weights,
            hidden_dim=hidden_dim,
            aux_hidden_dim=self.aux_hidden_dim,
            embed_proj_dim=embed_proj_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            hidden_proj_dim=self.hidden_proj_dim,
            num_res_blocks=num_res_blocks,
            aux_num_res_blocks=aux_num_res_blocks,
            dropout=dropout,
            prob_feature_channels=prob_feature_channels,
            use_token_embedding=use_token_embedding,
            use_probability_features=use_probability_features,
            use_residual_features=use_residual_features,
            use_attention_features=use_attention_features,
            use_last_hidden=use_last_hidden,
        )
        self.question_encoder = ModalityEnsembleSegmentEncoder(**common)
        self.answer_encoder = ModalityEnsembleSegmentEncoder(**common)
        self.modality_names = self.question_encoder.modality_names
        pair_dim = self.question_encoder.output_dim * 4
        self.fusion = _ModalityEvidenceFusion(
            self.modality_names,
            pair_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

    def _maybe_noise(self, numeric, attn, hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            hidden = hidden + torch.randn_like(hidden) * self.noise_std
        return numeric, attn, hidden

    def _pair_features(self, q_dict, a_dict):
        out = {}
        for name in self.modality_names:
            q = q_dict[name]
            a = a_dict[name]
            out[name] = torch.cat([q, a, a - q, torch.abs(a - q)], dim=-1)
        return out

    def forward(self, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask, return_multitask=False):
        q_numeric, q_attn, q_hidden = self._maybe_noise(q_numeric, q_attn, q_hidden)
        a_numeric, a_attn, a_hidden = self._maybe_noise(a_numeric, a_attn, a_hidden)
        q_dict = self.question_encoder(q_numeric, q_attn, q_token_ids, q_hidden, q_mask)
        a_dict = self.answer_encoder(a_numeric, a_attn, a_token_ids, a_hidden, a_mask)
        logits, modality_logits = self.fusion(self._pair_features(q_dict, a_dict))
        if return_multitask:
            return {"binary_logits": logits, "modality_logits": modality_logits, "modality_names": self.modality_names}
        return logits


class ContextQuestionAnswerModalityEnsembleProbe(nn.Module):
    """With-context evidence ensemble with directly supervised modality branches."""

    def __init__(
        self,
        embed_weights,
        numeric_channels,
        attention_channels,
        hidden_dim=128,
        aux_hidden_dim=None,
        embed_proj_dim=32,
        llm_hidden_dim=None,
        hidden_proj_dim=None,
        num_res_blocks=3,
        aux_num_res_blocks=2,
        dropout=0.1,
        noise_std=0.02,
        num_rag_classes=len(RAG_SIGNAL_CLASSES),
        prob_feature_channels=5,
        use_token_embedding=False,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        modality_dropout=0.0,
    ):
        super().__init__()
        self.numeric_channels = int(numeric_channels)
        self.attention_channels = int(attention_channels)
        self.hidden_dim = int(hidden_dim)
        self.aux_hidden_dim = int(aux_hidden_dim) if aux_hidden_dim is not None else max(32, self.hidden_dim // 2)
        self.embed_proj_dim = int(embed_proj_dim)
        self.llm_hidden_dim = int(llm_hidden_dim) if llm_hidden_dim is not None else int(embed_weights.shape[1])
        self.hidden_proj_dim = int(hidden_proj_dim) if hidden_proj_dim is not None else self.embed_proj_dim
        self.noise_std = float(noise_std)
        self.num_rag_classes = int(num_rag_classes)
        self.rag_signal_classes = RAG_SIGNAL_CLASSES[:num_rag_classes]
        common = dict(
            numeric_channels=numeric_channels,
            attention_channels=attention_channels,
            embed_weights=embed_weights,
            hidden_dim=hidden_dim,
            aux_hidden_dim=self.aux_hidden_dim,
            embed_proj_dim=embed_proj_dim,
            llm_hidden_dim=self.llm_hidden_dim,
            hidden_proj_dim=self.hidden_proj_dim,
            num_res_blocks=num_res_blocks,
            aux_num_res_blocks=aux_num_res_blocks,
            dropout=dropout,
            prob_feature_channels=prob_feature_channels,
            use_token_embedding=use_token_embedding,
            use_probability_features=use_probability_features,
            use_residual_features=use_residual_features,
            use_attention_features=use_attention_features,
            use_last_hidden=use_last_hidden,
        )
        self.context_encoder = ModalityEnsembleSegmentEncoder(**common)
        self.question_encoder = ModalityEnsembleSegmentEncoder(**common)
        self.answer_encoder = ModalityEnsembleSegmentEncoder(**common)
        self.modality_names = self.context_encoder.modality_names
        pair_dim = self.context_encoder.output_dim * 6
        self.fusion = _ModalityEvidenceFusion(
            self.modality_names,
            pair_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )
        self.aux_binary_head = nn.Linear(hidden_dim * len(self.modality_names) + len(self.modality_names), 2)
        self.rag_class_head = nn.Linear(hidden_dim * len(self.modality_names) + len(self.modality_names), num_rag_classes)

    def _maybe_noise(self, numeric, attn, hidden):
        if self.training and self.noise_std > 0:
            numeric = numeric + torch.randn_like(numeric) * self.noise_std
            if attn.numel() > 0:
                attn = attn + torch.randn_like(attn) * self.noise_std
            hidden = hidden + torch.randn_like(hidden) * self.noise_std
        return numeric, attn, hidden

    def _pair_features(self, c_dict, q_dict, a_dict):
        out = {}
        for name in self.modality_names:
            c = c_dict[name]
            q = q_dict[name]
            a = a_dict[name]
            out[name] = torch.cat([c, q, a, q - c, a - q, a - c], dim=-1)
        return out

    def forward(self, c_numeric, c_attn, c_token_ids, c_hidden, c_mask, q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask, return_multitask=False):
        c_numeric, c_attn, c_hidden = self._maybe_noise(c_numeric, c_attn, c_hidden)
        q_numeric, q_attn, q_hidden = self._maybe_noise(q_numeric, q_attn, q_hidden)
        a_numeric, a_attn, a_hidden = self._maybe_noise(a_numeric, a_attn, a_hidden)
        c_dict = self.context_encoder(c_numeric, c_attn, c_token_ids, c_hidden, c_mask)
        q_dict = self.question_encoder(q_numeric, q_attn, q_token_ids, q_hidden, q_mask)
        a_dict = self.answer_encoder(a_numeric, a_attn, a_token_ids, a_hidden, a_mask)
        pair_features = self._pair_features(c_dict, q_dict, a_dict)
        logits, modality_logits = self.fusion(pair_features)
        if not return_multitask:
            return logits
        branch_vecs = []
        for name in self.modality_names:
            branch_vecs.append(self.fusion.branch_bodies[name](pair_features[name]))
        aux_base = torch.cat(branch_vecs + [modality_logits], dim=-1)
        return {
            "binary_logits": logits,
            "modality_logits": modality_logits,
            "modality_names": self.modality_names,
            "aux_binary_logits": self.aux_binary_head(aux_base),
            "rag_class_logits": self.rag_class_head(aux_base),
        }


class FrozenQuestionAnswerEncoderFusionProbe(nn.Module):
    """Fusion head over pretrained frozen question-only single-modality encoders."""

    def __init__(self, encoders, modality_names, encoder_dim=128, hidden_dim=128, dropout=0.1, modality_dropout=0.25):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modality_names = list(modality_names)
        self.encoder_dim = int(encoder_dim)
        self.hidden_dim = int(hidden_dim)
        self.modality_dropout = float(modality_dropout)
        for encoder in self.encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        num_modalities = len(self.modality_names)
        num_pairwise = num_modalities * (num_modalities - 1) // 2
        final_dim = self.encoder_dim * num_modalities + num_modalities + num_pairwise
        self.final = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def _drop_modalities(self, vecs, logits):
        if not self.training or self.modality_dropout <= 0 or len(vecs) <= 1:
            return vecs, logits
        batch_size = logits.size(0)
        num_modalities = logits.size(1)
        keep = torch.rand(batch_size, num_modalities, device=logits.device) >= self.modality_dropout
        dropped_all = ~keep.any(dim=1)
        if dropped_all.any():
            restore = torch.randint(num_modalities, (int(dropped_all.sum().item()),), device=logits.device)
            keep[dropped_all] = False
            keep[dropped_all, restore] = True
        scale = 1.0 / max(1.0 - self.modality_dropout, 1e-6)
        keep_f = keep.to(logits.dtype) * scale
        return [vec * keep_f[:, i:i + 1] for i, vec in enumerate(vecs)], logits * keep_f

    def _combine(self, vecs, branch_logits, return_multitask=False):
        """Trainable head over per-branch pooled vecs + logits; shared by the raw forward
        and forward_from_vecs (precomputed cache)."""
        modality_logits = torch.cat(branch_logits, dim=-1)
        final_vecs, final_logits = self._drop_modalities(vecs, modality_logits)
        disagreements = []
        for i in range(final_logits.size(1)):
            for j in range(i + 1, final_logits.size(1)):
                disagreements.append(torch.abs(final_logits[:, i:i + 1] - final_logits[:, j:j + 1]))
        final_input = torch.cat(final_vecs + [final_logits] + disagreements, dim=-1) if disagreements else torch.cat(final_vecs + [final_logits], dim=-1)
        logits = self.final(final_input)
        if return_multitask:
            return {"binary_logits": logits, "modality_logits": modality_logits, "modality_names": self.modality_names}
        return logits

    def forward(self, branch_batches, return_multitask=False):
        vecs = []
        branch_logits = []
        with torch.no_grad():
            for name in self.modality_names:
                batch = branch_batches[name]
                encoder = self.encoders[name]
                vec = encoder.encode(*batch)
                logit = encoder.fusion[-1](vec)
                vecs.append(vec.detach())
                branch_logits.append(logit.detach())
        return self._combine(vecs, branch_logits, return_multitask)

    def forward_from_vecs(self, vecs_by_name, logits_by_name, return_multitask=False):
        """Same head, but branch pooled vecs + logits come precomputed."""
        vecs = [vecs_by_name[name] for name in self.modality_names]
        branch_logits = [logits_by_name[name] for name in self.modality_names]
        return self._combine(vecs, branch_logits, return_multitask)


class FrozenContextQuestionAnswerEncoderFusionProbe(nn.Module):
    """Late fusion over frozen with-context single-modality encoders."""

    def __init__(self, encoders, modality_names, encoder_dim=256, hidden_dim=128, dropout=0.1, modality_dropout=0.25):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modality_names = list(modality_names)
        self.encoder_dim = int(encoder_dim)
        self.hidden_dim = int(hidden_dim)
        self.modality_dropout = float(modality_dropout)
        for encoder in self.encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        num_modalities = len(self.modality_names)
        num_pairwise = num_modalities * (num_modalities - 1) // 2
        final_dim = self.encoder_dim * num_modalities + num_modalities + num_pairwise
        self.final = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def _drop_modalities(self, vecs, logits):
        if not self.training or self.modality_dropout <= 0 or len(vecs) <= 1:
            return vecs, logits
        batch_size, num_modalities = logits.shape
        keep = torch.rand(batch_size, num_modalities, device=logits.device) >= self.modality_dropout
        dropped_all = ~keep.any(dim=1)
        if dropped_all.any():
            restore = torch.randint(num_modalities, (int(dropped_all.sum().item()),), device=logits.device)
            keep[dropped_all] = False
            keep[dropped_all, restore] = True
        keep_f = keep.to(logits.dtype) / max(1.0 - self.modality_dropout, 1e-6)
        return [vec * keep_f[:, i:i + 1] for i, vec in enumerate(vecs)], logits * keep_f

    def _combine(self, vecs, branch_logits, return_multitask=False):
        """Trainable head over per-branch pooled vecs + branch logits. Shared by the raw
        forward (runs frozen encoders) and forward_from_vecs (precomputed cache)."""
        modality_logits = torch.cat(branch_logits, dim=-1)
        final_vecs, final_logits = self._drop_modalities(vecs, modality_logits)
        disagreements = []
        for i in range(final_logits.size(1)):
            for j in range(i + 1, final_logits.size(1)):
                disagreements.append(torch.abs(final_logits[:, i:i + 1] - final_logits[:, j:j + 1]))
        final_input = torch.cat(final_vecs + [final_logits] + disagreements, dim=-1)
        logits = self.final(final_input)
        if return_multitask:
            return {"binary_logits": logits, "modality_logits": modality_logits, "modality_names": self.modality_names}
        return logits

    def forward(self, branch_batches, return_multitask=False):
        vecs, branch_logits = [], []
        with torch.no_grad():
            for name in self.modality_names:
                encoder = self.encoders[name]
                vec = encoder.encode(*branch_batches[name])
                vecs.append(vec.detach())
                branch_logits.append(encoder.binary_head(vec).detach())
        return self._combine(vecs, branch_logits, return_multitask)

    def forward_from_vecs(self, vecs_by_name, logits_by_name, return_multitask=False):
        """Same head, but branch pooled vecs + logits come precomputed (frozen encoders
        skipped entirely). Used by the cached fast-fusion trainer."""
        vecs = [vecs_by_name[name] for name in self.modality_names]
        branch_logits = [logits_by_name[name] for name in self.modality_names]
        return self._combine(vecs, branch_logits, return_multitask)


class FrozenQuestionAnswerTokenFusionProbe(nn.Module):
    """Token-level fusion over frozen pretrained single-modality encoders."""

    def __init__(self, encoders, modality_names, hidden_dim=128, num_res_blocks=3, dropout=0.1, modality_dropout=0.25):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modality_names = list(modality_names)
        self.hidden_dim = int(hidden_dim)
        self.modality_dropout = float(modality_dropout)
        for encoder in self.encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        self.question_reducers = nn.ModuleDict()
        self.answer_reducers = nn.ModuleDict()
        for name in self.modality_names:
            encoder = self.encoders[name]
            self.question_reducers[name] = nn.Conv1d(int(encoder.question_encoder.stem[0].out_channels), 1, 1)
            self.answer_reducers[name] = nn.Conv1d(int(encoder.answer_encoder.stem[0].out_channels), 1, 1)

        num_modalities = len(self.modality_names)
        self.question_fusion_encoder = TokenResNetEncoder(num_modalities, self.hidden_dim, num_res_blocks, dropout)
        self.answer_fusion_encoder = TokenResNetEncoder(num_modalities, self.hidden_dim, num_res_blocks, dropout)
        pooled_dim = self.hidden_dim * 2
        self.final = nn.Sequential(
            nn.LayerNorm(pooled_dim * 2),
            nn.Linear(pooled_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def _modality_keep_mask(self, batch_size, device, dtype):
        num_modalities = len(self.modality_names)
        if not self.training or self.modality_dropout <= 0 or num_modalities <= 1:
            return torch.ones(batch_size, num_modalities, 1, device=device, dtype=dtype)
        keep = torch.rand(batch_size, num_modalities, device=device) >= self.modality_dropout
        dropped_all = ~keep.any(dim=1)
        if dropped_all.any():
            restore = torch.randint(num_modalities, (int(dropped_all.sum().item()),), device=device)
            keep[dropped_all] = False
            keep[dropped_all, restore] = True
        return keep.to(dtype=dtype).unsqueeze(-1) / max(1.0 - self.modality_dropout, 1e-6)

    @staticmethod
    def _token_features(encoder, batch):
        q_numeric, q_attn, q_token_ids, q_hidden, q_mask, a_numeric, a_attn, a_token_ids, a_hidden, a_mask = batch
        q = encoder._compose(q_numeric, q_attn, q_token_ids, q_hidden)
        a = encoder._compose(a_numeric, a_attn, a_token_ids, a_hidden)
        q_tokens = encoder.question_encoder.encode_tokens(q, q_mask)
        a_tokens = encoder.answer_encoder.encode_tokens(a, a_mask)
        q_vec = encoder.question_encoder.masked_pool(q_tokens, q_mask)
        a_vec = encoder.answer_encoder.masked_pool(a_tokens, a_mask)
        branch_vec = encoder.fusion[:-1](torch.cat([q_vec, a_vec], dim=-1))
        branch_logit = encoder.fusion[-1](branch_vec)
        return q_tokens, a_tokens, q_mask, a_mask, branch_logit

    def forward(self, branch_batches, return_multitask=False):
        q_channels, a_channels, branch_logits = [], [], []
        q_mask = a_mask = None
        with torch.no_grad():
            frozen_outputs = {name: self._token_features(self.encoders[name], branch_batches[name]) for name in self.modality_names}
        for name in self.modality_names:
            q_tokens, a_tokens, branch_q_mask, branch_a_mask, branch_logit = frozen_outputs[name]
            q_mask = branch_q_mask if q_mask is None else q_mask
            a_mask = branch_a_mask if a_mask is None else a_mask
            q_channels.append(self.question_reducers[name](q_tokens.detach()))
            a_channels.append(self.answer_reducers[name](a_tokens.detach()))
            branch_logits.append(branch_logit.detach())

        q_fused = torch.cat(q_channels, dim=1)
        a_fused = torch.cat(a_channels, dim=1)
        keep = self._modality_keep_mask(q_fused.size(0), q_fused.device, q_fused.dtype)
        q_vec = self.question_fusion_encoder(q_fused * keep, q_mask)
        a_vec = self.answer_fusion_encoder(a_fused * keep, a_mask)
        logits = self.final(torch.cat([q_vec, a_vec], dim=-1))
        if return_multitask:
            return {"binary_logits": logits, "modality_logits": torch.cat(branch_logits, dim=-1), "modality_names": self.modality_names}
        return logits


class PairedQuestionContextResNetProbe(nn.Module):
    """Dual pretrained-encoder probe over question-only and with-context features."""

    def __init__(
        self,
        question_encoder: QuestionAnswerResNetProbe,
        context_encoder: ContextQuestionAnswerResNetProbe,
        question_encoder_dim: int,
        context_encoder_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.question_encoder_dim = int(question_encoder_dim)
        self.context_encoder_dim = int(context_encoder_dim)
        self.freeze_encoders = bool(freeze_encoders)
        fused_dim = self.question_encoder_dim + self.context_encoder_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.with_context_head = nn.Linear(hidden_dim, 1)
        self.without_context_head = nn.Linear(hidden_dim, 1)
        if self.freeze_encoders:
            for module in (self.question_encoder, self.context_encoder):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.question_encoder.eval()
            self.context_encoder.eval()
        return self

    def forward(self, qa_batch, cqa_batch):
        if self.freeze_encoders:
            with torch.no_grad():
                qa_vec = self.question_encoder.encode(*qa_batch)
                cqa_vec = self.context_encoder.encode(*cqa_batch)
        else:
            qa_vec = self.question_encoder.encode(*qa_batch)
            cqa_vec = self.context_encoder.encode(*cqa_batch)
        fused = self.fusion(torch.cat([qa_vec, cqa_vec], dim=-1))
        return {
            "with_context_logits": self.with_context_head(fused),
            "without_context_logits": self.without_context_head(fused),
        }
