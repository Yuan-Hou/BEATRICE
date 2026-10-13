# Checkpoints

All checkpoints live under `beatrice/checkpoints/` and are loaded by the
service configs (`configs/experiment/services/serve_probe_scorer_{qwen,gemma}.yaml`)
and by the evaluation scripts. Fusion heads store their branch composition in
`branch_checkpoints` with package-relative paths; keep the working directory
at `beatrice/` when loading.

Signal families: H = layer hidden states, P = probability scalars,
R = residual norm + lens channels, A = attention entropy + mass.

## Qwen3.5-9B (hidden layer 17)

| File | Setting | Branches (greedy-selected) | Val AUROC |
|---|---|---|---|
| `qwen/fusion_context_greedy_k2.pth` | with-context | hid_prob_attn (HPA), hid_prob_resid (HPR) | 0.929 |
| `qwen/fusion_question_greedy_k3.pth` | closed-book | hid_resid (HR), hid_prob_attn (HPA), hid_attn (HA) | 0.834 |
| `qwen/fusion_joint_greedy_k3_head.pth` | joint (both settings) | head over the two fusions above | - |

Encoder checkpoints referenced by the heads sit under `qwen/encoders/`.

## Gemma 4 E4B-it (hidden layer 26)

| File | Setting | Branches (greedy-selected) | Val AUROC |
|---|---|---|---|
| `gemma/fusion_context_greedy_k3.pth` | with-context | hidden (H), attention (A), hid_prob (HP) | 0.940 |
| `gemma/fusion_question_greedy_k3.pth` | closed-book | hidden (H), hid_prob_attn (HPA), quad (HPRA) | 0.952 |

Encoder checkpoints under `gemma/encoders/`.

## Format

Each fusion head: `state_dict`, architecture hyperparameters,
`branch_checkpoints` (name + path per branch), `modality_names`, label
definition (`positive_transition`, `binary_target_name`), `best_auc`,
`best_epoch`. Each encoder checkpoint: `state_dict` plus the full feature and
architecture configuration needed to rebuild its dataset transform (channel
counts, span lengths, signal-family switches). No optimizer state is stored.

Retraining any of these from the shipped data reproduces the reported numbers
to seed-level variation; the greedy-k search that selected the branch subsets
is stage 6 of `docs/PIPELINE.md`.
