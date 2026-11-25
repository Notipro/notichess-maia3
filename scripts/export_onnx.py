# Copyright (C) 2026 NotiChess contributors
# SPDX-License-Identifier: AGPL-3.0-only
#
# The Maia-3 model and the code this script builds on (github.com/CSSLab/maia3)
# belong to the CSSLab, University of Toronto, and are licensed under the AGPL-3.0.

"""Export the Maia3-5M checkpoint to the ONNX graph that notichess-maia3 loads.

Usage (everything pinned; works offline once the packages are in the uv cache):

    git clone https://github.com/CSSLab/maia3 /tmp/maia3
    git -C /tmp/maia3 checkout 1e13597c42d4858b7cfd7cfdae01e297263364b2
    curl -L -o maia3-5m.pt \
      https://huggingface.co/UofTCSSLab/Maia3-5M/resolve/b6559de2398d7140b985f28fd2c19fb5e47ddabe/maia3-5m.pt
    uv run --no-project --python 3.12 \
      --with torch==2.5.1 --with numpy==2.1.3 --with chess==1.11.1 --with onnx==1.17.0 \
      python scripts/export_onnx.py --maia3-src /tmp/maia3 \
        --checkpoint maia3-5m.pt --output maia3-5m.onnx

The model is built and loaded with the official code (`maia3.uci.load_model`,
`maia3-5m` preset of `maia3.model_registry`). The exported forward pass differs
from `MAIA3Model.forward` in form only, and the script checks the two agree
before writing anything:

1. RMSNorm is written out (x / sqrt(mean(x^2) + eps) * weight): PyTorch 2.5
   has no opset 17 export for `aten::rms_norm`.
2. The input projection `Linear(concat(tokens, elo_self_emb, elo_oppo_emb))`
   becomes three matmuls summed. tract 0.23 cannot optimize the
   concat-then-MatMul pattern ("Trying to substitute a 64,batch_size,512 by
   1,batch_size,512"); the split graph optimizes and runs about 3x faster.
3. The policy head uses matmul and a vectorized promotion block (same index
   order as the official loops); the ponder head is dropped (unused).

Inputs: tokens f32[1,64,97], elo_self f32[1], elo_oppo f32[1].
Outputs: logits_move f32[1,4352], logits_value f32[1,3] (loss, draw, win).
Batch size is fixed to 1. Weights stay float32.
"""

import argparse
import hashlib
import math
import sys
import types
from pathlib import Path

CHECKPOINT_SHA256 = "ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f"
OPSET = 17


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rms_forward(self, x):
    import torch

    eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * self.weight


def split_forward(self, tokens, self_elos, oppo_elos):
    import torch
    import torch.nn.functional as F

    cfg = self.cfg
    tokens = tokens[:, :, : 12 * cfg.history]
    se = self.interpolate_elo(self_elos)
    oe = self.interpolate_elo(oppo_elos)
    w = self.token_projection.weight
    nt = tokens.shape[-1]
    d = cfg.dim_emb
    x = (
        tokens @ w[:, :nt].T
        + (se @ w[:, nt : nt + d].T).unsqueeze(1)
        + (oe @ w[:, nt + d :].T).unsqueeze(1)
        + self.token_projection.bias
    )
    if hasattr(self, "abs_pe"):
        x = self.abs_pe(x)
    x = self.transformer(x)

    sq_from = self.proj_sq_from(x)
    sq_to = self.proj_sq_to(x)
    scores_base = torch.matmul(sq_from, sq_to.transpose(1, 2)) / math.sqrt(cfg.head_hid_dim)
    flat = scores_base.reshape(x.size(0), 4096)
    # Promotions: rank-7 file f -> rank-8 file t, piece p (q, r, b, n) at
    # 4096 + f * 32 + t * 4 + p, exactly the order of the official loops.
    promo = self.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(cfg.head_hid_dim)  # (B, 8 to, 4)
    base = scores_base[:, 48:56, 56:64]  # (B, 8 from, 8 to)
    promotions = (base.unsqueeze(-1) + promo.unsqueeze(1)).reshape(x.size(0), 256)

    xm = self.last_ln(x.mean(dim=1))
    value = self.fc_value(F.relu(self.fc_value_hid(xm)))
    return torch.cat([flat, promotions], dim=1), value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maia3-src", required=True, help="checkout of github.com/CSSLab/maia3")
    parser.add_argument("--checkpoint", required=True, help="maia3-5m.pt from UofTCSSLab/Maia3-5M")
    parser.add_argument("--output", default="maia3-5m.onnx")
    args = parser.parse_args()

    digest = sha256(args.checkpoint)
    if digest != CHECKPOINT_SHA256:
        sys.exit(f"unexpected checkpoint sha256 {digest}, expected {CHECKPOINT_SHA256}")

    sys.path.insert(0, str(Path(args.maia3_src).resolve()))
    import torch
    from maia3.model_registry import resolve_model_spec
    from maia3.uci import load_model

    spec = resolve_model_spec("maia3-5m")
    cfg = types.SimpleNamespace(**spec.config, device="cpu", checkpoint_path=args.checkpoint, trust_checkpoint=False)
    official = load_model(cfg)
    assert all(p.dtype == torch.float32 for p in official.parameters()), "expected float32 weights"

    # Reference outputs from the untouched official model.
    torch.manual_seed(0)
    probe_tokens = (torch.rand(4, 64, 12 * cfg.history + 1) < 0.1).float()
    probe_elos = [(600, 2600), (1500, 1500), (2600, 600), (1234, 1876)]
    with torch.no_grad():
        reference = [
            official(probe_tokens[i : i + 1], torch.tensor([float(a)]), torch.tensor([float(b)]))[:2]
            for i, (a, b) in enumerate(probe_elos)
        ]

    # Same module, rewritten forward.
    for module in official.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.forward = types.MethodType(rms_forward, module)
    official.forward = types.MethodType(split_forward, official)
    with torch.no_grad():
        worst = 0.0
        for i, (a, b) in enumerate(probe_elos):
            got = official(probe_tokens[i : i + 1], torch.tensor([float(a)]), torch.tensor([float(b)]))
            for g, r in zip(got, reference[i]):
                worst = max(worst, float((g - r).abs().max()))
    print(f"max |rewritten - official| on logits: {worst:.2e}")
    if worst > 1e-3:
        sys.exit("rewritten forward pass does not match the official model")

    tokens = torch.zeros(1, 64, 12 * cfg.history + 1)
    elo = torch.tensor([1500.0])
    torch.onnx.export(
        official,
        (tokens, elo, elo),
        args.output,
        input_names=["tokens", "elo_self", "elo_oppo"],
        output_names=["logits_move", "logits_value"],
        opset_version=OPSET,
        dynamo=False,
    )
    digest = sha256(args.output)
    Path(args.output + ".sha256").write_text(f"{digest}  {Path(args.output).name}\n")
    print(f"{digest}  {args.output}")


if __name__ == "__main__":
    main()
