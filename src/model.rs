// Copyright (C) 2026 NotiChess contributors
// SPDX-License-Identifier: AGPL-3.0-only
//
// The Maia-3 model and the code this encoding is derived from
// (github.com/CSSLab/maia3) belong to the CSSLab, University of Toronto,
// and are licensed under the AGPL-3.0.

//! The ONNX graph of `scripts/export_onnx.py`, run by tract (optimized plan,
//! batch 1, single thread).

use std::sync::Arc;

use tract_onnx::prelude::*;

use crate::encoding::{POLICY_SIZE, TOKEN_DIM};

pub struct Model(Arc<TypedRunnableModel>);

impl Model {
    pub fn load(path: &str) -> TractResult<Self> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, 64, TOKEN_DIM]).into())?
            .with_input_fact(1, f32::fact([1]).into())?
            .with_input_fact(2, f32::fact([1]).into())?
            .into_typed()?
            .into_optimized()?
            .into_runnable()?;
        let model = Self(plan);
        // A wrong graph fails here, at `isready`, rather than at the first `go`.
        model.move_logits(vec![0.0; 64 * TOKEN_DIM], 1500.0, 1500.0)?;
        Ok(model)
    }

    /// The 4352 move logits for `tokens` (see `encoding::encode`).
    pub fn move_logits(
        &self,
        tokens: Vec<f32>,
        elo_self: f32,
        elo_oppo: f32,
    ) -> TractResult<Vec<f32>> {
        let outputs = self.0.run(tvec!(
            Tensor::from_shape(&[1, 64, TOKEN_DIM], &tokens)?.into_tvalue(),
            tensor1(&[elo_self]).into_tvalue(),
            tensor1(&[elo_oppo]).into_tvalue(),
        ))?;
        let logits: Vec<f32> = outputs
            .first()
            .ok_or_else(|| TractError::msg("the model returned no output"))?
            .to_plain_array_view::<f32>()?
            .iter()
            .copied()
            .collect();
        if logits.len() != POLICY_SIZE {
            return Err(TractError::msg(format!(
                "the model returned {} move logits, expected {POLICY_SIZE}",
                logits.len()
            )));
        }
        Ok(logits)
    }
}
