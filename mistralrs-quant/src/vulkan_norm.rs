//! Fused residual + RMSNorm kernels for Vulkan (decode is latency-bound on
//! the many small norm/add dispatches of sandwich-norm models such as LFM3).
//!
//! One work-group per row computes, for f32 contiguous rows of `cols`:
//!   a   = (pre  ? rms(x) * w1 : x) + r          (written to `a`)
//!   out = post ? rms(a) * w2  : (unused)        (written to `out`)
//! where `rms(v) = v / sqrt(mean(v^2) + eps)`.
use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::vulkan_backend::{glsl, kernels};
use candle_core::{DType, Result, Storage, Tensor};

const WG: usize = 256;

fn source(_wg: usize) -> String {
    let mut s = glsl::prelude(WG, 6);
    s += r#"
layout(std430, set = 0, binding = 0) readonly buffer B0 { float X[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { float R[]; };
layout(std430, set = 0, binding = 2) readonly buffer B2 { float W1[]; };
layout(std430, set = 0, binding = 3) readonly buffer B3 { float W2[]; };
layout(std430, set = 0, binding = 4) buffer B4 { float A[]; };
layout(std430, set = 0, binding = 5) buffer B5 { float O[]; };
layout(push_constant) uniform PC {
    int rows; int cols; int pre; int post; float eps1; float eps2;
    int xo; int ro; int w1o; int w2o;
} pc;
layout(local_size_x = WG) in;
shared float sh[WG];

float block_sum(float v) {
    int t = int(gl_LocalInvocationID.x);
    sh[t] = v;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    float r = sh[0];
    barrier();
    return r;
}

void main() {
    int row = grow();
    if (row >= pc.rows) return;
    int t = int(gl_LocalInvocationID.x);
    int xb = pc.xo + row * pc.cols;
    int rb = pc.ro + row * pc.cols;
    int ab = row * pc.cols;
    float inv1 = 1.0;
    if (pc.pre != 0) {
        float ss = 0.0;
        for (int c = t; c < pc.cols; c += WG) { float v = X[xb + c]; ss += v * v; }
        inv1 = inversesqrt(block_sum(ss) / float(pc.cols) + pc.eps1);
    }
    float ss2 = 0.0;
    for (int c = t; c < pc.cols; c += WG) {
        float v = X[xb + c];
        if (pc.pre != 0) v = v * inv1 * W1[pc.w1o + c];
        float a = v + R[rb + c];
        A[ab + c] = a;
        ss2 += a * a;
    }
    if (pc.post != 0) {
        float inv2 = inversesqrt(block_sum(ss2) / float(pc.cols) + pc.eps2);
        for (int c = t; c < pc.cols; c += WG) {
            O[ab + c] = A[ab + c] * inv2 * W2[pc.w2o + c];
        }
    }
}
"#;
    s
}

fn eligible(ts: &[&Tensor], cols: usize) -> bool {
    kernels::native_enabled()
        && ts
            .iter()
            .all(|t| t.device().is_vulkan() && t.dtype() == DType::F32 && t.is_contiguous())
        && cols > 0
}

/// Fused `a = (w1.map(rms(x)*w1) | x) + residual`, `out = w2.map(rms(a)*w2)`.
/// Returns `None` when the inputs do not qualify (caller keeps its unfused path).
pub fn vulkan_residual_rms_norm(
    x: &Tensor,
    residual: &Tensor,
    pre: Option<(&Tensor, f32)>,
    post: Option<(&Tensor, f32)>,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let cols = *x.dims().last().unwrap_or(&0);
    let mut all = vec![x, residual];
    all.extend(pre.map(|p| p.0));
    all.extend(post.map(|p| p.0));
    if !eligible(&all, cols)
        || x.shape() != residual.shape()
        || pre.is_some_and(|(w, _)| w.elem_count() != cols)
        || post.is_some_and(|(w, _)| w.elem_count() != cols)
    {
        return Ok(None);
    }
    let rows = x.elem_count() / cols;
    let (xs, xl) = x.storage_and_layout();
    let (rs, rl) = residual.storage_and_layout();
    let w1 = pre.map(|(w, _)| w.storage_and_layout());
    let w2 = post.map(|(w, _)| w.storage_and_layout());
    // Borrow the buffer handles (cloning a VulkanStorage copies it on the device).
    let vk = |s: &Storage| match s {
        Storage::Vulkan(s) => Some(s.buf()),
        _ => None,
    };
    let (Some(xb), Some(rb)) = (vk(&xs), vk(&rs)) else {
        return Ok(None);
    };
    let w1v = w1.as_ref().and_then(|(s, _)| vk(s));
    let w2v = w2.as_ref().and_then(|(s, _)| vk(s));
    let dev = match &*xs {
        Storage::Vulkan(s) => s.device().clone(),
        _ => return Ok(None),
    };
    let a = dev.alloc(DType::F32, rows * cols)?;
    let out = if post.is_some() {
        Some(dev.alloc(DType::F32, rows * cols)?)
    } else {
        None
    };
    // Unused bindings point at valid buffers.
    let w1b = w1v.unwrap_or(xb);
    let w2b = w2v.unwrap_or(xb);
    let ob = out.as_ref().map_or(a.buf(), |s| s.buf());
    let push = kernels::Push::default()
        .us(rows)?
        .us(cols)?
        .i(i32::from(pre.is_some()))
        .i(i32::from(post.is_some()))
        .f(pre.map_or(0.0, |p| p.1))
        .f(post.map_or(0.0, |p| p.1))
        .us(xl.start_offset())?
        .us(rl.start_offset())?
        .us(w1.as_ref().map_or(0, |(_, l)| l.start_offset()))?
        .us(w2.as_ref().map_or(0, |(_, l)| l.start_offset()))?;
    let lim = dev.limits();
    let gx = rows.min(lim.max_groups_x as usize).max(1);
    let gy = rows.div_ceil(gx);
    dev.launch(
        "mistralrs_residual_rms_norm",
        source,
        &[xb, rb, w1b, w2b, a.buf(), ob],
        push,
        &[],
        [gx as u32, gy as u32, 1],
    )?;
    let shape = x.shape().clone();
    let a = Tensor::from_storage(Storage::Vulkan(a), shape.clone(), BackpropOp::none(), false);
    let out =
        out.map(|o| Tensor::from_storage(Storage::Vulkan(o), shape, BackpropOp::none(), false));
    Ok(Some((a, out)))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, w, eps)
    }

    #[test]
    fn vulkan_residual_rms_norm_matches_unfused() -> Result<()> {
        let vk = match Device::new_vulkan(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let cpu = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (3, 2048), &cpu)?;
        let r = Tensor::randn(0f32, 1.0, (3, 2048), &cpu)?;
        let w1 = Tensor::randn(1f32, 0.1, 2048, &cpu)?;
        let w2 = Tensor::randn(1f32, 0.1, 2048, &cpu)?;
        let g = |t: &Tensor| t.to_device(&vk);
        let close = |a: &Tensor, b: &Tensor, what: &str| -> Result<()> {
            let d = (a.to_device(&cpu)? - b)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(d < 1e-4, "{what}: {d}");
            Ok(())
        };
        // add + norm
        let (a, o) =
            super::vulkan_residual_rms_norm(&g(&x)?, &g(&r)?, None, Some((&g(&w2)?, 1e-6)))?
                .unwrap();
        let sum = (&x + &r)?;
        close(&a, &sum, "add sum")?;
        close(&o.unwrap(), &rms(&sum, &w2, 1e-6)?, "add norm")?;
        // norm + residual (+ norm)
        let (a, o) = super::vulkan_residual_rms_norm(
            &g(&x)?,
            &g(&r)?,
            Some((&g(&w1)?, 1e-5)),
            Some((&g(&w2)?, 1e-6)),
        )?
        .unwrap();
        let res = (rms(&x, &w1, 1e-5)? + &r)?;
        close(&a, &res, "residual")?;
        close(&o.unwrap(), &rms(&res, &w2, 1e-6)?, "residual then norm")?;
        Ok(())
    }
}
