# Triton kernels for FlashMD
#
# Main kernels:
# 1. fused_cutoff_gather_multiply_scatter - fuses cutoff + gather + multiply + scatter
# 2. fused_linear_tanh - fuses linear layer + tanh activation for filter network
# 3. fused_linear_tanh_fp16 - FP16 version for GPTQ W16A16 quantization
# 4. linear_fp16 - FP16 linear (no activation) for GPTQ W16A16 layer 1
# 5. fused_distance_gaussian_rbf_cutoff - fuses distance + Gaussian RBF + cosine cutoff
# 6. fused_tanh_linear - fuses tanh + linear for InteractionBlock (tanh first, then matmul)
# 7. CSR kernels - CSR-based segment reduce for atomic-free scatter-add

from .cfconv_kernels import (
    # Cutoff + gather + multiply + scatter fusion
    fused_cutoff_gather_multiply_scatter_kernel,
    fused_cutoff_gather_multiply_scatter,
    fused_cutoff_gather_multiply_scatter_autograd,
    # Linear + tanh fusion for filter network (FP32)
    fused_linear_tanh_kernel,
    fused_linear_tanh,
    fused_linear_tanh_autograd,
    # FP16 kernels for GPTQ W16A16
    fused_linear_tanh_fp16_kernel,
    fused_linear_tanh_fp16,
    fused_linear_tanh_fp16_autograd,
    linear_fp16_kernel,
    linear_fp16,
    linear_fp16_autograd,
    # Distance + Gaussian RBF + cutoff fusion
    fused_distance_gaussian_rbf_cutoff_kernel,
    fused_distance_gaussian_rbf_cutoff,
    fused_distance_gaussian_rbf_cutoff_autograd,
    # Tanh + linear fusion for InteractionBlock (tanh first, then matmul)
    fused_tanh_linear_kernel,
    fused_tanh_linear,
    fused_tanh_linear_autograd,
)

# CSR kernels for segment reduce
from .csr_kernels import (
    build_csr_index,
    build_src_csr_index,
    fused_csr_cfconv,
    fused_csr_cfconv_autograd,
    fused_src_csr_grad_x,
)

__all__ = [
    "fused_cutoff_gather_multiply_scatter_kernel",
    "fused_cutoff_gather_multiply_scatter",
    "fused_cutoff_gather_multiply_scatter_autograd",
    "fused_linear_tanh_kernel",
    "fused_linear_tanh",
    "fused_linear_tanh_autograd",
    # FP16 for GPTQ
    "fused_linear_tanh_fp16_kernel",
    "fused_linear_tanh_fp16",
    "fused_linear_tanh_fp16_autograd",
    "linear_fp16_kernel",
    "linear_fp16",
    "linear_fp16_autograd",
    # Distance + Gaussian RBF + cutoff fusion
    "fused_distance_gaussian_rbf_cutoff_kernel",
    "fused_distance_gaussian_rbf_cutoff",
    "fused_distance_gaussian_rbf_cutoff_autograd",
    # Tanh + linear fusion for InteractionBlock
    "fused_tanh_linear_kernel",
    "fused_tanh_linear",
    "fused_tanh_linear_autograd",
    # CSR kernels
    "build_csr_index",
    "build_src_csr_index",
    "fused_csr_cfconv",
    "fused_csr_cfconv_autograd",
    "fused_src_csr_grad_x",
]
