#include <torch/extension.h>

std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_quantize", &nvfp4_quantize,
        "bf16 [M, K] -> (packed e2m1 uint8 [M, K/2], e4m3 scales [M, K/16], fp32 outer [M, 1])");
}
