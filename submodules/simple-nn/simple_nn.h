#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> nearestNeighborCUDA(
    const torch::Tensor& coefs,
    const torch::Tensor& codebook);