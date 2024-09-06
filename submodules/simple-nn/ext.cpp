#include <torch/extension.h>
#include "simple_nn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nearestNeighbor", &nearestNeighborCUDA);
}
