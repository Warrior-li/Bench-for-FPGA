#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>

using namespace sycl;

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_MASK_WIDTH = 10;

// ----------- kernel 1: basic -------------
template <typename T>
void conv1d(queue &q, const std::vector<T> &in, const std::vector<T> &mask,
            std::vector<T> &out, int input_width, int mask_width) {

  buffer<T> in_buf(in.data(), range<1>(input_width));
  buffer<T> out_buf(out.data(), range<1>(input_width));
  buffer<T> mask_buf(mask.data(), range<1>(mask_width));

  q.submit([&](handler &h) {
    auto in_acc = in_buf.template get_access<access::mode::read>(h);
    auto out_acc = out_buf.template get_access<access::mode::write>(h);
    auto mask_acc = mask_buf.template get_access<access::mode::read>(h);

    h.parallel_for(range<1>(input_width), [=](id<1> i) {
      T sum = 0;
      int start = i[0] - mask_width / 2;
      for (int j = 0; j < mask_width; j++) {
        if (start + j >= 0 && start + j < input_width) {
          sum += in_acc[start + j] * mask_acc[j];
        }
      }
      out_acc[i] = sum;
    });
  });
}

// ----------- kernel 2: tiled -------------
template <typename T>
void conv1d_tiled(queue &q, const std::vector<T> &in, const std::vector<T> &mask,
                  std::vector<T> &out, int input_width, int mask_width) {

  buffer<T> in_buf(in.data(), range<1>(input_width));
  buffer<T> out_buf(out.data(), range<1>(input_width));
  buffer<T> mask_buf(mask.data(), range<1>(mask_width));

  int halo = mask_width / 2;
  int tile_size = BLOCK_SIZE + 2 * halo;

  q.submit([&](handler &h) {
    auto in_acc = in_buf.template get_access<access::mode::read>(h);
    auto out_acc = out_buf.template get_access<access::mode::write>(h);
    auto mask_acc = mask_buf.template get_access<access::mode::read>(h);

    accessor<T, 1, access::mode::read_write, access::target::local> tile_acc(range<1>(tile_size), h);

    h.parallel_for(nd_range<1>(range<1>(input_width), range<1>(BLOCK_SIZE)), [=](nd_item<1> item) {
      int gid = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int group = item.get_group(0);
      int local_size = item.get_local_range(0);

      int start = group * local_size - halo;

      // Cooperative load
      for (int i = lid; i < tile_size; i += local_size) {
        int idx = start + i;
        tile_acc[i] = (idx >= 0 && idx < input_width) ? in_acc[idx] : 0;
      }
      item.barrier(access::fence_space::local_space);

      if (gid < input_width) {
        T sum = 0;
        for (int j = 0; j < mask_width; j++) {
          sum += tile_acc[lid + j] * mask_acc[j];
        }
        out_acc[gid] = sum;
      }
    });
  });
}

// ----------- kernel 3: tiled + cache-aware -------------
template <typename T>
void conv1d_tiled_caching(queue &q, const std::vector<T> &in, const std::vector<T> &mask,
                          std::vector<T> &out, int input_width, int mask_width) {

  buffer<T> in_buf(in.data(), range<1>(input_width));
  buffer<T> out_buf(out.data(), range<1>(input_width));
  buffer<T> mask_buf(mask.data(), range<1>(mask_width));

  int halo = mask_width / 2;

  q.submit([&](handler &h) {
    auto in_acc = in_buf.template get_access<access::mode::read>(h);
    auto out_acc = out_buf.template get_access<access::mode::write>(h);
    auto mask_acc = mask_buf.template get_access<access::mode::read>(h);

    accessor<T, 1, access::mode::read_write, access::target::local> tile_acc(range<1>(BLOCK_SIZE), h);

    h.parallel_for(nd_range<1>(range<1>(input_width), range<1>(BLOCK_SIZE)), [=](nd_item<1> item) {
      int gid = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int group = item.get_group(0);
      int local_size = item.get_local_range(0);

      tile_acc[lid] = in_acc[gid];
      item.barrier(access::fence_space::local_space);

      int start = gid - halo;
      T sum = 0;
      for (int j = 0; j < mask_width; j++) {
        int idx = start + j;
        if (idx >= 0 && idx < input_width) {
          if (idx >= group * local_size && idx < (group + 1) * local_size) {
            sum += tile_acc[lid + j - halo] * mask_acc[j];
          } else {
            sum += in_acc[idx] * mask_acc[j];
          }
        }
      }
      out_acc[gid] = sum;
    });
  });
}

// ----------- CPU reference -------------
template <typename T>
void reference(const std::vector<T> &in, const std::vector<T> &mask,
               std::vector<T> &out, int input_width, int mask_width) {
  for (int i = 0; i < input_width; i++) {
    T sum = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        sum += in[start + j] * mask[j];
      }
    }
    out[i] = sum;
  }
}

template <typename T>
bool check(const std::vector<T>& ref, const std::vector<T>& res, float tol=1e-3) {
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs((float)ref[i] - (float)res[i]) > tol) {
      std::cout << "FAIL at " << i << ": " << ref[i] << " vs " << res[i] << std::endl;
      return false;
    }
  }
  return true;
}

// ----------- Run + Benchmark -------------
template<typename T>
void test_conv1d(queue& q, int input_width, int mask_width, int repeat, const std::string& dtype) {
  std::vector<T> input(input_width), output(input_width), ref_output(input_width), mask(mask_width);

  for (int i = 0; i < mask_width; i++) mask[i] = 1;
  for (int i = 0; i < input_width; i++) input[i] = static_cast<T>(i % 10);

  // CPU reference
  reference(input, mask, ref_output, input_width, mask_width);

  // 1. 基础
  output.assign(input_width, 0);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  auto end = std::chrono::steady_clock::now();
  double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[" << dtype << "] conv1d: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;

  // 2. tiled
  output.assign(input_width, 0);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  end = std::chrono::steady_clock::now();
  t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[" << dtype << "] conv1d_tiled: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;

  // 3. tiled + caching
  output.assign(input_width, 0);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  end = std::chrono::steady_clock::now();
  t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[" << dtype << "] conv1d_tiled_caching: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char* argv[]) {
  int input_width = 4096, mask_width = 5, repeat = 10;
  if (argc > 1) input_width = std::atoi(argv[1]);
  if (argc > 2) mask_width = std::atoi(argv[2]);
  if (argc > 3) repeat = std::atoi(argv[3]);
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  // 选择设备
#ifdef FPGA
  ext::intel::fpga_selector selector;
#else
  default_selector selector;
#endif
  queue q(selector);

  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
  std::cout << "Input size: " << input_width << ", Mask width: " << mask_width << ", Repeat: " << repeat << std::endl;

  test_conv1d<double>(q, input_width, mask_width, repeat, "FP64");
  test_conv1d<float>(q, input_width, mask_width, repeat, "FP32");
  test_conv1d<int16_t>(q, input_width, mask_width, repeat, "INT16");
  return 0;
}