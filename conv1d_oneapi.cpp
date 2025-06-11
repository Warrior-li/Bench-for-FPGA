#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_MASK_WIDTH = 10;

// ----------- kernel 1: basic -------------
void conv1d(cl::sycl::queue &q, const std::vector<float> &in, const std::vector<float> &mask,
            std::vector<float> &out, int input_width, int mask_width) {

  cl::sycl::buffer<float, 1> in_buf(in.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> out_buf(out.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> mask_buf(mask.data(), cl::sycl::range<1>(mask_width));

  q.submit([&](cl::sycl::handler &h) {
    auto in_acc = in_buf.get_access<cl::sycl::access::mode::read>(h);
    auto out_acc = out_buf.get_access<cl::sycl::access::mode::write>(h);
    auto mask_acc = mask_buf.get_access<cl::sycl::access::mode::read>(h);

    h.parallel_for<class basic_conv>(cl::sycl::range<1>(input_width), [=](cl::sycl::id<1> i) {
      float sum = 0.0f;
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
void conv1d_tiled(cl::sycl::queue &q, const std::vector<float> &in, const std::vector<float> &mask,
                  std::vector<float> &out, int input_width, int mask_width) {

  cl::sycl::buffer<float, 1> in_buf(in.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> out_buf(out.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> mask_buf(mask.data(), cl::sycl::range<1>(mask_width));

  int halo = mask_width / 2;
  int tile_size = BLOCK_SIZE + 2 * halo;

  q.submit([&](cl::sycl::handler &h) {
    auto in_acc = in_buf.get_access<cl::sycl::access::mode::read>(h);
    auto out_acc = out_buf.get_access<cl::sycl::access::mode::write>(h);
    auto mask_acc = mask_buf.get_access<cl::sycl::access::mode::read>(h);

    cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
        cl::sycl::access::target::local>
      tile_acc(cl::sycl::range<1>(tile_size), h);

    h.parallel_for<class tiled_conv>(
      cl::sycl::nd_range<1>(cl::sycl::range<1>(input_width), cl::sycl::range<1>(BLOCK_SIZE)), [=](cl::sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        int lid = item.get_local_id(0);
        int group = item.get_group(0);
        int local_size = item.get_local_range(0);

        int start = group * local_size - halo;
        // Cooperative load
        for (int i = lid; i < tile_size; i += local_size) {
          int idx = start + i;
          tile_acc[i] = (idx >= 0 && idx < input_width) ? in_acc[idx] : 0.0f;
        }
        item.barrier(cl::sycl::access::fence_space::local_space);

        if (gid < input_width) {
          float sum = 0.0f;
          for (int j = 0; j < mask_width; j++) {
            sum += tile_acc[lid + j] * mask_acc[j];
          }
          out_acc[gid] = sum;
        }
      });
  });
}

// ----------- kernel 3: tiled + cache-aware -------------
void conv1d_tiled_caching(cl::sycl::queue &q, const std::vector<float> &in, const std::vector<float> &mask,
                          std::vector<float> &out, int input_width, int mask_width) {

  cl::sycl::buffer<float, 1> in_buf(in.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> out_buf(out.data(), cl::sycl::range<1>(input_width));
  cl::sycl::buffer<float, 1> mask_buf(mask.data(), cl::sycl::range<1>(mask_width));

  int halo = mask_width / 2;

  q.submit([&](cl::sycl::handler &h) {
    auto in_acc = in_buf.get_access<cl::sycl::access::mode::read>(h);
    auto out_acc = out_buf.get_access<cl::sycl::access::mode::write>(h);
    auto mask_acc = mask_buf.get_access<cl::sycl::access::mode::read>(h);

    cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
        cl::sycl::access::target::local>
      tile_acc(cl::sycl::range<1>(BLOCK_SIZE), h);

    h.parallel_for<class cache_conv>(
      cl::sycl::nd_range<1>(cl::sycl::range<1>(input_width), cl::sycl::range<1>(BLOCK_SIZE)), [=](cl::sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        int lid = item.get_local_id(0);
        int group = item.get_group(0);
        int local_size = item.get_local_range(0);

        tile_acc[lid] = in_acc[gid];
        item.barrier(cl::sycl::access::fence_space::local_space);

        int start = gid - halo;
        float sum = 0.0f;
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
void reference(const std::vector<float> &in, const std::vector<float> &mask,
               std::vector<float> &out, int input_width, int mask_width) {
  for (int i = 0; i < input_width; i++) {
    float sum = 0.0f;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        sum += in[start + j] * mask[j];
      }
    }
    out[i] = sum;
  }
}

bool check(const std::vector<float>& ref, const std::vector<float>& res, float tol=1e-3f) {
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs(ref[i] - res[i]) > tol) {
      std::cout << "FAIL at " << i << ": " << ref[i] << " vs " << res[i] << std::endl;
      return false;
    }
  }
  return true;
}

// ----------- Run + Benchmark -------------
void test_conv1d(cl::sycl::queue& q, int input_width, int mask_width, int repeat) {
  std::vector<float> input(input_width), output(input_width), ref_output(input_width), mask(mask_width);

  for (int i = 0; i < mask_width; i++) mask[i] = 1.0f;
  for (int i = 0; i < input_width; i++) input[i] = static_cast<float>(i % 10);

  // CPU reference
  reference(input, mask, ref_output, input_width, mask_width);

  // 1. basic
  output.assign(input_width, 0.0f);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  auto end = std::chrono::steady_clock::now();
  double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[FP32] conv1d: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;

  // 2. tiled
  output.assign(input_width, 0.0f);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  end = std::chrono::steady_clock::now();
  t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[FP32] conv1d_tiled: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;

  // 3. tiled + caching
  output.assign(input_width, 0.0f);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching(q, input, mask, output, input_width, mask_width);
    q.wait();
  }
  end = std::chrono::steady_clock::now();
  t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)repeat;
  std::cout << "[FP32] conv1d_tiled_caching: " << t << " us, " << (check(ref_output, output) ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char* argv[]) {
  int input_width = 4096, mask_width = 5, repeat = 10;
  if (argc > 1) input_width = std::atoi(argv[1]);
  if (argc > 2) mask_width = std::atoi(argv[2]);
  if (argc > 3) repeat = std::atoi(argv[3]);
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  cl::sycl::queue q; // 由环境变量选择设备

  std::cout << "Device: " << q.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
  std::cout << "Input size: " << input_width << ", Mask width: " << mask_width << ", Repeat: " << repeat << std::endl;

  test_conv1d(q, input_width, mask_width, repeat);
  return 0;
}
