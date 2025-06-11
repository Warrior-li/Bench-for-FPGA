#include <CL/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 256
#define MAX_MASK_WIDTH 10

// ----------- Kernel -------------
class basic_conv;
void conv1d(cl::sycl::queue &q, float *in, float *mask, float *out, int input_width, int mask_width) {
    cl::sycl::buffer<float, 1> in_buf(in, cl::sycl::range<1>(input_width));
    cl::sycl::buffer<float, 1> out_buf(out, cl::sycl::range<1>(input_width));
    cl::sycl::buffer<float, 1> mask_buf(mask, cl::sycl::range<1>(mask_width));

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

// ----------- CPU reference -------------
void reference(float *in, float *mask, float *out, int input_width, int mask_width) {
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

// ----------- 检查 -------------
int check(float* ref, float* res, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(ref[i] - res[i]) > tol) {
            printf("FAIL at %d: %f vs %f\n", i, ref[i], res[i]);
            return 0;
        }
    }
    return 1;
}

// ----------- C 计时函数 -------------
double get_time_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

// ----------- Main -------------
int main(int argc, char* argv[]) {
    int input_width = 4096, mask_width = 5, repeat = 10;
    if (argc > 1) input_width = atoi(argv[1]);
    if (argc > 2) mask_width = atoi(argv[2]);
    if (argc > 3) repeat = atoi(argv[3]);
    input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    float* input = (float*) malloc(input_width * sizeof(float));
    float* mask  = (float*) malloc(mask_width * sizeof(float));
    float* output = (float*) malloc(input_width * sizeof(float));
    float* ref_out = (float*) malloc(input_width * sizeof(float));

    for (int i = 0; i < input_width; i++) input[i] = (float)(i % 10);
    for (int i = 0; i < mask_width; i++) mask[i] = 1.0f;

    // CPU 参考结果
    reference(input, mask, ref_out, input_width, mask_width);

    #ifdef FPGA_EMULATOR
        sycl::queue q(sycl::intel::fpga_emulator_selector{});
    #else
        sycl::queue q(sycl::default_selector{});
    #endif

    printf("Device: %s\n", q.get_device().get_info<cl::sycl::info::device::name>().c_str());
    printf("Input size: %d, Mask width: %d, Repeat: %d\n", input_width, mask_width, repeat);

    double t_start = get_time_ms();
    for (int i = 0; i < repeat; i++) {
        conv1d(q, input, mask, output, input_width, mask_width);
        q.wait();
    }
    double t_end = get_time_ms();
    printf("[FP32] conv1d: %.3f us, %s\n", (t_end - t_start) * 1000.0 / repeat, check(ref_out, output, input_width, 1e-3f) ? "PASS" : "FAIL");

    free(input);
    free(mask);
    free(output);
    free(ref_out);
    return 0;
}
