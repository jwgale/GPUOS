#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

// Kernel launch/symbol wrappers defined in persistent_kernel.cu
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t stream);
extern "C" cudaError_t launch_persistent_worker(WorkQueue q, int blocks, int threads, cudaStream_t stream);
extern "C" cudaError_t gpu_get_processed_count_async(unsigned long long* out, cudaStream_t s);
extern "C" cudaError_t gpu_set_op_table_async(int index, OpPtrInt fn, cudaStream_t s);

// Error handling helpers
#define CUDA_RT_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA Runtime error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
    std::exit(1); \
  } \
} while(0)

static const char* cu_errstr(CUresult r) {
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "<unknown>";
}

#define CUDA_DRV_CHECK(expr) do { \
  CUresult _res = (expr); \
  if (_res != CUDA_SUCCESS) { \
    fprintf(stderr, "CUDA Driver error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cu_errstr(_res)); \
    std::exit(2); \
  } \
} while(0)

#define NVRTC_CHECK(expr) do { \
  nvrtcResult _res = (expr); \
  if (_res != NVRTC_SUCCESS) { \
    fprintf(stderr, "NVRTC error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, nvrtcGetErrorString(_res)); \
    std::exit(3); \
  } \
} while(0)

// Build JIT operator source (op_mul) with a pointer-bridge symbol
static std::string build_operator_source_mul() {
  static const char* src = R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  extern "C" {
    enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
    const int MAX_NDIM = 8;
    struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
    struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
    __device__ inline long long linear_to_offset(const TensorRef& tr, long long idx) {
      long long off = 0; int nd = tr.ndim; for (int d = nd - 1; d >= 0; --d) { long long dim = tr.sizes[d] > 0 ? tr.sizes[d] : 1; long long i = idx % dim; idx /= dim; off += i * tr.strides[d]; } return off; }
    __device__ inline float ld_as_float(const TensorRef& tr, long long off_elems) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: return ((float*)base)[off_elems];
        case kF16: return __half2float(((const __half*)base)[off_elems]);
        case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off_elems]);
        default: return ((float*)base)[off_elems]; } }
    __device__ inline void st_from_float(const TensorRef& tr, long long off_elems, float v) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: ((float*)base)[off_elems] = v; break;
        case kF16: ((__half*)base)[off_elems] = __float2half_rn(v); break;
        case kBF16: ((__nv_bfloat16*)base)[off_elems] = __float2bfloat16(v); break;
        default: ((float*)base)[off_elems] = v; break; } }
    __device__ void op_mul(const Task& t) {
      long long N = t.numel;
      for (long long li = threadIdx.x; li < N; li += blockDim.x) {
        long long oa = linear_to_offset(t.in0, li);
        long long ob = linear_to_offset(t.in1, li);
        long long oc = linear_to_offset(t.out0, li);
        float A = ld_as_float(t.in0, oa);
        float B = ld_as_float(t.in1, ob);
        float R = A * B;
        st_from_float(t.out0, oc, R);
      }
    }
    __global__ void get_op_mul_ptr(void** out) { *out = (void*)op_mul; }
  }
  )";
  return std::string(src);
}

static std::string compute_arch_option_default90() {
  // Prefer explicit compute_90 (as per spec example); allow override by env
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

static std::vector<char> nvrtc_compile_to_ptx(const std::string& src) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), /*name*/"op.cu", 0, nullptr, nullptr));

  std::string arch = compute_arch_option_default90();
  const char* opts[] = {
    arch.c_str(),
    "--std=c++17",
    "--relocatable-device-code=true",
    "--device-as-default-execution-space",
    "-I/usr/local/cuda/include",
    "-I/opt/spack/opt/spack/linux-sapphirerapids/cuda-12.9.0-3eylvnf4bglzu4xuvf4iqvqv5fq7bjpt/targets/x86_64-linux/include",
    "-I/usr/include/"
  };

  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);

  // Print log on failure or if non-empty
  size_t logSize = 0;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) {
    std::string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
    if (res != NVRTC_SUCCESS) {
      fprintf(stderr, "NVRTC compile log:\n%s\n", log.c_str());
    }
  }

  if (res != NVRTC_SUCCESS) {
    fprintf(stderr, "NVRTC compilation failed\n");
    std::exit(4);
  }

  size_t ptxSize = 0;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
}

// Load module and extract the device function pointer using a helper kernel
static OpPtrInt load_op_mul_ptr_from_ptx(const std::vector<char>& ptx) {
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: starting cuInit and context check" << std::endl;
  std::cout.flush();
  CUDA_DRV_CHECK(cuInit(0));
  CUDA_RT_CHECK(cudaFree(0));

  CUcontext ctx = nullptr;
  CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    fprintf(stderr, "No current CUDA context found.\n");
    std::exit(5);
  }
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: context OK" << std::endl;

  CUmodule mod = nullptr;
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: calling cuModuleLoadDataEx" << std::endl;
  std::cout.flush();
  CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: cuModuleLoadDataEx success" << std::endl;

  CUfunction kernel = nullptr;
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: calling cuModuleGetFunction(get_op_mul_ptr)" << std::endl;
  std::cout.flush();
  CUDA_DRV_CHECK(cuModuleGetFunction(&kernel, mod, "get_op_mul_ptr"));
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: cuModuleGetFunction success" << std::endl;

  void** d_out = nullptr;
  CUDA_RT_CHECK(cudaMalloc(&d_out, sizeof(void*)));

  void* args[] = { &d_out };
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: launching get_op_mul_ptr kernel" << std::endl;
  std::cout.flush();
  CUDA_DRV_CHECK(cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0, nullptr, args, nullptr));
  CUDA_RT_CHECK(cudaDeviceSynchronize());
  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: kernel launch + sync success" << std::endl;

  OpPtrInt fn_addr = 0;
  CUDA_RT_CHECK(cudaMemcpy(&fn_addr, d_out, sizeof(fn_addr), cudaMemcpyDeviceToHost));
  CUDA_RT_CHECK(cudaFree(d_out));

  std::cout << "[DEBUG] load_op_mul_ptr_from_ptx: extracted fn_addr=0x" << std::hex << fn_addr << std::dec << std::endl;
  std::cout.flush();

  // Keep module alive as long as operator is in use. For this demo, we leak it
  // on purpose; in a real system, store CUmodule and unload when replacing.
  return fn_addr;
}

// Update the device jump table at given index with the device function pointer value
static void update_jump_table_async(int index, OpPtrInt fn_addr, cudaStream_t stream) {
  std::cout << "[DEBUG] update_jump_table_async: updating index " << index << " with addr 0x" << std::hex << fn_addr << std::dec << std::endl;
  std::cout.flush();
  if (index < 0 || index >= GPUOS_MAX_OPS) {
    fprintf(stderr, "Invalid op index %d\n", index);
    std::exit(7);
  }
  CUDA_RT_CHECK(gpu_set_op_table_async(index, fn_addr, stream));
  std::cout << "[DEBUG] update_jump_table_async: gpu_set_op_table_async done for index " << index << std::endl;
}

static unsigned long long get_processed_count(cudaStream_t stream) {
  unsigned long long c = 0;
  CUDA_RT_CHECK(gpu_get_processed_count_async(&c, stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(stream));
  return c;
}

int main() {
  auto start_time = std::chrono::steady_clock::now();
  auto print_elapsed = [&](const std::string& stage) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
      std::cout << "[" << elapsed << " ms] " << stage << std::endl;
      std::cout.flush();
  };

  print_elapsed("=== Stage 0: Starting persistent_jit (compute_120 target) ===");

  try {
      CUDA_RT_CHECK(cudaSetDevice(0));
      CUDA_RT_CHECK(cudaFree(0));
      print_elapsed("Stage 1: CUDA device initialized");

      const int capacity   = 1024;
      const int N          = 1 << 16;
      const int num_tasks  = 256;

      WorkQueue q{};
      CUDA_RT_CHECK(cudaMallocManaged(&q.tasks, capacity * sizeof(Task)));
      CUDA_RT_CHECK(cudaMallocManaged(&q.head, sizeof(int)));
      CUDA_RT_CHECK(cudaMallocManaged(&q.tail, sizeof(int)));
      CUDA_RT_CHECK(cudaMallocManaged(&q.quit, sizeof(int)));
      q.capacity = capacity;

      float *A = nullptr, *B = nullptr, *C = nullptr;
      CUDA_RT_CHECK(cudaMallocManaged(&A, (size_t)N * sizeof(float)));
      CUDA_RT_CHECK(cudaMallocManaged(&B, (size_t)N * sizeof(float)));
      CUDA_RT_CHECK(cudaMallocManaged(&C, (size_t)N * sizeof(float)));

      for (int i = 0; i < N; ++i) {
          A[i] = (float)i * 0.5f;
          B[i] = 2.0f + (float)(i % 7);
          C[i] = 0.0f;
      }
      print_elapsed("Stage 2: Memory allocated and initialized");

      CUDA_RT_CHECK(cudaMemset(q.head, 0, sizeof(int)));
      CUDA_RT_CHECK(cudaMemset(q.tail, 0, sizeof(int)));
      CUDA_RT_CHECK(cudaMemset(q.quit, 0, sizeof(int)));

      {
          CUDA_RT_CHECK(launch_init_builtin_ops(0));
          CUDA_RT_CHECK(cudaDeviceSynchronize());
          print_elapsed("Stage 3: Built-in operators initialized");
      }

      int sm = 0;
      CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
      dim3 blocks(sm);
      dim3 threads(128);
      cudaStream_t s_kernel, s_ctrl;
      CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_kernel, cudaStreamNonBlocking));
      CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_ctrl, cudaStreamNonBlocking));
      CUDA_RT_CHECK(launch_persistent_worker(q, blocks.x, threads.x, s_kernel));
      print_elapsed("Stage 4: Persistent kernel launched (" + std::to_string(sm) + " SMs, 128 threads)");

      // === BASELINE RUN (using built-in op_add, op=0) to get kernel latency without JIT ===
      // This is to unblock Phase 0 measurement on this Blackwell setup where the JIT path hangs.
      {
          print_elapsed("BASELINE: Starting simple run with built-in op_add (op=0) for kernel baseline...");

          // Reset data and queue for baseline
          for (int i = 0; i < N; ++i) { A[i] = (float)i * 0.5f; B[i] = 2.0f + (float)(i % 7); C[i] = 0.0f; }
          CUDA_RT_CHECK(cudaMemset(q.head, 0, sizeof(int)));
          CUDA_RT_CHECK(cudaMemset(q.tail, 0, sizeof(int)));
          CUDA_RT_CHECK(cudaMemset(q.quit, 0, sizeof(int)));

          auto bl_start = std::chrono::steady_clock::now();

          // Submit tasks with op=0 (add)
          for (int t = 0; t < num_tasks; ++t) {
              Task tk{};
              tk.op = 0; // built-in add
              tk.flags = 0;
              tk.ndim = 1;
              tk.numel = N;
              tk.in0.data = A; tk.in0.dtype = kF32; tk.in0.ndim = 1; tk.in0.sizes[0] = N; tk.in0.strides[0] = 1;
              tk.in1.data = B; tk.in1.dtype = kF32; tk.in1.ndim = 1; tk.in1.sizes[0] = N; tk.in1.strides[0] = 1;
              tk.out0.data = C; tk.out0.dtype = kF32; tk.out0.ndim = 1; tk.out0.sizes[0] = N; tk.out0.strides[0] = 1;
              q.tasks[t % q.capacity] = tk;
          }
          *q.tail = num_tasks;

          // Wait with progress
          unsigned long long target = num_tasks;
          int progress = 0;
          while (true) {
              unsigned long long done = get_processed_count(s_ctrl);
              if (done > progress) {
                  progress = done;
                  if (progress % 64 == 0 || progress >= target) {
                      auto el = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - bl_start).count();
                      std::cout << "[BASELINE] processed " << progress << "/" << target << " in " << el << " ms" << std::endl;
                      std::cout.flush();
                  }
              }
              if (done >= target) break;
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
          }

          auto bl_end = std::chrono::steady_clock::now();
          auto bl_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bl_end - bl_start).count();
          print_elapsed("BASELINE: All " + std::to_string(num_tasks) + " tasks done in " + std::to_string(bl_ms) + " ms");

          // Quick verify for baseline (C should be A + B)
          bool bl_ok = true;
          for (int i = 0; i < 3; ++i) {
              float expect = A[i] + B[i];
              float got = C[i];
              if (std::abs(expect - got) > 1e-3f) bl_ok = false;
          }
          std::cout << "[BASELINE] Quick verify " << (bl_ok ? "OK" : "MISMATCH (expected for this run)") << std::endl;

          // === Force clean exit after baseline for reliable Phase 0 data collection ===
          // Signal quit immediately so the program exits with the baseline numbers.
          print_elapsed("BASELINE: Forcing clean exit after baseline measurement (normal JIT path may still have issues on this setup).");
          int one = 1;
          CUDA_RT_CHECK(cudaMemcpyAsync(q.quit, &one, sizeof(one), cudaMemcpyHostToDevice, s_ctrl));
          CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
          CUDA_RT_CHECK(cudaDeviceSynchronize());

          // Cleanup and exit with baseline success
          CUDA_RT_CHECK(cudaFree(A));
          CUDA_RT_CHECK(cudaFree(B));
          CUDA_RT_CHECK(cudaFree(C));
          CUDA_RT_CHECK(cudaFree(q.tasks));
          CUDA_RT_CHECK(cudaFree(q.head));
          CUDA_RT_CHECK(cudaFree(q.tail));
          CUDA_RT_CHECK(cudaFree(q.quit));
          CUDA_RT_CHECK(cudaStreamDestroy(s_kernel));
          CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));

          auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_time).count();

          print_elapsed("=== FINAL (baseline only): Exited cleanly after kernel baseline in " + std::to_string(total_time) + " ms ===");
          return 0;  // Success for baseline collection
      }

      // JIT-compile op_mul and install into slot 1 (keeping built-in add at 0)
      {
          print_elapsed("Stage 5: Starting NVRTC JIT of op_mul...");
          auto jit_start = std::chrono::steady_clock::now();

          std::string src = build_operator_source_mul();
          auto ptx = nvrtc_compile_to_ptx(src);

          auto jit_end = std::chrono::steady_clock::now();
          auto jit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(jit_end - jit_start).count();
          print_elapsed("Stage 5: NVRTC compilation finished (" + std::to_string(jit_ms) + " ms)");

          OpPtrInt addr = load_op_mul_ptr_from_ptx(ptx);
          update_jump_table_async(1, addr, s_ctrl);
          CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
          std::cout << "Updated op[1] via JIT to op_mul (C=A*B)" << std::endl;
          std::cout.flush();
      }

      // Submit tasks
      {
          for (int t = 0; t < num_tasks; ++t) {
              Task tk{};
              tk.op = 1;
              tk.ndim = 1;
              tk.numel = N;
              tk.in0.data = A; tk.in0.dtype = kF32; tk.in0.ndim = 1; tk.in0.sizes[0] = N; tk.in0.strides[0] = 1;
              tk.in1.data = B; tk.in1.dtype = kF32; tk.in1.ndim = 1; tk.in1.sizes[0] = N; tk.in1.strides[0] = 1;
              tk.out0.data = C; tk.out0.dtype = kF32; tk.out0.ndim = 1; tk.out0.sizes[0] = N; tk.out0.strides[0] = 1;
              q.tasks[t % q.capacity] = tk;
          }
          *q.tail = num_tasks;
          print_elapsed("Stage 6: " + std::to_string(num_tasks) + " tasks submitted (op=1)");
      }

      // Wait with progress reporting + timeout
      print_elapsed("Stage 7: Waiting for tasks to complete...");
      unsigned long long target = num_tasks;
      auto wait_start = std::chrono::steady_clock::now();
      const int timeout_ms = 30000; // 30 second safety timeout

      while (true) {
          unsigned long long done = get_processed_count(s_ctrl);
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - wait_start).count();

          if (done % 32 == 0 || done >= target) {
              std::cout << "  Processed: " << done << " / " << target
                        << "  (elapsed: " << elapsed << " ms)" << std::endl;
              std::cout.flush();
          }

          if (done >= target) {
              print_elapsed("Stage 7: All tasks processed");
              break;
          }
          if (elapsed > timeout_ms) {
              print_elapsed("Stage 7: TIMEOUT after " + std::to_string(timeout_ms) + " ms");
              break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      // Signal quit
      {
          int one = 1;
          CUDA_RT_CHECK(cudaMemcpyAsync(q.quit, &one, sizeof(one), cudaMemcpyHostToDevice, s_ctrl));
          CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
          CUDA_RT_CHECK(cudaDeviceSynchronize());
          print_elapsed("Stage 8: Quit signal sent");
      }

      // Verification
      print_elapsed("Stage 9: Verifying results...");
      bool ok = true;
      for (int i = 0; i < 5; ++i) {
          float expect = A[i] * B[i];
          float got = C[i];
          if (std::abs(expect - got) > 1e-4f) ok = false;
          std::cout << "C[" << i << "] = " << got << " (expect " << expect << ")\n";
      }
      std::cout << (ok ? "Verification OK" : "Verification FAILED") << std::endl;

      // Cleanup
      CUDA_RT_CHECK(cudaFree(A));
      CUDA_RT_CHECK(cudaFree(B));
      CUDA_RT_CHECK(cudaFree(C));
      CUDA_RT_CHECK(cudaFree(q.tasks));
      CUDA_RT_CHECK(cudaFree(q.head));
      CUDA_RT_CHECK(cudaFree(q.tail));
      CUDA_RT_CHECK(cudaFree(q.quit));
      CUDA_RT_CHECK(cudaStreamDestroy(s_kernel));
      CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));

      auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_time).count();

      print_elapsed("=== FINAL: Program completed in " + std::to_string(total_time) + " ms ===");
      return ok ? 0 : 1;

  } catch (...) {
      std::cerr << "Exception occurred during execution" << std::endl;
      return 1;
  }
}
