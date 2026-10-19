// Experimental synchronous matrix backend. No model structure or tokenizer lives here.
// Matrices can be streamed from host mappings or uploaded once and kept on the device.
#include <node_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(napi_status s) { if (s != napi_ok) throw std::runtime_error("Invalid native API argument."); }
static void cu(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
static void bl(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(s)); }
static const napi_type_tag contextTag = {0x5350524f55544355ULL, 0x434f4e5445585431ULL};
static const napi_type_tag matrixTag = {0x5350524f55544d41ULL, 0x4d41545249583031ULL};
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct DeviceBuffer {
  void* ptr = nullptr; size_t capacity = 0;
  void ensure(size_t bytes) {
    if (bytes <= capacity) return;
    if (ptr) { cu(cudaFree(ptr)); ptr = nullptr; capacity = 0; }
    cu(cudaMalloc(&ptr, bytes)); capacity = bytes;
  }
  void clear() { if (ptr) cudaFree(ptr); ptr = nullptr; capacity = 0; }
};
struct Context;
struct ImportedMemory {
  Context* owner; int device; size_t elements; void* ptr = nullptr;
  ImportedMemory(Context* c, int d, size_t n): owner(c), device(d), elements(n) {}
  void close() { if (ptr) { cudaSetDevice(device); cudaIpcCloseMemHandle(ptr); ptr = nullptr; } }
  ~ImportedMemory() { close(); }
};
struct ResidentWeight {
  Context* owner; int device; DeviceBuffer storage;
  ResidentWeight(Context* c, int d): owner(c), device(d) {}
  void close() { if (storage.ptr) { cudaSetDevice(device); storage.clear(); } }
  ~ResidentWeight() { close(); }
};
// Each captured plan owns stable pinned metadata and its matching device buffer.
struct GraphPlan {
  Context* owner; int device; size_t pages, blockSize;
  DeviceBuffer inputs;
  int32_t* host = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool sample, sampled = false, closed = false;
  std::shared_ptr<ResidentWeight> output;
  size_t outputSize;
  size_t vocab = INT32_MAX;
  double calls = 0, tiles = 0, ops = 0;
  Clock::time_point started;
  GraphPlan(Context* c, int d): owner(c), device(d) {}
  void close() {
    if (closed) return;
    cudaSetDevice(device);
    if (exec) cudaGraphExecDestroy(exec);
    exec = nullptr; inputs.clear();
    if (host) cudaFreeHost(host);
    host = nullptr; output.reset(); closed = true;
  }
  ~GraphPlan() { close(); }
};
struct Context {
  int device; size_t tileBytes; bool closed = false, directWeights;
  cudaStream_t stream = nullptr; cudaEvent_t begin = nullptr, end = nullptr; cublasHandle_t handle = nullptr;
  DeviceBuffer matrix, raw, input, output;
  DeviceBuffer blockTable, sampledToken;
  std::vector<int32_t> blockPages;
  std::vector<std::weak_ptr<ResidentWeight>> weights;
  std::vector<std::weak_ptr<ResidentWeight>> tensors;
  std::vector<std::weak_ptr<ImportedMemory>> imports;
  double ipcCopyCalls = 0, ipcCopyBytes = 0, ipcCopyMs = 0;
  double calls = 0, tiles = 0, h2dBytes = 0, d2hBytes = 0, uploadMs = 0, gpuComputeMs = 0, wallMs = 0, peakScratch = 0;
  double weightUploadBytes = 0, weightUploadMs = 0, matvecWeightUploadBytes = 0, peakResident = 0;
  double deviceOps = 0, deviceForwards = 0, deviceForwardMs = 0, metadataBytes = 0;
  bool tokenActive = false;
  GraphPlan* capturing = nullptr;
  std::vector<std::weak_ptr<GraphPlan>> graphs;
  double graphCaptures = 0, graphLaunches = 0, graphCaptureMs = 0, graphNodes = 0, graphInputBytes = 0;
  void abortCapture() {
    if (!capturing) return;
    cudaGraph_t discarded = nullptr;
    cudaStreamEndCapture(stream, &discarded);
    if (discarded) cudaGraphDestroy(discarded);
    calls = capturing->calls; tiles = capturing->tiles; deviceOps = capturing->ops;
    capturing->close(); capturing = nullptr;
  }
  Context(int d, size_t bytes, bool direct): device(d), tileBytes(bytes), directWeights(direct) {}
  void start() {
    cu(cudaSetDevice(device)); cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cu(cudaEventCreate(&begin)); cu(cudaEventCreate(&end));
    bl(cublasCreate(&handle)); bl(cublasSetStream(handle, stream)); bl(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  }
  void activate() { if (closed) throw std::runtime_error("CUDA backend is closed."); cu(cudaSetDevice(device)); }
  size_t residentBytes() const {
    size_t bytes = 0;
    for (const auto& ref : weights) if (auto weight = ref.lock()) bytes += weight->storage.capacity;
    return bytes;
  }
  size_t tensorBytes() const {
    size_t bytes = 0;
    for (const auto& ref : tensors) if (auto tensor = ref.lock()) bytes += tensor->storage.capacity;
    return bytes;
  }
  void close() {
    if (closed) return;
    cudaSetDevice(device);
    abortCapture();
    if (stream) cudaStreamSynchronize(stream);
    for (auto& ref : graphs) if (auto graph = ref.lock()) graph->close();
    graphs.clear();
    for (auto& ref : imports) if (auto memory = ref.lock()) memory->close();
    imports.clear();
    if (handle) cublasDestroy(handle);
    for (auto& ref : weights) if (auto weight = ref.lock()) weight->close();
    weights.clear();
    for (auto& ref : tensors) if (auto tensor = ref.lock()) tensor->close();
    tensors.clear(); blockTable.clear(); sampledToken.clear();
    matrix.clear(); raw.clear(); input.clear(); output.clear();
    if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end);
    if (stream) cudaStreamDestroy(stream);
    closed = true;
  }
  ~Context() { close(); }
};
struct Matrix {
  void* mapping = MAP_FAILED; size_t mappingBytes = 0; const char* data = nullptr;
  size_t rows, cols; int dtype; bool closed = false;
  std::shared_ptr<ResidentWeight> resident;
  void close() {
    if (resident) resident->close();
    resident.reset();
    if (mapping != MAP_FAILED) munmap(mapping, mappingBytes);
    mapping = MAP_FAILED; data = nullptr; closed = true;
  }
  ~Matrix() { close(); }
};

static std::vector<napi_value> args(napi_env e, napi_callback_info info, size_t count) {
  std::vector<napi_value> v(count); size_t n = count; ck(napi_get_cb_info(e, info, &n, v.data(), nullptr, nullptr));
  if (n != count) throw std::runtime_error("Wrong argument count."); return v;
}
static size_t integer(napi_env e, napi_value v, size_t max = 9007199254740991ULL) {
  double n; ck(napi_get_value_double(e, v, &n));
  if (!std::isfinite(n) || n < 0 || std::floor(n) != n || n > static_cast<double>(max)) throw std::runtime_error("Expected a nonnegative safe integer.");
  return static_cast<size_t>(n);
}
static std::string str(napi_env e, napi_value v) {
  size_t n; ck(napi_get_value_string_utf8(e, v, nullptr, 0, &n));
  std::vector<char> b(n + 1); ck(napi_get_value_string_utf8(e, v, b.data(), b.size(), &n));
  if (std::strlen(b.data()) != n) throw std::runtime_error("Embedded NUL in string."); return std::string(b.data(), n);
}
template<class T> static T* unwrap(napi_env e, napi_value v, const napi_type_tag& tag) {
  bool valid = false; ck(napi_check_object_type_tag(e, v, &tag, &valid));
  if (!valid) throw std::runtime_error("Invalid native handle type.");
  void* p; ck(napi_unwrap(e, v, &p)); return static_cast<T*>(p);
}
template<class T> static napi_value wrap(napi_env e, std::unique_ptr<T> p, const napi_type_tag& tag) {
  napi_value v; ck(napi_create_object(e, &v)); ck(napi_type_tag_object(e, v, &tag));
  ck(napi_wrap(e, v, p.get(), [](napi_env, void* ptr, void*) { delete static_cast<T*>(ptr); }, nullptr, nullptr));
  p.release(); return v;
}
static napi_value array(napi_env e, size_t count, float** data) {
  napi_value buffer, result; void* ptr;
  ck(napi_create_arraybuffer(e, count * sizeof(float), &ptr, &buffer));
  ck(napi_create_typedarray(e, napi_float32_array, count, buffer, 0, &result)); *data = static_cast<float*>(ptr); return result;
}
static float* floats(napi_env e, napi_value v, size_t expected) {
  napi_typedarray_type type; size_t count, offset; void* data; napi_value buffer;
  ck(napi_get_typedarray_info(e, v, &type, &count, &data, &buffer, &offset));
  if (type != napi_float32_array || count != expected) throw std::runtime_error("Float32Array shape mismatch."); return static_cast<float*>(data);
}
static napi_value fail(napi_env e, const std::exception& error) { napi_throw_error(e, nullptr, error.what()); return nullptr; }
static napi_value nothing(napi_env e) { napi_value v; ck(napi_get_undefined(e, &v)); return v; }
static void number(napi_env e, napi_value obj, const char* key, double value) {
  napi_value v; ck(napi_create_double(e, value, &v)); ck(napi_set_named_property(e, obj, key, v));
}
static float readScalar(const Matrix& m, size_t index) {
  if (m.dtype == 0) { float f; std::memcpy(&f, m.data + index * 4, 4); return f; }
  uint16_t bits; std::memcpy(&bits, m.data + index * 2, 2);
  if (m.dtype == 1) { uint32_t u = uint32_t(bits) << 16; float f; std::memcpy(&f, &u, 4); return f; }
  const float sign = bits & 0x8000 ? -1.0f : 1.0f;
  int exp = (bits >> 10) & 31, frac = bits & 1023;
  return sign * (exp == 0 ? std::ldexp(float(frac), -24) : exp == 31
    ? (frac ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
    : std::ldexp(1.0f + frac / 1024.0f, exp - 15));
}
__global__ static void convert(const uint16_t* source, float* output, size_t count, int dtype) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = dtype == 1 ? __uint_as_float(uint32_t(source[i]) << 16) : __half2float(reinterpret_cast<const __half*>(source)[i]);
}
#include "cuda-matvec.inc"

static napi_value create(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 3); int d = static_cast<int>(integer(e, a[0], 1024)); size_t mib = integer(e, a[1], 1024);
    bool direct; ck(napi_get_value_bool(e, a[2], &direct));
    if (!mib) throw std::runtime_error("tileMiB must be positive.");
    auto c = std::make_unique<Context>(d, mib * 1024 * 1024, direct); c->start(); return wrap(e, std::move(c), contextTag);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value mapMatrix(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 5); auto path = str(e, a[0]); size_t offset = integer(e, a[1]);
    auto dtype = str(e, a[2]); size_t rows = integer(e, a[3], INT32_MAX), cols = integer(e, a[4], INT32_MAX);
    int type = dtype == "F32" ? 0 : dtype == "BF16" ? 1 : dtype == "F16" ? 2 : -1;
    if (type < 0 || !rows || !cols || rows > SIZE_MAX / cols / 4) throw std::runtime_error("Invalid matrix descriptor.");
    size_t bytes = rows * cols * (type == 0 ? 4 : 2);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open matrix file: " + path);
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 0 || offset > size_t(st.st_size) || bytes > size_t(st.st_size) - offset) {
      ::close(fd); throw std::runtime_error("Matrix exceeds checkpoint file.");
    }
    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE)), base = offset / page * page;
    auto m = std::make_unique<Matrix>(); m->rows = rows; m->cols = cols; m->dtype = type; m->mappingBytes = bytes + offset - base;
    m->mapping = mmap(nullptr, m->mappingBytes, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(base)); ::close(fd);
    if (m->mapping == MAP_FAILED) throw std::runtime_error("Cannot map checkpoint matrix.");
    m->data = static_cast<const char*>(m->mapping) + offset - base;
    return wrap(e, std::move(m), matrixTag);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value row(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 2); auto m = unwrap<Matrix>(e, a[0], matrixTag); size_t index = integer(e, a[1]);
    if (m->closed || index >= m->rows) throw std::runtime_error("Invalid matrix row or closed matrix.");
    float* out; auto result = array(e, m->cols, &out);
    for (size_t j = 0; j < m->cols; j++) out[j] = readScalar(*m, index * m->cols + j);
    return result;
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value uploadMatrix(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 4); auto c = unwrap<Context>(e, a[0], contextTag); auto m = unwrap<Matrix>(e, a[1], matrixTag);
    size_t limit = integer(e, a[2]), reserve = integer(e, a[3]);
    c->activate(); if (m->closed) throw std::runtime_error("Matrix is closed.");
    if (m->resident) throw std::runtime_error("Matrix already has a resident allocation.");
    size_t bytes = m->rows * m->cols * (m->dtype == 0 ? 4 : 2), used = c->residentBytes();
    if (limit && (used > limit || bytes > limit - used)) throw std::runtime_error("Resident weight budget exceeded.");
    size_t freeBytes, total; cu(cudaMemGetInfo(&freeBytes, &total));
    if (freeBytes < reserve || bytes > freeBytes - reserve) throw std::runtime_error("Insufficient free CUDA memory after reserve.");
    auto weight = std::make_shared<ResidentWeight>(c, c->device);
    weight->storage.ensure(bytes);
    auto upload = Clock::now();
    cu(cudaMemcpyAsync(weight->storage.ptr, m->data, bytes, cudaMemcpyHostToDevice, c->stream));
    cu(cudaStreamSynchronize(c->stream));
    double elapsed = ms(upload); c->uploadMs += elapsed; c->weightUploadMs += elapsed;
    c->h2dBytes += bytes; c->weightUploadBytes += bytes;
    c->weights.push_back(weight); m->resident = std::move(weight);
    c->peakResident = std::max(c->peakResident, double(used + bytes));
    return nothing(e);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value matvec(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 3); auto c = unwrap<Context>(e, a[0], contextTag); auto m = unwrap<Matrix>(e, a[1], matrixTag);
    c->activate(); if (m->closed) throw std::runtime_error("Matrix is closed.");
    if (m->resident && (m->resident->owner != c || !m->resident->storage.ptr)) throw std::runtime_error("Resident matrix belongs to a different or closed CUDA backend.");
    float* x = floats(e, a[2], m->cols); float* y; auto result = array(e, m->rows, &y);
    auto wall = Clock::now(); size_t rowsPerTile = c->tileBytes / (m->cols * sizeof(float));
    if (!rowsPerTile) throw std::runtime_error("Matrix row exceeds configured CUDA tile capacity.");
    rowsPerTile = std::min(rowsPerTile, m->rows);
    size_t elements = rowsPerTile * m->cols;
    bool direct = c->directWeights && m->dtype != 0;
    if ((!m->resident && m->dtype == 0) || (m->dtype != 0 && !direct)) c->matrix.ensure(elements * 4);
    if (!m->resident && m->dtype != 0) c->raw.ensure(elements * 2);
    c->input.ensure(m->cols * 4); c->output.ensure(rowsPerTile * 4);
    c->peakScratch = std::max(c->peakScratch, double(c->matrix.capacity + c->raw.capacity + c->input.capacity + c->output.capacity));
    auto upload = Clock::now(); cu(cudaMemcpyAsync(c->input.ptr, x, m->cols * 4, cudaMemcpyHostToDevice, c->stream));
    cu(cudaStreamSynchronize(c->stream)); c->uploadMs += ms(upload); c->h2dBytes += m->cols * 4;
    const float alpha = 1, beta = 0; size_t width = m->dtype == 0 ? 4 : 2;
    for (size_t r = 0; r < m->rows; r += rowsPerTile) {
      size_t n = std::min(rowsPerTile, m->rows - r), count = n * m->cols;
      const void* source;
      if (m->resident) source = static_cast<const char*>(m->resident->storage.ptr) + r * m->cols * width;
      else {
        source = m->dtype == 0 ? c->matrix.ptr : c->raw.ptr;
        upload = Clock::now();
        cu(cudaMemcpyAsync(const_cast<void*>(source), m->data + r * m->cols * width, count * width, cudaMemcpyHostToDevice, c->stream));
        cu(cudaStreamSynchronize(c->stream)); c->uploadMs += ms(upload); c->h2dBytes += count * width;
        c->matvecWeightUploadBytes += count * width;
      }
      cu(cudaEventRecord(c->begin, c->stream));
      if (direct) directMatvec(c, source, static_cast<const float*>(c->input.ptr), static_cast<float*>(c->output.ptr), n, m->cols, m->dtype);
      else {
        if (m->dtype != 0) {
          convert<<<static_cast<unsigned>((count + 255) / 256), 256, 0, c->stream>>>(static_cast<const uint16_t*>(source), static_cast<float*>(c->matrix.ptr), count, m->dtype);
          cu(cudaGetLastError());
        }
        // Row-major [rows, cols] is column-major [cols, rows]; transpose yields W*x.
        bl(cublasSgemv(c->handle, CUBLAS_OP_T, static_cast<int>(m->cols), static_cast<int>(n), &alpha,
          static_cast<const float*>(m->dtype == 0 ? source : c->matrix.ptr), static_cast<int>(m->cols), static_cast<const float*>(c->input.ptr), 1,
          &beta, static_cast<float*>(c->output.ptr), 1));
      }
      cu(cudaEventRecord(c->end, c->stream)); cu(cudaEventSynchronize(c->end));
      float elapsed; cu(cudaEventElapsedTime(&elapsed, c->begin, c->end)); c->gpuComputeMs += elapsed;
      cu(cudaMemcpyAsync(y + r, c->output.ptr, n * 4, cudaMemcpyDeviceToHost, c->stream)); cu(cudaStreamSynchronize(c->stream));
      c->d2hBytes += n * 4; c->tiles++;
    }
    c->calls++; c->wallMs += ms(wall); return result;
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value stats(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 1); auto c = unwrap<Context>(e, a[0], contextTag); c->activate();
    napi_value o; ck(napi_create_object(e, &o)); cudaDeviceProp prop; cu(cudaGetDeviceProperties(&prop, c->device));
    napi_value name; ck(napi_create_string_utf8(e, prop.name, NAPI_AUTO_LENGTH, &name)); ck(napi_set_named_property(e, o, "deviceName", name));
    size_t freeBytes, total; cu(cudaMemGetInfo(&freeBytes, &total));
    number(e,o,"device",c->device); number(e,o,"deviceFreeBytes",freeBytes); number(e,o,"deviceTotalBytes",total);
    number(e,o,"matvecCalls",c->calls); number(e,o,"tiles",c->tiles); number(e,o,"hostToDeviceBytes",c->h2dBytes);
    number(e,o,"deviceToHostBytes",c->d2hBytes); number(e,o,"uploadWallMs",c->uploadMs); number(e,o,"gpuComputeMs",c->gpuComputeMs);
    number(e,o,"matvecWallMs",c->wallMs); number(e,o,"peakScratchBytes",c->peakScratch);
    number(e,o,"residentWeightBytes",c->residentBytes()); number(e,o,"peakResidentWeightBytes",c->peakResident);
    number(e,o,"weightUploadBytes",c->weightUploadBytes); number(e,o,"weightUploadWallMs",c->weightUploadMs);
    number(e,o,"matvecWeightUploadBytes",c->matvecWeightUploadBytes);
    number(e,o,"deviceTensorBytes",c->tensorBytes()); number(e,o,"deviceOpCalls",c->deviceOps);
    number(e,o,"deviceForwards",c->deviceForwards); number(e,o,"deviceForwardEventMs",c->deviceForwardMs);
    number(e,o,"blockTableUploadBytes",c->metadataBytes);
    number(e,o,"graphCaptures",c->graphCaptures); number(e,o,"graphLaunches",c->graphLaunches);
    number(e,o,"graphCaptureWallMs",c->graphCaptureMs); number(e,o,"graphNodes",c->graphNodes);
    number(e,o,"graphInputUploadBytes",c->graphInputBytes);
    number(e,o,"ipcCopyCalls",c->ipcCopyCalls); number(e,o,"ipcCopyBytes",c->ipcCopyBytes); number(e,o,"ipcCopyEventMs",c->ipcCopyMs);
    size_t liveGraphs = 0;
    for (auto& ref : c->graphs) if (auto graph = ref.lock()) if (!graph->closed) liveGraphs++;
    number(e,o,"liveGraphs",liveGraphs); return o;
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value closeContext(napi_env e, napi_callback_info info) {
  try { auto a = args(e,info,1); unwrap<Context>(e,a[0],contextTag)->close(); return nothing(e); }
  catch (const std::exception& err) { return fail(e,err); }
}
static napi_value closeMatrix(napi_env e, napi_callback_info info) {
  try { auto a = args(e,info,1); unwrap<Matrix>(e,a[0],matrixTag)->close(); return nothing(e); }
  catch (const std::exception& err) { return fail(e,err); }
}
#include "cuda-device.inc"

static napi_value init(napi_env e, napi_value exports) {
  napi_property_descriptor properties[] = {
    {"create",nullptr,create,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"mapMatrix",nullptr,mapMatrix,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"uploadMatrix",nullptr,uploadMatrix,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"row",nullptr,row,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"matvec",nullptr,matvec,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"stats",nullptr,stats,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"closeContext",nullptr,closeContext,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"closeMatrix",nullptr,closeMatrix,nullptr,nullptr,nullptr,napi_default,nullptr}
  };
  ck(napi_define_properties(e,exports,sizeof(properties)/sizeof(properties[0]),properties));
  registerDeviceOps(e, exports); return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
