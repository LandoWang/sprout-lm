// Generic device operators. Qwen's layer structure remains in TypeScript.
static const napi_type_tag tensorTag = {0x5350524f55545645ULL, 0x43544f5230303031ULL};
struct DeviceTensor {
  std::shared_ptr<ResidentWeight> allocation;
  size_t size;
};
static DeviceTensor* tensor(napi_env e, napi_value value, Context* c) {
  auto t = unwrap<DeviceTensor>(e, value, tensorTag);
  if (t->allocation->owner != c || !t->allocation->storage.ptr) throw std::runtime_error("Device tensor belongs to a different or closed backend.");
  return t;
}
static float* data(DeviceTensor* t) { return static_cast<float*>(t->allocation->storage.ptr); }
static unsigned grid(size_t n) { return static_cast<unsigned>((n + 255) / 256); }
static void sameSize(DeviceTensor* a, DeviceTensor* b) {
  if (a->size != b->size) throw std::runtime_error("Device vector shape mismatch.");
}
static double scalar(napi_env e, napi_value v) {
  double n; ck(napi_get_value_double(e, v, &n));
  if (!std::isfinite(n)) throw std::runtime_error("Expected a finite scalar."); return n;
}
__device__ static float deviceScalar(const void* p, size_t i, int type) {
  if (type == 0) return static_cast<const float*>(p)[i];
  auto bits = static_cast<const uint16_t*>(p)[i];
  return type == 1 ? __uint_as_float(uint32_t(bits) << 16) : __half2float(reinterpret_cast<const __half*>(p)[i]);
}
__global__ static void embeddingKernel(const void* w, float* out, size_t offset, size_t width, int type, const int32_t* state) {
  if (state) offset = size_t(state[0]) * width;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < width) out[i] = deviceScalar(w, offset + i, type);
}
__global__ static void addKernel(const float* a, const float* b, float* out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __fadd_rn(a[i], b[i]);
}
__global__ static void normKernel(const float* x, const float* w, float* out, size_t n, double eps) {
  __shared__ double scale;
  if (threadIdx.x == 0) {
    double sum = 0; for (size_t i = 0; i < n; i++) sum += double(x[i]) * double(x[i]);
    scale = 1.0 / sqrt(sum / double(n) + eps);
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) out[i] = __fmul_rn(__double2float_rn(double(x[i]) * scale), w[i]);
}
__global__ static void ropeKernel(float* x, size_t n, size_t dim, size_t pos, double theta, const int32_t* state) {
  if (state) pos = size_t(state[1]);
  size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x, half = dim / 2;
  if (pair >= n / 2) return;
  size_t d = pair % half, base = (pair / half) * dim + d;
  double angle = double(pos) / pow(theta, double(2 * d) / double(dim));
  double cs = cos(angle), sn = sin(angle), a = x[base], b = x[base + half];
  x[base] = float(a * cs - b * sn); x[base + half] = float(b * cs + a * sn);
}
__global__ static void swigluKernel(const float* gate, const float* up, float* out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { double g = gate[i]; out[i] = __fmul_rn(float(g / (1.0 + exp(-g))), up[i]); }
}
__global__ static void appendKVKernel(const float* k, const float* v, float* cache, const int32_t* pages,
 size_t layer, size_t position, size_t blockSize, size_t numBlocks, size_t width, const int32_t* state) {
  if (state) position = size_t(state[1]);
  size_t d = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (d >= width) return;
  size_t token = size_t(pages[position / blockSize]) * blockSize + position % blockSize;
  size_t stride = numBlocks * blockSize * width;
  cache[2 * layer * stride + token * width + d] = k[d];
  cache[(2 * layer + 1) * stride + token * width + d] = v[d];
}
__global__ static void attentionKernel(const float* q, const float* cache, float* out, float* scratch,
 const int32_t* pages, size_t layer, size_t position, size_t blockSize, size_t numBlocks,
 size_t heads, size_t kvHeads, size_t dim, const int32_t* state) {
  if (state) position = size_t(state[1]);
  size_t h = blockIdx.x, width = kvHeads * dim, stride = numBlocks * blockSize * width;
  size_t kvHead = h / (heads / kvHeads), n = position + 1;
  float* scores = scratch + h * n;
  for (size_t p = threadIdx.x; p < n; p += blockDim.x) {
    size_t token = size_t(pages[p / blockSize]) * blockSize + p % blockSize;
    const float* key = cache + 2 * layer * stride + token * width + kvHead * dim;
    double dot = 0; for (size_t d = 0; d < dim; d++) dot += double(q[h * dim + d]) * double(key[d]);
    scores[p] = float(dot / sqrt(double(dim)));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double maximum = -INFINITY, total = 0;
    for (size_t p = 0; p < n; p++) maximum = fmax(maximum, double(scores[p]));
    for (size_t p = 0; p < n; p++) { scores[p] = float(exp(double(scores[p]) - maximum)); total += double(scores[p]); }
    for (size_t p = 0; p < n; p++) scores[p] = float(double(scores[p]) / total);
  }
  __syncthreads();
  for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
    double sum = 0;
    for (size_t p = 0; p < n; p++) {
      size_t token = size_t(pages[p / blockSize]) * blockSize + p % blockSize;
      sum += double(scores[p]) * double(cache[(2 * layer + 1) * stride + token * width + kvHead * dim + d]);
    }
    out[h * dim + d] = float(sum);
  }
}
__global__ static void argmaxKernel(const float* x, size_t n, int32_t* out) {
  __shared__ float values[256]; __shared__ int32_t indices[256]; __shared__ int invalid[256];
  unsigned t = threadIdx.x; float best = -INFINITY; int32_t index = INT32_MAX; int bad = 0;
  for (size_t i = t; i < n; i += blockDim.x) {
    float v = x[i]; if (!isfinite(v)) bad = 1;
    if (v > best || (v == best && int32_t(i) < index)) { best = v; index = int32_t(i); }
  }
  values[t] = best; indices[t] = index; invalid[t] = bad; __syncthreads();
  for (unsigned s = 128; s; s /= 2) {
    if (t < s) {
      invalid[t] |= invalid[t + s];
      if (values[t + s] > values[t] || (values[t + s] == values[t] && indices[t + s] < indices[t])) {
        values[t] = values[t + s]; indices[t] = indices[t + s];
      }
    }
    __syncthreads();
  }
  if (!t) *out = invalid[0] ? -1 : indices[0];
}

static napi_value deviceVector(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 4); auto c = unwrap<Context>(e, a[0], contextTag); c->activate(); if (c->capturing) throw std::runtime_error("Operation is not allowed during graph capture.");
    size_t n = integer(e, a[1], SIZE_MAX / 4), reserve = integer(e, a[3]);
    if (!n) throw std::runtime_error("Empty device tensor.");
    size_t freeBytes, total; cu(cudaMemGetInfo(&freeBytes, &total));
    if (freeBytes < reserve || n * 4 > freeBytes - reserve) throw std::runtime_error("Insufficient free CUDA memory for device tensor.");
    bool hasSource; ck(napi_is_typedarray(e, a[2], &hasSource));
    float* source = hasSource ? floats(e, a[2], n) : nullptr;
    auto t = std::make_unique<DeviceTensor>(); t->size = n; t->allocation = std::make_shared<ResidentWeight>(c, c->device);
    t->allocation->storage.ensure(n * 4);
    if (source) { cu(cudaMemcpyAsync(t->allocation->storage.ptr, source, n * 4, cudaMemcpyHostToDevice, c->stream)); cu(cudaStreamSynchronize(c->stream)); c->h2dBytes += n * 4; }
    else cu(cudaMemsetAsync(t->allocation->storage.ptr, 0, n * 4, c->stream));
    c->tensors.push_back(t->allocation); return wrap(e, std::move(t), tensorTag);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value deviceBegin(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 2); auto c = unwrap<Context>(e, a[0], contextTag); c->activate(); if (c->capturing) throw std::runtime_error("Operation is not allowed during graph capture.");
    if (c->tokenActive) throw std::runtime_error("Previous device token has not completed.");
    napi_typedarray_type type; size_t n, offset; void* ptr; napi_value buffer;
    ck(napi_get_typedarray_info(e, a[1], &type, &n, &ptr, &buffer, &offset));
    if (type != napi_int32_array || !n || n > INT32_MAX) throw std::runtime_error("Expected nonempty Int32Array block table.");
    c->blockPages.assign(static_cast<int32_t*>(ptr), static_cast<int32_t*>(ptr) + n);
    c->blockTable.ensure(n * 4);
    cu(cudaMemcpyAsync(c->blockTable.ptr, c->blockPages.data(), n * 4, cudaMemcpyHostToDevice, c->stream));
    c->h2dBytes += n * 4; c->metadataBytes += n * 4;
    cu(cudaEventRecord(c->begin, c->stream)); c->tokenActive = true; return nothing(e);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value deviceFinish(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 1); auto c = unwrap<Context>(e, a[0], contextTag); c->activate(); if (c->capturing) throw std::runtime_error("Operation is not allowed during graph capture.");
    if (c->tokenActive) {
      cu(cudaEventRecord(c->end, c->stream)); cu(cudaEventSynchronize(c->end));
      float elapsed; cu(cudaEventElapsedTime(&elapsed, c->begin, c->end));
      c->deviceForwardMs += elapsed; c->deviceForwards++; c->tokenActive = false;
    } else cu(cudaStreamSynchronize(c->stream));
    return nothing(e);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value deviceEmbedding(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 4); auto c = unwrap<Context>(e, a[0], contextTag); c->activate();
    auto m = unwrap<Matrix>(e, a[1], matrixTag); size_t row = integer(e, a[2]); auto out = tensor(e, a[3], c);
    if (m->closed || !m->resident || m->resident->owner != c || !m->resident->storage.ptr || row >= m->rows || out->size != m->cols) throw std::runtime_error("Invalid device embedding.");
    if (c->capturing) c->capturing->vocab = std::min(c->capturing->vocab, m->rows);
    embeddingKernel<<<grid(m->cols),256,0,c->stream>>>(m->resident->storage.ptr,data(out),row*m->cols,m->cols,m->dtype,c->capturing ? static_cast<const int32_t*>(c->capturing->inputs.ptr) : nullptr);
    cu(cudaGetLastError()); c->deviceOps++; return nothing(e);
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value deviceMatvec(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 4); auto c = unwrap<Context>(e, a[0], contextTag); c->activate();
    auto m = unwrap<Matrix>(e, a[1], matrixTag); auto x = tensor(e,a[2],c), y = tensor(e,a[3],c);
    if (m->closed || !m->resident || m->resident->owner != c || !m->resident->storage.ptr || x->size != m->cols || y->size != m->rows || x == y) throw std::runtime_error("Invalid device matvec shape or storage.");
    if (c->directWeights && m->dtype != 0) {
      directMatvec(c, m->resident->storage.ptr, data(x), data(y), m->rows, m->cols, m->dtype);
      c->tiles++; c->calls++; c->deviceOps++; return nothing(e);
    }
    size_t rows = std::min(m->rows, c->tileBytes / (m->cols * 4));
    if (!rows) throw std::runtime_error("Matrix row exceeds tile capacity.");
    if (m->dtype != 0) c->matrix.ensure(rows * m->cols * 4);
    c->peakScratch = std::max(c->peakScratch, double(c->matrix.capacity));
    float alpha=1,beta=0;
    for (size_t r=0;r<m->rows;r+=rows) {
      size_t n=std::min(rows,m->rows-r), count=n*m->cols;
      const char* source=static_cast<const char*>(m->resident->storage.ptr)+r*m->cols*(m->dtype==0?4:2);
      if(m->dtype!=0) {
        convert<<<grid(count),256,0,c->stream>>>(reinterpret_cast<const uint16_t*>(source),static_cast<float*>(c->matrix.ptr),count,m->dtype);
        cu(cudaGetLastError());
      }
      bl(cublasSgemv(c->handle,CUBLAS_OP_T,int(m->cols),int(n),&alpha,
        m->dtype==0?reinterpret_cast<const float*>(source):static_cast<const float*>(c->matrix.ptr),int(m->cols),data(x),1,&beta,data(y)+r,1));
      c->tiles++;
    }
    c->calls++; c->deviceOps++; return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceAdd(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,4); auto c=unwrap<Context>(e,a[0],contextTag); c->activate();
    auto x=tensor(e,a[1],c),y=tensor(e,a[2],c),out=tensor(e,a[3],c); sameSize(x,y);sameSize(x,out);
    addKernel<<<grid(x->size),256,0,c->stream>>>(data(x),data(y),data(out),x->size);cu(cudaGetLastError());c->deviceOps++;return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceNorm(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,5);auto c=unwrap<Context>(e,a[0],contextTag);c->activate();
    auto x=tensor(e,a[1],c),w=tensor(e,a[2],c),out=tensor(e,a[3],c);sameSize(x,w);sameSize(x,out);
    double eps=scalar(e,a[4]);if(eps<=0)throw std::runtime_error("Invalid RMSNorm epsilon.");
    normKernel<<<1,256,0,c->stream>>>(data(x),data(w),data(out),x->size,eps);cu(cudaGetLastError());c->deviceOps++;return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceRope(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,5);auto c=unwrap<Context>(e,a[0],contextTag);c->activate();auto x=tensor(e,a[1],c);
    size_t dim=integer(e,a[2]),pos=integer(e,a[3]);double theta=scalar(e,a[4]);
    if(!dim||dim%2||x->size%dim||theta<=0)throw std::runtime_error("Invalid RoPE shape.");
    ropeKernel<<<grid(x->size/2),256,0,c->stream>>>(data(x),x->size,dim,pos,theta,c->capturing ? static_cast<const int32_t*>(c->capturing->inputs.ptr) : nullptr);cu(cudaGetLastError());c->deviceOps++;return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceSwiglu(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,4);auto c=unwrap<Context>(e,a[0],contextTag);c->activate();
    auto gate=tensor(e,a[1],c),up=tensor(e,a[2],c),out=tensor(e,a[3],c);sameSize(gate,up);sameSize(gate,out);
    swigluKernel<<<grid(gate->size),256,0,c->stream>>>(data(gate),data(up),data(out),gate->size);cu(cudaGetLastError());c->deviceOps++;return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceAttention(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,13);auto c=unwrap<Context>(e,a[0],contextTag);c->activate();
    auto q=tensor(e,a[1],c),k=tensor(e,a[2],c),v=tensor(e,a[3],c),cache=tensor(e,a[4],c),out=tensor(e,a[5],c),scores=tensor(e,a[6],c);
    size_t layer=integer(e,a[7]),pos=integer(e,a[8]),block=integer(e,a[9],INT32_MAX),pages=integer(e,a[10],INT32_MAX),kvHeads=integer(e,a[11],INT32_MAX),dim=integer(e,a[12],INT32_MAX);
    if(!block||!pages||!kvHeads||!dim||q->size%dim||q->size/dim%kvHeads||k->size!=kvHeads*dim||v->size!=k->size||out->size!=q->size)throw std::runtime_error("Invalid device attention shape.");
    size_t width=kvHeads*dim,heads=q->size/dim;
    if(pages>SIZE_MAX/block/width/2)throw std::runtime_error("KV size overflow.");
    size_t stride=pages*block*width*2;
    if(cache->size%stride||layer>=cache->size/stride||pos>=pages*block||scores->size/heads<pos+1||(!c->tokenActive&&!c->capturing)||c->blockPages.size()!=pos/block+1)throw std::runtime_error("Invalid device KV layout or position.");
    for(auto page:c->blockPages)if(page<0||size_t(page)>=pages)throw std::runtime_error("Invalid physical KV page.");
    const int32_t* state = c->capturing ? static_cast<const int32_t*>(c->capturing->inputs.ptr) : nullptr;
    const int32_t* table = state ? state + 2 : static_cast<const int32_t*>(c->blockTable.ptr);
    appendKVKernel<<<grid(width),256,0,c->stream>>>(data(k),data(v),data(cache),table,layer,pos,block,pages,width,state);
    attentionKernel<<<static_cast<unsigned>(heads),256,0,c->stream>>>(data(q),data(cache),data(out),data(scores),table,layer,pos,block,pages,heads,kvHeads,dim,state);
    cu(cudaGetLastError());c->deviceOps++;return nothing(e);
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceRead(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,2);auto c=unwrap<Context>(e,a[0],contextTag);c->activate(); if (c->capturing) throw std::runtime_error("Operation is not allowed during graph capture.");auto x=tensor(e,a[1],c);
    float* out;auto result=array(e,x->size,&out);cu(cudaMemcpyAsync(out,data(x),x->size*4,cudaMemcpyDeviceToHost,c->stream));cu(cudaStreamSynchronize(c->stream));c->d2hBytes+=x->size*4;return result;
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value deviceArgmax(napi_env e,napi_callback_info info) {
  try {
    auto a=args(e,info,2);auto c=unwrap<Context>(e,a[0],contextTag);c->activate(); if (c->capturing) throw std::runtime_error("Operation is not allowed during graph capture.");auto x=tensor(e,a[1],c);
    if(x->size>INT32_MAX)throw std::runtime_error("Vocabulary exceeds int32.");c->sampledToken.ensure(4);
    argmaxKernel<<<1,256,0,c->stream>>>(data(x),x->size,static_cast<int32_t*>(c->sampledToken.ptr));cu(cudaGetLastError());
    int32_t id;cu(cudaMemcpyAsync(&id,c->sampledToken.ptr,4,cudaMemcpyDeviceToHost,c->stream));cu(cudaStreamSynchronize(c->stream));c->d2hBytes+=4;
    if(id<0)throw std::runtime_error("Non-finite logits.");napi_value result;ck(napi_create_int32(e,id,&result));return result;
  } catch(const std::exception& err) { return fail(e,err); }
}
#include "cuda-graph.inc"
#include "cuda-ipc.inc"

static void registerDeviceOps(napi_env e,napi_value exports) {
  napi_property_descriptor p[]={
    {"ipcExport",nullptr,ipcExport,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"ipcImport",nullptr,ipcImport,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"ipcCopyKV",nullptr,ipcCopyKV,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"graphBegin",nullptr,graphBegin,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"graphEnd",nullptr,graphEnd,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"graphAbort",nullptr,graphAbort,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"graphLaunch",nullptr,graphLaunch,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"graphClose",nullptr,graphClose,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceEnqueueArgmax",nullptr,deviceEnqueueArgmax,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceVector",nullptr,deviceVector,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceBegin",nullptr,deviceBegin,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceFinish",nullptr,deviceFinish,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceEmbedding",nullptr,deviceEmbedding,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceMatvec",nullptr,deviceMatvec,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceAdd",nullptr,deviceAdd,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceNorm",nullptr,deviceNorm,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceRope",nullptr,deviceRope,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceSwiglu",nullptr,deviceSwiglu,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceAttention",nullptr,deviceAttention,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceRead",nullptr,deviceRead,nullptr,nullptr,nullptr,napi_default,nullptr},
    {"deviceArgmax",nullptr,deviceArgmax,nullptr,nullptr,nullptr,napi_default,nullptr}
  };ck(napi_define_properties(e,exports,sizeof(p)/sizeof(p[0]),p));
}
