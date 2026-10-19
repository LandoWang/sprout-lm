// W is stored in checkpoint dtype; x, accumulators and y stay FP32.
// Conversion happens in registers, never through an expanded FP32 weight buffer.
template<int Type> __device__ static float weight16(uint16_t bits) {
  if constexpr (Type == 1) return __uint_as_float(uint32_t(bits) << 16);
  else return __half2float(__ushort_as_half(bits));
}

// Four independent rows per CTA, one warp per row. Aligned rows use packed
// 64-bit weight loads and 128-bit activation loads. Odd widths use scalar loads.
template<int Type, bool Packed> __global__ static void directMatvecKernel(
 const uint16_t* weights, const float* x, float* y, size_t rows, size_t cols) {
  size_t row = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
  if (row >= rows) return; // The whole warp exits together.
  unsigned lane = threadIdx.x % 32;
  const uint16_t* w = weights + row * cols;
  float a = 0, b = 0, c = 0, d = 0;
  if constexpr (Packed) {
    for (size_t j = size_t(lane) * 4; j < cols; j += 128) {
      uint2 v = *reinterpret_cast<const uint2*>(w + j);
      float4 input = *reinterpret_cast<const float4*>(x + j);
      a = __fmaf_rn(weight16<Type>(uint16_t(v.x)), input.x, a);
      b = __fmaf_rn(weight16<Type>(uint16_t(v.x >> 16)), input.y, b);
      c = __fmaf_rn(weight16<Type>(uint16_t(v.y)), input.z, c);
      d = __fmaf_rn(weight16<Type>(uint16_t(v.y >> 16)), input.w, d);
    }
  } else {
    for (size_t j = lane; j < cols; j += 32)
      a = __fmaf_rn(weight16<Type>(w[j]), x[j], a);
  }
  float sum = (a + b) + (c + d);
  for (unsigned offset = 16; offset; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (!lane) y[row] = sum;
}

static void directMatvec(Context* c, const void* weights, const float* x, float* y,
 size_t rows, size_t cols, int dtype) {
  unsigned blocks = static_cast<unsigned>((rows + 3) / 4);
  if (dtype == 1) {
    if (cols % 4 == 0) directMatvecKernel<1,true><<<blocks,128,0,c->stream>>>(static_cast<const uint16_t*>(weights),x,y,rows,cols);
    else directMatvecKernel<1,false><<<blocks,128,0,c->stream>>>(static_cast<const uint16_t*>(weights),x,y,rows,cols);
  } else {
    if (cols % 4 == 0) directMatvecKernel<2,true><<<blocks,128,0,c->stream>>>(static_cast<const uint16_t*>(weights),x,y,rows,cols);
    else directMatvecKernel<2,false><<<blocks,128,0,c->stream>>>(static_cast<const uint16_t*>(weights),x,y,rows,cols);
  }
  cu(cudaGetLastError());
}
