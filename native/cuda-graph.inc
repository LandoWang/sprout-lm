// Generic capture/replay. The TypeScript Runner supplies the operator sequence.
static const napi_type_tag graphTag = {0x5350524f55544752ULL, 0x4150483030303031ULL};
struct GraphHandle { std::shared_ptr<GraphPlan> plan; };
static GraphPlan* graphPlan(napi_env e, napi_value value, Context* c) {
  auto g = unwrap<GraphHandle>(e, value, graphTag)->plan.get();
  if (g->owner != c || g->closed) throw std::runtime_error("Graph belongs to a different or closed backend.");
  return g;
}
static napi_value graphBegin(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 5); auto c = unwrap<Context>(e, a[0], contextTag); c->activate();
    if (c->tokenActive || c->capturing) throw std::runtime_error("Cannot capture during an active token or graph capture.");
    size_t pages = integer(e, a[1], INT32_MAX / 4 - 2), block = integer(e, a[2], INT32_MAX);
    auto out = tensor(e, a[3], c); bool sample; ck(napi_get_value_bool(e, a[4], &sample));
    if (!pages || !block || pages > size_t(INT32_MAX) / block) throw std::runtime_error("Invalid graph KV capacity.");
    auto g = std::make_shared<GraphPlan>(c, c->device);
    g->started = Clock::now(); g->pages = pages; g->blockSize = block;
    g->output = out->allocation; g->outputSize = out->size; g->sample = sample;
    // Only the conversion baseline needs weight scratch. Direct kernels use registers.
    // Reserve the whole baseline tile so later captures cannot move its address.
    if (!c->directWeights) c->matrix.ensure(c->tileBytes);
    c->sampledToken.ensure(4);
    c->peakScratch = std::max(c->peakScratch, double(c->matrix.capacity));
    g->inputs.ensure((pages + 2) * 4);
    cu(cudaMallocHost(reinterpret_cast<void**>(&g->host), (pages + 2) * 4));
    cu(cudaStreamSynchronize(c->stream));
    g->calls = c->calls; g->tiles = c->tiles; g->ops = c->deviceOps;
    c->graphs.erase(std::remove_if(c->graphs.begin(), c->graphs.end(), [](const auto& ref) { return ref.expired(); }), c->graphs.end());
    c->graphs.push_back(g);
    auto handle = std::make_unique<GraphHandle>(); handle->plan = g;
    auto result = wrap(e, std::move(handle), graphTag);
    cu(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeThreadLocal));
    c->capturing = g.get(); return result;
  } catch (const std::exception& err) { return fail(e, err); }
}
static napi_value graphEnd(napi_env e, napi_callback_info info) {
  cudaGraph_t captured = nullptr;
  try {
    auto a = args(e, info, 2); auto c = unwrap<Context>(e, a[0], contextTag); c->activate();
    auto g = graphPlan(e, a[1], c);
    if (c->capturing != g) throw std::runtime_error("Graph is not being captured.");
    if (g->sample && !g->sampled) throw std::runtime_error("Sampling graph needs an argmax operation.");
    auto status = cudaStreamEndCapture(c->stream, &captured); c->capturing = nullptr;
    double calls = c->calls - g->calls, tiles = c->tiles - g->tiles, ops = c->deviceOps - g->ops;
    c->calls = g->calls; c->tiles = g->tiles; c->deviceOps = g->ops;
    g->calls = calls; g->tiles = tiles; g->ops = ops;
    cu(status);
    size_t nodes = 0; cu(cudaGraphGetNodes(captured, nullptr, &nodes));
    cu(cudaGraphInstantiateWithFlags(&g->exec, captured, 0));
    cu(cudaGraphUpload(g->exec, c->stream)); cu(cudaStreamSynchronize(c->stream));
    cu(cudaGraphDestroy(captured)); captured = nullptr;
    c->graphCaptures++; c->graphNodes += nodes; c->graphCaptureMs += ms(g->started);
    return nothing(e);
  } catch (const std::exception& err) {
    if (captured) cudaGraphDestroy(captured);
    return fail(e, err);
  }
}
static napi_value graphAbort(napi_env e, napi_callback_info info) {
  try { auto a = args(e, info, 1); auto c = unwrap<Context>(e,a[0],contextTag); c->activate(); c->abortCapture(); return nothing(e); }
  catch (const std::exception& err) { return fail(e,err); }
}
static napi_value graphClose(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e, info, 2); auto c = unwrap<Context>(e,a[0],contextTag); c->activate();
    auto g = graphPlan(e,a[1],c);
    if (c->capturing == g) c->abortCapture(); else g->close();
    return nothing(e);
  } catch (const std::exception& err) { return fail(e,err); }
}
static napi_value deviceEnqueueArgmax(napi_env e, napi_callback_info info) {
  try {
    auto a = args(e,info,2); auto c = unwrap<Context>(e,a[0],contextTag); c->activate(); auto x = tensor(e,a[1],c);
    if (!c->capturing || x->size > INT32_MAX) throw std::runtime_error("Expected active graph capture and int32 vocabulary.");
    if (!c->capturing->sample || c->capturing->output != x->allocation || c->capturing->sampled) throw std::runtime_error("Graph argmax must sample its output exactly once.");
    c->capturing->sampled = true;
    argmaxKernel<<<1,256,0,c->stream>>>(data(x),x->size,static_cast<int32_t*>(c->sampledToken.ptr));
    cu(cudaGetLastError()); return nothing(e);
  } catch (const std::exception& err) { return fail(e,err); }
}
static napi_value graphLaunch(napi_env e, napi_callback_info info) {
  Context* c = nullptr;
  try {
    auto a = args(e,info,5); c = unwrap<Context>(e,a[0],contextTag); c->activate(); auto g = graphPlan(e,a[1],c);
    if (!g->exec || c->capturing || c->tokenActive) throw std::runtime_error("Graph is not executable or backend is busy.");
    size_t token = integer(e,a[3],INT32_MAX), position = integer(e,a[4],INT32_MAX);
    if (token >= g->vocab) throw std::runtime_error("Invalid graph token.");
    napi_typedarray_type type; size_t n, offset; void* ptr; napi_value buffer;
    ck(napi_get_typedarray_info(e,a[2],&type,&n,&ptr,&buffer,&offset));
    if (type != napi_int32_array || !n || n > g->pages || position >= g->pages * g->blockSize || n != position / g->blockSize + 1) throw std::runtime_error("Invalid graph block table or position.");
    auto pages = static_cast<int32_t*>(ptr);
    for (size_t i=0; i<n; i++) if (pages[i]<0 || size_t(pages[i])>=g->pages) throw std::runtime_error("Invalid graph physical page.");
    g->host[0] = int32_t(token); g->host[1] = int32_t(position);
    std::memcpy(g->host+2,pages,n*4);
    cu(cudaMemcpyAsync(g->inputs.ptr,g->host,(n+2)*4,cudaMemcpyHostToDevice,c->stream));
    c->h2dBytes += (n+2)*4; c->metadataBytes += n*4; c->graphInputBytes += 8;
    cu(cudaEventRecord(c->begin,c->stream));
    cu(cudaGraphLaunch(g->exec,c->stream)); c->graphLaunches++;
    c->calls += g->calls; c->tiles += g->tiles; c->deviceOps += g->ops;
    napi_value result; float* logits = nullptr;
    if (g->sample) {
      // Input reads have completed before this same-stream copy overwrites host[0].
      cu(cudaMemcpyAsync(g->host,c->sampledToken.ptr,4,cudaMemcpyDeviceToHost,c->stream)); c->d2hBytes += 4;
    } else {
      result = array(e,g->outputSize,&logits);
      cu(cudaMemcpyAsync(logits,g->output->storage.ptr,g->outputSize*4,cudaMemcpyDeviceToHost,c->stream)); c->d2hBytes += g->outputSize*4;
    }
    cu(cudaEventRecord(c->end,c->stream)); cu(cudaEventSynchronize(c->end));
    float elapsed; cu(cudaEventElapsedTime(&elapsed,c->begin,c->end)); c->deviceForwardMs += elapsed; c->deviceForwards++;
    if (g->sample) { if(g->host[0]<0)throw std::runtime_error("Non-finite logits."); ck(napi_create_int32(e,g->host[0],&result)); }
    return result;
  } catch (const std::exception& err) {
    // Preserve metadata/host buffer lifetimes even if a launch or readback failed.
    if (c && !c->closed && !c->capturing) cudaStreamSynchronize(c->stream);
    return fail(e,err);
  }
}
