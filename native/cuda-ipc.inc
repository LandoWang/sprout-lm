// Same-host cross-process GPU KV exchange. The coordinator owns page lifetimes.
static const napi_type_tag ipcTag = {0x5350524f55544950ULL, 0x434d454d4f525931ULL};
struct ImportHandle { std::shared_ptr<ImportedMemory> memory; };
static void ipcIdle(Context* c) {
  c->activate();
  if (c->capturing || c->tokenActive) throw std::runtime_error("IPC operation requires an idle backend.");
}
static napi_value ipcExport(napi_env e, napi_callback_info info) {
  try {
    auto a=args(e,info,2); auto c=unwrap<Context>(e,a[0],contextTag); ipcIdle(c);
    auto t=tensor(e,a[1],c); cu(cudaStreamSynchronize(c->stream));
    cudaIpcMemHandle_t handle; cu(cudaIpcGetMemHandle(&handle,data(t)));
    napi_value out; ck(napi_create_buffer_copy(e,sizeof(handle),&handle,nullptr,&out)); return out;
  } catch(const std::exception& err) { return fail(e,err); }
}
static napi_value ipcImport(napi_env e, napi_callback_info info) {
  try {
    auto a=args(e,info,4); auto c=unwrap<Context>(e,a[0],contextTag); ipcIdle(c);
    bool isBuffer; ck(napi_is_buffer(e,a[1],&isBuffer));
    if(!isBuffer) throw std::runtime_error("Expected CUDA IPC handle buffer.");
    void* ptr; size_t bytes; ck(napi_get_buffer_info(e,a[1],&ptr,&bytes));
    if(bytes!=sizeof(cudaIpcMemHandle_t)) throw std::runtime_error("Invalid CUDA IPC handle size.");
    int source=int(integer(e,a[2],INT32_MAX)), count; cu(cudaGetDeviceCount(&count));
    size_t elements=integer(e,a[3],SIZE_MAX/4);
    if(source>=count || source==c->device || !elements) throw std::runtime_error("IPC requires distinct valid GPUs and a nonempty allocation.");
    int peer=0; cu(cudaDeviceCanAccessPeer(&peer,c->device,source));
    if(!peer) throw std::runtime_error("CUDA P2P is unavailable; no host-staging fallback.");
    cudaIpcMemHandle_t handle; std::memcpy(&handle,ptr,sizeof(handle));
    auto memory=std::make_shared<ImportedMemory>(c,c->device,elements);
    cu(cudaIpcOpenMemHandle(&memory->ptr,handle,cudaIpcMemLazyEnablePeerAccess));
    c->imports.push_back(memory);
    auto out=std::make_unique<ImportHandle>(); out->memory=memory; return wrap(e,std::move(out),ipcTag);
  } catch(const std::exception& err) { return fail(e,err); }
}
static std::vector<int32_t> ipcPages(napi_env e,napi_value value,size_t expected,size_t limit) {
  napi_typedarray_type type; size_t n,offset; void* ptr; napi_value buffer;
  ck(napi_get_typedarray_info(e,value,&type,&n,&ptr,&buffer,&offset));
  if(type!=napi_int32_array || n!=expected) throw std::runtime_error("Invalid IPC page table length.");
  auto p=static_cast<int32_t*>(ptr); std::vector<int32_t> result(p,p+n), sorted=result;
  for(auto page:result) if(page<0 || size_t(page)>=limit) throw std::runtime_error("Invalid IPC physical page.");
  std::sort(sorted.begin(),sorted.end());
  if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()) throw std::runtime_error("Duplicate IPC physical page.");
  return result;
}
static napi_value ipcCopyKV(napi_env e,napi_callback_info info) {
  Context* c=nullptr;
  try {
    auto a=args(e,info,9); c=unwrap<Context>(e,a[0],contextTag); ipcIdle(c);
    auto source=unwrap<ImportHandle>(e,a[1],ipcTag)->memory;
    if(source->owner!=c || !source->ptr) throw std::runtime_error("IPC memory belongs to a different or closed backend.");
    auto target=tensor(e,a[2],c);
    size_t tokens=integer(e,a[5],INT32_MAX), block=integer(e,a[6],INT32_MAX), layers=integer(e,a[7],INT32_MAX), width=integer(e,a[8],INT32_MAX);
    if(!tokens || !block || !layers || !width || block>SIZE_MAX/4/width/2/layers) throw std::runtime_error("Invalid IPC KV layout.");
    size_t pageElements=block*width, planes=2*layers, unit=planes*pageElements;
    if(source->elements%unit || target->size%unit) throw std::runtime_error("IPC allocation/layout mismatch.");
    size_t srcCount=source->elements/unit,dstCount=target->size/unit,n=(tokens+block-1)/block;
    auto src=ipcPages(e,a[3],n,srcCount), dst=ipcPages(e,a[4],n,dstCount);
    size_t srcPitch=srcCount*pageElements*4,dstPitch=dstCount*pageElements*4;
    cu(cudaEventRecord(c->begin,c->stream));
    for(size_t i=0;i<n;i++) {
      size_t count=std::min(block,tokens-i*block)*width;
      // Each pitch steps to the next layer's K or V plane. Never copy tail garbage.
      cu(cudaMemcpy2DAsync(data(target)+size_t(dst[i])*pageElements,dstPitch,
        static_cast<const float*>(source->ptr)+size_t(src[i])*pageElements,srcPitch,
        count*4,planes,cudaMemcpyDeviceToDevice,c->stream));
    }
    cu(cudaEventRecord(c->end,c->stream)); cu(cudaEventSynchronize(c->end));
    float elapsed; cu(cudaEventElapsedTime(&elapsed,c->begin,c->end));
    size_t bytes=tokens*width*planes*4;
    c->ipcCopyCalls++; c->ipcCopyBytes+=bytes; c->ipcCopyMs+=elapsed;
    napi_value result; ck(napi_create_object(e,&result)); number(e,result,"bytes",bytes); number(e,result,"eventMs",elapsed); return result;
  } catch(const std::exception& err) {
    // Do not let an error return while a partial copy can still touch either pool.
    if(c && !c->closed && !c->capturing) cudaStreamSynchronize(c->stream);
    return fail(e,err);
  }
}
