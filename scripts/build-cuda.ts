import { mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
if (process.platform !== 'linux') throw new Error('The experimental CUDA backend currently builds on Linux.');
const cuda = process.env.CUDA_PATH ?? '/usr/local/cuda';
await mkdir('build', { recursive: true });
const result = spawnSync(resolve(cuda, 'bin/nvcc'), ['-O3', '-std=c++17', '-shared', '-Xcompiler=-fPIC',
  '-arch=sm_80', '-DNAPI_VERSION=8', '-DNODE_GYP_MODULE_NAME=sprout_cuda',
  '-I' + (process.env.NODE_INCLUDE ?? '/usr/include/node'), 'native/cuda-offload.cu',
  '-lcublas', '-lcudart', '-Xlinker', '-rpath', '-Xlinker', resolve(cuda, 'lib64'), '-o', 'build/sprout_cuda.node'], { stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
