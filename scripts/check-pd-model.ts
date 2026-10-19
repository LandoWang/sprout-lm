import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname, resolve } from 'node:path';
import { PDWorkers } from '../src/pd-workers.ts';
import { argmax } from '../src/runner.ts';

const [directory,referenceDir]=process.argv.slice(2);
if(!directory||!referenceDir)throw new Error('Usage: node scripts/check-pd-model.ts MODEL_DIR REFERENCE_DIR');
const reference=JSON.parse(await readFile(join(referenceDir,'reference.json'),'utf8'));
const started=performance.now();
const workers=await PDWorkers.start({directory,tokens:256,prefillDevice:Number(process.env.PREFILL_DEVICE??0),
  decodeDevice:Number(process.env.DECODE_DEVICE??1),cuda:{graph:true,reserveMiB:4096}});
try {
  const spec=workers.prefill.spec,layout=workers.prefill.layout;
  if(reference.modelId)assert.equal(spec.id,reference.modelId);
  let maxError=0,maxToleranceRatio=0,positions=0;
  const results=[];
  const source=Array.from({length:layout.numBlocks},(_,i)=>layout.numBlocks-1-i);
  const target=Array.from({length:layout.numBlocks},(_,i)=>(i+5)%layout.numBlocks);
  for(const [index,sample] of reference.cases.entries()) {
    const request=`reference-${index}`,count=sample.tokens.length,n=Math.ceil(count/layout.blockSize);
    let output=await workers.forward('prefill',{request,from:0,tokens:sample.tokens,blocks:source.slice(0,n),output:'logits'});
    const transfer=await workers.transfer({request,sourcePages:source.slice(0,n),targetPages:target.slice(0,n),tokens:count});
    const tokens=[];
    for(let step=0;step<sample.output.length;step++) {
      const expected=await readFile(join(referenceDir,sample.logits[step]));assert.equal(expected.length,spec.vocabSize*4);
      assert.equal(output.logits!.length,spec.vocabSize);
      for(let i=0;i<spec.vocabSize;i++) {
        const value=expected.readFloatLE(i*4),error=Math.abs(output.logits![i]-value),tolerance=1e-3+Math.abs(value)*1e-4;
        maxError=Math.max(maxError,error);maxToleranceRatio=Math.max(maxToleranceRatio,error/tolerance);
        assert.ok(error<=tolerance,`case ${index}, step ${step}, logit ${i}: ${output.logits![i]} vs ${value}`);
      }
      const token=argmax(Float32Array.from(output.logits!));assert.equal(token,sample.output[step]);tokens.push(token);positions++;
      if(step+1<sample.output.length) {
        const from=count+step;
        output=await workers.forward('decode',{request,from,tokens:[token],blocks:target.slice(0,Math.ceil((from+1)/layout.blockSize)),output:'logits'});
      }
    }
    results.push({input:sample.tokens,output:tokens,transfer});
  }
  const report={model:spec.id,coordinatorPid:process.pid,prefillPid:workers.prefill.pid,decodePid:workers.decode.pid,
    reference:{torch:reference.torch,transformers:reference.transformers,dtype:reference.dtype},positions,logitsPerPosition:spec.vocabSize,
    maxError,maxToleranceRatio,tolerance:'1e-3 + abs(reference) * 1e-4',results,durationMs:performance.now()-started,stats:await workers.stats()};
  if(process.env.REPORT_PATH){const path=resolve(process.env.REPORT_PATH);await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report));
} finally {await workers.close();}
