/** Bounded Blob/byte weight initialization and failure recovery in real WebGPU. */
import * as gpu from '/bridge.js';
globalThis.UZU_WEBGPU_PROFILE = true;
const checks = [];
const assert = (value, message) => { if (!value) throw new Error(message); };
const rejects = (operation, message) => { let rejected=false;try {operation();}catch {rejected=true;}assert(rejected,message); };
const equal = (actual, expected, message) => {
  assert(actual.length===expected.length,message+' length');
  assert(actual.every((value,index)=>value===expected[index]),message);
};
let context;
try {
  await gpu.prepareContext();context=JSON.parse(gpu.createContext());const id=context.id;
  const read=(buffer,size)=>{const data=new Uint8Array(size);gpu.readBuffer(id,buffer,data);return data;};
  const bytes=Uint8Array.from({length:9*1024*1024+3},(_,index)=>(index*13+7)&255);
  const blob=new Blob([bytes.subarray(0,17),bytes.subarray(17,4*1024*1024+1),bytes.subarray(4*1024*1024+1)]);
  gpu.initializeBufferFromSource(id,100,blob,0,bytes.length);
  equal(read(100,bytes.length),bytes,'multi-chunk Blob upload');
  gpu.wait(id,gpu.submit(id,'[]',new Uint32Array()));
  equal(read(100,bytes.length),bytes,'first normal submission recreated weights');
  checks.push('composite Blob: bounded chunks survive first ordinary submit');
  const counters=JSON.parse(gpu.profileSnapshot(false))[0];
  assert(counters.streamUploadChunks===3 && counters.streamMaxChunkBytes===4*1024*1024,'upload chunk budget exceeded');
  checks.push('one acknowledged 4 MiB upload at a time');

  gpu.initializeBufferFromSource(id,101,bytes,5,19);
  equal(read(101,19),bytes.subarray(5,24),'Uint8Array range upload');
  assert(bytes.byteLength===9*1024*1024+3 && bytes[0]===7,'caller ArrayBuffer detached');
  checks.push('typed-array source remains owned by caller');
  for(const length of [1,2,3,5,6,7]){
    const buffer=200+length;
    gpu.initializeBufferFromSource(id,buffer,blob,3,length);
    const expected=new Uint8Array(Math.ceil(length/4)*4);expected.set(bytes.subarray(3,3+length));
    equal(read(buffer,expected.length),expected,'final word zero padding '+length);
    gpu.destroyBuffer(id,buffer);
  }
  checks.push('unaligned file offsets and partial final words preserve bytes and zero padding');
  const header=new Uint8Array(7);gpu.readWeightSource(blob,4*1024*1024-3,header);
  equal(header,bytes.subarray(4*1024*1024-3,4*1024*1024+4),'cross-chunk source read');
  checks.push('header/range reads span cached Blob parts');

  for(const source of [new Uint8Array(),new Blob(),{},null]) rejects(()=>gpu.weightSourceSize(source),'invalid source accepted');
  for(const [offset,length] of [[-1,4],[0,0],[0,bytes.length+1],[Number.MAX_SAFE_INTEGER,1],[0.5,4]]){
    rejects(()=>gpu.initializeBufferFromSource(id,300,blob,offset,length),'invalid source range accepted');
  }
  rejects(()=>gpu.readWeightSource(blob,bytes.length-1,new Uint8Array(2)),'truncated CPU source read accepted');
  checks.push('source type, zero size, safe integers and exact bounds validated');

  class TruncatedBlob extends Blob {
    slice(start,end){return start>=4*1024*1024 ? new Blob() : super.slice(start,end);}
  }
  rejects(()=>gpu.initializeBufferFromSource(id,301,new TruncatedBlob([blob]),0,bytes.length),'short later chunk accepted');
  rejects(()=>read(301,4),'partially initialized buffer escaped');
  gpu.initializeBufferFromSource(id,301,new Blob([new Uint8Array([1,2,3,4])]),0,4);
  equal(read(301,4),new Uint8Array([1,2,3,4]),'retry after partial source failure');
  checks.push('truncated later chunk destroys partial allocation and permits retry');

  class TooLargeBlob extends Blob {get size(){return context.maxBufferSize+4;}}
  rejects(()=>gpu.initializeBufferFromSource(id,302,new TooLargeBlob(),0,context.maxBufferSize+4),'oversized GPU allocation accepted');
  gpu.initializeBufferFromSource(id,302,new Uint8Array([8,9,10]),0,3);
  equal(read(302,4),new Uint8Array([8,9,10,0]),'retry after GPU validation failure');
  checks.push('GPU allocation validation unwinds scopes and recovers');

  for (const flag of [new Int32Array(1), new Uint8Array(new SharedArrayBuffer(4)), new Int32Array(new SharedArrayBuffer(0)), {}]) {
    rejects(()=>gpu.initializeBufferFromSource(id,310,blob,0,4,flag),'invalid cancellation flag accepted');
  }
  const cancellation=new Int32Array(new SharedArrayBuffer(8));
  Atomics.store(cancellation,0,1);
  rejects(()=>gpu.initializeBufferFromSource(id,310,blob,0,4,cancellation),'pre-cancelled load allocated a buffer');
  rejects(()=>read(310,4),'pre-cancelled buffer escaped');
  checks.push('shared cancellation flag validated before GPU allocation');

  Atomics.store(cancellation,0,0);
  const setterUrl=URL.createObjectURL(new Blob(['self.onmessage=({data})=>{postMessage(\"ready\");for(;;){while(Atomics.load(data,1)===0)Atomics.wait(data,1,0);Atomics.store(data,1,0);Atomics.store(data,0,1);Atomics.notify(data,0);}};'],{type:'text/javascript'}));
  const setter=new Worker(setterUrl,{type:"module"});
  await new Promise((resolve,reject)=>{setter.onmessage=resolve;setter.onerror=reject;setter.postMessage(cancellation);});
  URL.revokeObjectURL(setterUrl);
  class CancelledBlob extends Blob {
    slice(start,end){
      if(start>=4*1024*1024){Atomics.store(cancellation,1,1);Atomics.notify(cancellation,1);Atomics.wait(cancellation,0,0,5000);}
      return super.slice(start,end);
    }
  }
  const beforeCancel=JSON.parse(gpu.profileSnapshot(false))[0].streamUploadChunks;
  let cancelled=false,cancelError;
  try {gpu.initializeBufferFromSource(id,311,new CancelledBlob([blob]),0,bytes.length,cancellation);}
  catch(error){cancelError=String(error);cancelled=cancelError.includes('Model loading cancelled');}
  assert(cancelled,'external flag did not cancel a synchronous partial upload: '+JSON.stringify({cancelError,flag:Atomics.load(cancellation,0)}));
  assert(JSON.parse(gpu.profileSnapshot(false))[0].streamUploadChunks===beforeCancel+1,'cancellation uploaded another chunk');
  rejects(()=>read(311,4),'cancelled partial buffer escaped');
  checks.push('another worker cancels after one acknowledged chunk and partial buffers are destroyed');
  Atomics.store(cancellation,0,0);
  gpu.initializeBufferFromSource(id,311,new Uint8Array([11,12,13]),0,3,cancellation);
  equal(read(311,4),new Uint8Array([11,12,13,0]),'retry after cancellation');
  checks.push('reset cancellation flag permits fresh initialization without stale data');
  Atomics.store(cancellation,0,0);
  cancelled=false;
  try {gpu.readWeightSource(new CancelledBlob([blob]),0,new Uint8Array(4*1024*1024+4),cancellation);}
  catch(error){cancelled=String(error).includes('Model loading cancelled');}
  setter.terminate();
  assert(cancelled,'CPU/header reads ignored a shared cancellation flag');
  checks.push('bounded CPU source reads observe cancellation between chunks');

  gpu.destroyBuffer(id,301);gpu.createBuffer(id,303,4);gpu.wait(id,gpu.submit(id,'[]',new Uint32Array()));
  equal(read(303,4),new Uint8Array(4),'recycled weights leaked previous contents');
  checks.push('destroy and ordinary buffer reuse preserve initialization');
  const stats=JSON.parse(gpu.profileSnapshot(false))[0];
  const limits={maxBufferSize:context.maxBufferSize,maxStorageBufferBindingSize:context.maxStorageBufferBindingSize};
  gpu.destroyContext(id);context=undefined;
  rejects(()=>gpu.initializeBufferFromSource(id,400,blob,0,4),'disposed context accepted upload');
  checks.push('disposed context rejects source initialization');
  postMessage({ok:true,checks,stats,limits});
}catch(error){postMessage({ok:false,checks,error:String(error?.stack??error)});}
finally{if(context)gpu.destroyContext(context.id);}
