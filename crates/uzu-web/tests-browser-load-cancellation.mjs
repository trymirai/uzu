// Real model loading through a composite Blob, with cancellation written by
// another worker while the engine worker is blocked inside createSession.
import init, * as wasm from '/wasm/uzu_web.js';
globalThis.UZU_WEBGPU_PROFILE = true;
const checks=[];
let session, setter;
const assert=(value,message)=>{if(!value)throw new Error(message);};
try {
  await init();
  const read=async name=>new Uint8Array(await(await fetch('/model/'+name)).arrayBuffer());
  const config=await read('config.json'),tokenizer=await read('tokenizer.json');
  const downloaded=await(await fetch('/model/model.safetensors')).blob();
  const parts=[];
  for(let offset=0;offset<downloaded.size;offset+=4*1024*1024)parts.push(downloaded.slice(offset,offset+4*1024*1024));
  const weights=new Blob(parts);
  const flag=new Int32Array(new SharedArrayBuffer(8));
  for(const invalid of [new Int32Array(1),new Uint8Array(new SharedArrayBuffer(4)),new Int32Array(new SharedArrayBuffer(0))]){
    let rejected=false;
    try{await wasm.createSession({config,tokenizer,weights,loadCancellation:invalid});}
    catch(error){rejected=String(error).includes('loadCancellation must be');}
    assert(rejected,'Invalid load cancellation flag accepted');
  }
  checks.push('invalid cancellation flags rejected before GPU allocation');
  Atomics.store(flag,0,1);
  let cancelled=false;
  try{await wasm.createSession({config,tokenizer,weights,loadCancellation:flag});}
  catch(error){cancelled=String(error).includes('Model loading cancelled');}
  assert(cancelled&&JSON.parse(wasm.webGpuProfile(false)).length===0,'Pre-cancelled load allocated a context');
  checks.push('pre-cancelled model load creates no GPU context');
  Atomics.store(flag,0,0);
  const url=URL.createObjectURL(new Blob(['self.onmessage=({data})=>{postMessage(\"ready\");for(;;){while(Atomics.load(data,1)===0)Atomics.wait(data,1,0);Atomics.store(data,1,0);Atomics.store(data,0,1);Atomics.notify(data,0);}};'],{type:'text/javascript'}));
  setter=new Worker(url,{type:"module"});
  await new Promise((resolve,reject)=>{setter.onmessage=resolve;setter.onerror=reject;setter.postMessage(flag);});
  URL.revokeObjectURL(url);
  let reads=0,uploadedChunks=0;
  class CancelledBlob extends Blob {
    slice(start,end){
      reads++;
      if(reads===6){
        uploadedChunks=JSON.parse(wasm.webGpuProfile(false))[0]?.streamUploadChunks??0;
        Atomics.store(flag,1,1);Atomics.notify(flag,1);Atomics.wait(flag,0,0,5000);
      }
      return super.slice(start,end);
    }
  }
  cancelled=false;
  try{await wasm.createSession({config,tokenizer,weights:new CancelledBlob(parts),loadCancellation:flag});}
  catch(error){cancelled=String(error).includes('Model loading cancelled');}
  assert(cancelled&&reads===6&&uploadedChunks>0,'Mid-load cancellation failed: '+JSON.stringify({cancelled,reads,uploadedChunks}));
  assert(JSON.parse(wasm.webGpuProfile(false)).length===0,'Cancelled model retained a context or partial buffers');
  checks.push({name:'external flag cancels synchronous model load and releases partial context',reads,uploadedChunks});
  Atomics.store(flag,0,0);
  session=await wasm.createSession({config,tokenizer,weights,loadCancellation:flag,seed:42,encoding:{type:'hanashi',name:'qwen3.5'}});
  assert(session.debugContextLength()===16384,'Omitted context did not use model recommendation: '+session.debugContextLength());
  checks.push('omitted context uses the engine recommendation of 16384 tokens');
  let done;
  await session.run({messages:[{role:'user',content:'Say hello in one sentence.'}],reasoningEffort:'disabled',samplingPolicy:{type:'Greedy'}},event=>{if(event.type==='done')done=event;});
  assert(done?.finishReason==='Stop'&&done.text.trim(),'Fresh load after cancellation did not finish naturally: '+JSON.stringify(done));
  checks.push({name:'fresh load recovers and omitted output limit stops at model EOS',text:done.text,tokens:done.stats.totalStats.tokensCountOutput});
  await session.run({messages:[{role:'user',content:'Read this repeated text and say ready. '+('The river flows through the valley. '.repeat(360))}],reasoningEffort:'disabled',samplingPolicy:{type:'Greedy'},maxOutputTokens:2},event=>{if(event.type==='done')done=event;});
  assert(done?.stats.totalStats.tokensCountInput>2048&&done.stats.totalStats.tokensCountOutput===2,'Large recommended context did not prefill beyond the old cap: '+JSON.stringify(done));
  checks.push({name:'actual prompt above 2048 tokens prefills through multiple batches',inputTokens:done.stats.totalStats.tokensCountInput,outputTokens:done.stats.totalStats.tokensCountOutput});
  session.dispose();session=undefined;
  assert(JSON.parse(wasm.webGpuProfile(false)).length===0,'Disposed recovered session retained a context');
  checks.push('recovered session disposal releases its GPU context');
  postMessage({ok:true,checks});
}catch(error){postMessage({ok:false,checks,error:String(error?.stack??error)});}
finally{session?.dispose();setter?.terminate();}
