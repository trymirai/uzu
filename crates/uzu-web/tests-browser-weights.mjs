import init, * as wasm from '/wasm/uzu_web.js';
try {
  await init();
  if (!wasm.supportsBlobWeights()) throw new Error('Blob weights capability missing');
  postMessage({ok:true,...JSON.parse(await wasm.testWebGpuWeightLoading())});
}catch(error){postMessage({ok:false,error:String(error?.stack??error)});}
