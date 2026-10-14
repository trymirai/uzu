/**
 * Submission snapshot freshness, failure and lifetime checks in a real browser.
 * UZU_WORKER_SCRIPT=crates/uzu-engine/tests/webgpu-readback.mjs node crates/uzu-engine/tests/webgpu-bridge.mjs
 */
import * as gpu from '/bridge.js';
globalThis.UZU_WEBGPU_PROFILE = true;
const checks = [];
const assert = (value, message) => { if (!value) throw new Error(message); };
const counters = () => JSON.parse(gpu.profileSnapshot(false))[0];
let context;
try {
  await gpu.prepareContext();
  context = JSON.parse(gpu.createContext());
  const id = context.id;
  const submit = (commands, written = []) => gpu.submit(id, JSON.stringify(commands), new Uint32Array(written));
  const wait = submission => gpu.wait(id, submission);
  const fill = (id, value) => ({op:'fill',destination:id,offset:0,size:4,value});
  const read = buffer => {const words = new Uint32Array(1);gpu.readBuffer(id,buffer,new Uint8Array(words.buffer));return words[0];};
  for(const buffer of [10,11,12])gpu.createBuffer(id,buffer,4);
  gpu.uploadBuffer(id,10,new Uint8Array(new Uint32Array([100]).buffer));
  gpu.registerPipeline(id,1,`
    @group(0) @binding(0) var<storage, read> input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    struct Params { data: vec4<u32> }
    @group(0) @binding(2) var<uniform> params: Params;
    @compute @workgroup_size(1) fn main() { output[0]=input[0]+params.data.x; }
  `);
  const dispatch=(input,output)=>({op:'dispatch',pipeline:1,bindings:[input,output],parameters:[1,0,0,0],workgroups:[1,1,1]});
  const first=submit([dispatch(10,11)], [11]);
  const next=submit([dispatch(11,12)], [12]);
  wait(first);
  assert(read(11)===101,'prior submission output snapshot was stale');
  wait(next);
  assert(read(12)===102,'chained submission output snapshot was stale');
  assert(counters().prefetchCopies===2,'read-only use scheduled a redundant snapshot: '+JSON.stringify(counters()));
  assert(counters().prefetchedReads===2,'known snapshots were not used');
  checks.push('one snapshot per writer; read-only chained token preserves previous snapshot');

  const older=submit([fill(11,13)], [11]);
  const newer=submit([fill(11,27)], [11]);
  wait(older);
  assert(read(11)===0x1b1b1b1b,'latest writer snapshot was replaced by earlier completion');
  wait(newer);
  checks.push('latest write wins with two outstanding submissions');

  gpu.uploadBuffer(id,11,new Uint8Array(new Uint32Array([500]).buffer));
  wait(submit([]));
  assert(read(11)===500,'CPU upload did not invalidate cached GPU bytes');
  checks.push('upload invalidates old snapshot');

  let rejected=false;
  try {read(999);}catch {rejected=true;}
  assert(rejected,'unknown readback was not rejected');
  assert(read(12)===102,'unknown readback corrupted known snapshot');
  checks.push('unknown readback error preserves snapshots');

  gpu.registerPipeline(id,99,'@group(0) @binding(0) var<storage, read_write> broken: array<u32>; not valid WGSL');
  rejected=false;
  try {wait(submit([{op:'dispatch',pipeline:99,bindings:[11],parameters:[0,0,0,0],workgroups:[1,1,1]}], [11]));}catch {rejected=true;}
  assert(rejected,'validation failure did not complete submission');
  rejected=false;
  try {read(11);}catch {rejected=true;}
  assert(rejected,'validation failure did not notify prefetched read');
  wait(submit([fill(11,8)], [11]));
  assert(read(11)===0x08080808,'readback did not recover after failed writer');
  checks.push('failed writer wakes cached read; subsequent write recovers');

  gpu.destroyBuffer(id,11);
  gpu.createBuffer(id,13,4);
  wait(submit([]));
  assert(read(13)===0,'recycled physical allocation leaked old snapshot');
  assert(read(12)===102,'physical recycle corrupted another buffer snapshot');
  checks.push('destroy and physical pooling invalidate logical cache');

  // Access flags come from the Rust bindings, never WGSL text. Reordering
  // legal attributes must not retain the result from the previous pipeline.
  gpu.registerPipeline(id,2,`
    @binding(0) @group(0) var<storage, read> input: array<u32>;
    @binding(1) @group(0) var<storage, read_write> output: array<u32>;
    struct Params { data: vec4<u32> }
    @binding(2) @group(0) var<uniform> params: Params;
    @compute @workgroup_size(1) fn main() { output[0]=input[0]+params.data.x+10u; }
  `);
  wait(submit([{...dispatch(10,12),pipeline:2}], [12]));
  assert(read(12)===111,'attribute ordering hid a write behind a stale snapshot');
  checks.push('explicit write IDs support reordered shader attributes');

  // Low-level callers without metadata keep demand-read semantics. Clearing
  // snapshots is essential when an unclassified shader or command writes.
  wait(gpu.submit(id,JSON.stringify([fill(12,9)])));
  const beforeUnknownRead=counters().prefetchedReads;
  assert(read(12)===0x09090909,'unclassified write retained a stale snapshot');
  assert(counters().prefetchedReads===beforeUnknownRead,'unclassified submission retained cached data');
  checks.push('missing access metadata invalidates snapshots');

  const boundedIds=Array.from({length:130},(_,i)=>1000+i);
  for(const buffer of boundedIds)gpu.createBuffer(id,buffer,4);
  wait(submit(boundedIds.map((buffer,i)=>fill(buffer,i)), boundedIds));
  const beforeEvictionRead=counters().prefetchedReads;
  assert(read(boundedIds[0])===0,'evicted snapshot demand-read returned stale data');
  assert(counters().prefetchedReads===beforeEvictionRead,'snapshot cache exceeded its 128-buffer limit');
  assert(read(boundedIds.at(-1))===0x81818181,'retained snapshot was incorrect');
  assert(counters().prefetchedReads===beforeEvictionRead+1,'latest snapshot was not retained');
  for(const buffer of boundedIds)gpu.destroyBuffer(id,buffer);
  checks.push('bounded snapshot cache evicts safely to demand reads');

  gpu.createBuffer(id,2000,68);
  wait(submit([{...fill(2000,42),size:68}], [2000]));
  const beforeLargeRead=counters().prefetchedReads;
  assert(read(2000)===0x2a2a2a2a,'large buffer demand read failed');
  assert(counters().prefetchedReads===beforeLargeRead,'large buffer received an eager snapshot');
  gpu.destroyBuffer(id,2000);
  checks.push('large allocations retain demand readback');

  const stats=counters();
  submit([fill(13,77)], [13]);
  gpu.destroyContext(id);context=undefined;
  rejected=false;
  try {read(13);}catch {rejected=true;}
  assert(rejected,'disposed context retained usable cache');
  checks.push('dispose with pending snapshot rejects further access');
  postMessage({ok:true,checks,stats});
}catch(error){postMessage({ok:false,checks,error:String(error?.stack??error)});}
finally{if(context)gpu.destroyContext(context.id);}
