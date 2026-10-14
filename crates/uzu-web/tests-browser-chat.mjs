// Served by uzu-engine/tests/webgpu-bridge.mjs via UZU_WORKER_SCRIPT. The model
// must be Qwen3.5 (any supported size); every reply is actual local inference.
import init, * as wasm from '/wasm/uzu_web.js';
const checks = [];
let session;
function assert(ok, message) { if (!ok) throw new Error(message); }
try {
  await init();
  const encoding = { type: 'hanashi', name: 'qwen3.5' };
  const support = wasm.inspectEncoding(encoding);
  assert(support.supportsTools && support.reasoning.kind === 'toggle', 'Incorrect encoding capabilities');
  checks.push({name:'desktop encoding capabilities',support});
  const read = async name => new Uint8Array(await (await fetch('/model/'+name)).arrayBuffer());
  const configBytes=await read('config.json');
  wasm.inspectModelConfig(configBytes);
  const invalid=JSON.parse(new TextDecoder().decode(configBytes));
  invalid.decoder_config.transformer_config.layer_configs[0].mixer_config.head_dim=512;
  let rejected=false;
  try {wasm.inspectModelConfig(new TextEncoder().encode(JSON.stringify(invalid)));}catch(error){rejected=String(error).includes('head dimensions');}
  assert(rejected,'Unsupported head dimension was not rejected before weights loaded');
  checks.push('model architecture preflight without GPU allocation');
  for (const seed of [-1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1, '42']) {
    rejected = false;
    try { await wasm.createSession({config:configBytes,tokenizer:new Uint8Array([1]),weights:new Uint8Array([1]),seed}); }
    catch(error) { rejected=String(error).includes('seed must be a nonnegative safe integer'); }
    assert(rejected, 'Invalid sampling seed was not rejected before model loading: '+String(seed));
  }
  checks.push('sampling seed validation before GPU allocation');
  session = await wasm.createSession({config:configBytes,tokenizer:await read('tokenizer.json'),weights:await (await fetch('/model/model.safetensors')).blob(),contextLength:2048,encoding,seed:42});
  const base = {repoId:'test',samplingPolicy:{type:'Greedy'},reasoningEffort:'disabled',maxOutputTokens:128};
  const run = async (prompt,extra={},execute) => {
    let done;
    const events=[];
    await session.run({...base,messages:[{role:'user',content:prompt}],...extra},event=>{events.push(event);if(event.type==='done')done=event;},execute);
    assert(done, 'No completed reply');
    return {done,events};
  };
  const basic = await run('Say hello in one sentence.');
  assert(basic.done.text.trim() && basic.done.finishReason==='Stop','Basic Hanashi reply failed: '+JSON.stringify(basic.done));
  checks.push({name:'Hanashi local inference',text:basic.done.text});
  const second = await run('unused',{messages:[{role:'user',content:'Say hello in one sentence.'},{role:'assistant',content:basic.done.text},{role:'user',content:'What is two plus two? Answer briefly.'}]});
  assert(second.done.text.includes('4') || second.done.text.toLowerCase().includes('four'),'Second turn failed: '+second.done.text);
  assert(session.debugReusedTokens()>0, 'Desktop encoding did not reuse completed history');
  checks.push({name:'completed history cache',reusedTokens:session.debugReusedTokens(),text:second.done.text});
  const title = await run('Give this chat a concise descriptive title. Reply with only the title without quotes or explanation. Aim for about 25 characters. Message: "Explain how rainbows form".',{maxOutputTokens:40});
  assert(title.done.text.trim() && !title.done.parsed?.chainOfThought,'Title generation produced no title');
  checks.push({name:'real title generation',title:title.done.text});
  const stochastic={samplingPolicy:{type:'Stochastic',temperature:0.6,topK:20,topP:0.95},maxOutputTokens:16};
  await run('Describe a rainbow in one sentence.',stochastic);
  const seededTokens=Array.from(session.debugTokenIds());
  await run('Describe a rainbow in one sentence.',stochastic);
  assert(seededTokens.length>0 && JSON.stringify(Array.from(session.debugTokenIds()))===JSON.stringify(seededTokens),'Fresh states did not reproduce seeded stochastic sampling');
  checks.push({name:'reproducible seeded stochastic sampling',tokens:seededTokens.length});
  const tool = (name,description,properties={},required=[])=>({type:'function',function:{name,description,parameters:{type:'object',properties,required,additionalProperties:false}}});
  const dateCalls=[];
  const date = await run('Use get_current_date_time to find the current UTC year. Tell me the year returned by the tool.',{tools:[tool('get_current_date_time','Returns the current UTC date and time.')]},call=>{dateCalls.push(call);return {value:new Date().toISOString()};});
  assert(dateCalls.length>0 && dateCalls[0].name==='get_current_date_time','Date tool was not called: '+JSON.stringify(date.done));
  assert(date.done.transcript.some(item=>item.type==='toolCall' && item.called),'Date tool transcript missing');
  assert(date.done.text.includes(String(new Date().getUTCFullYear())),'Date result not used: '+date.done.text);
  checks.push({name:'date tool and automatic continuation',calls:dateCalls,text:date.done.text});
  const namingCalls=[];
  const named = await run('Call set_chat_name with name exactly "Rainbow science" and hidden true, then say Done.',{tools:[tool('set_chat_name','Set the name of the current chat.',{name:{type:'string'},hidden:{type:'boolean'}},['name','hidden'])]},call=>{namingCalls.push(call);return {chatName:call.arguments.name,value:'Chat name accepted.'};});
  assert(namingCalls.length && named.events.some(event=>event.type==='chatName'&&event.name==='Rainbow science'),'Naming tool failed: '+JSON.stringify(named.done));
  assert(!named.done.transcript.some(item=>item.type==='toolCall'&&item.name==='set_chat_name'),'Hidden naming leaked into transcript: '+JSON.stringify({calls:namingCalls,done:named.done}));
  checks.push({name:'naming tool',chatName:named.done.chatName,text:named.done.text});
  const chartCalls=[];
  const chart = await run('Use show_chart to display a bar chart titled Sales, labels May and June, dataset Sales with data 2 and 4.',{maxOutputTokens:256,tools:[tool('show_chart','Display a chart inline.',{chart:{type:'object',properties:{type:{const:'bar'},title:{type:'string'},labels:{type:'array',items:{type:'string'}},datasets:{type:'array',items:{type:'object',properties:{label:{type:'string'},data:{type:'array',items:{type:'number'}}},required:['label','data']}}},required:['type','title','labels','datasets']}},['chart'])]},call=>{chartCalls.push(call);const chart=typeof call.arguments.chart==='string'?JSON.parse(call.arguments.chart):call.arguments.chart;return {chart,value:chart};});
  assert(chartCalls.length && chart.done.transcript.some(item=>item.type==='chart'),'Chart tool failed: '+JSON.stringify(chart.done));
  checks.push({name:'chart tool',chart:chart.done.transcript.find(item=>item.type==='chart')});
  const schema={type:'object',properties:{answer:{type:'string',enum:['blue']}},required:['answer'],additionalProperties:false};
  const json=await run('What color is a clear daytime sky? Return a JSON object with answer.',{grammar:{type:'JsonSchema',schema},maxOutputTokens:64});
  assert(JSON.parse(json.done.text).answer==='blue','JSON schema constraint failed: '+json.done.text);
  checks.push({name:'JSON schema constrained GPU sampling',text:json.done.text});
  const regex=await run('Answer yes.',{grammar:{type:'Regex',pattern:'yes|no'},maxOutputTokens:16});
  assert(/^(yes|no)$/.test(regex.done.text),'Regex constraint failed: '+regex.done.text);
  checks.push({name:'regex constrained GPU sampling',text:regex.done.text});
  const builtin=await run('Return a JSON object with one field named count and number 3.',{grammar:{type:'BuiltinJson'},maxOutputTokens:32});
  assert(typeof JSON.parse(builtin.done.text)==='object','Builtin JSON constraint failed: '+builtin.done.text);
  checks.push({name:'builtin JSON constrained GPU sampling',text:builtin.done.text});
  let done, scheduled=false;
  await session.run({...base,messages:[{role:'user',content:'Explain photosynthesis in detail.'}],reasoningEffort:'default',maxOutputTokens:200},event=>{if(event.type==='done')done=event;else if(!scheduled&&event.type==='transcriptDelta'){scheduled=true;setTimeout(()=>session.cancel(),0);}});
  assert(done?.finishReason==='Cancelled','Reasoning cancellation failed');
  assert(done.transcript.some(item=>item.type==='thinking'),'Reasoning transcript missing');
  checks.push({name:'reasoning and cancellation',tokens:done.stats.totalStats.tokensCountOutput});
  const recovery=await run('Say hello.');
  assert(recovery.done.text.trim()&&recovery.done.finishReason==='Stop','Cancellation recovery failed');
  checks.push('cancellation recovery');
  let pendingToolCalled = false;
  const cancelledTool = await run('Call get_current_date_time now.',{tools:[tool('get_current_date_time','Returns the current UTC date and time.')]},()=>{pendingToolCalled=true;setTimeout(()=>session.cancel(),0);return new Promise(()=>{});});
  assert(pendingToolCalled && cancelledTool.done.finishReason==='Cancelled','Pending tool did not cancel');
  checks.push('cancellation while tool promise is pending');
  postMessage({ok:true,checks});
} catch(error) {postMessage({ok:false,checks,error:String(error?.stack??error)});}
finally {session?.dispose();}
