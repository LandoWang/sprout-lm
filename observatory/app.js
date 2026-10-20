const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const colors = { prefill:'#ed895c', decode:'#83b899', transfer:'#b2a4d1', wait:'#dbbd74', backpressure:'#dbbd74', dispatch:'#8caed1', return:'#8caed1', api:'#91b9ad' };
const names = { prefill:'Prefill 执行', decode:'Decode 执行', transfer:'KV Worker 执行', dispatch:'下发 → 开始', return:'完成 → 回执', wait:'准入 / 调度等待', backpressure:'HTTP 背压', api:'请求 / 文本处理' };
const states = { finished:'生成完成', cancelled:'已取消', failed:'失败', closed:'HTTP 已关闭', incomplete:'未见终态' };
let runList = [], data, selected, windowStart = 0, windowEnd = 1, page = 0, planFilter = '', loading = false, epoch = 0, lastWindow = 'full';
let timelineGeometry, selectedDetail;
const fmt = n => n === null || n === undefined ? '—' : n >= 1000 ? `${(n/1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 2 : 1)} ms`;
const plain = (tag,text,cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const svg = (tag,attrs = {},text) => { const e = document.createElementNS(NS,tag); for (const [key,value] of Object.entries(attrs)) e.setAttribute(key,String(value)); if (text !== undefined) e.textContent = text; return e; };
const request = () => data?.requests.find(r => r.id === selected);
const relative = n => n - (request()?.received ?? request()?.observedStart ?? 0);
const setError = text => { $('error').textContent = text; $('error').hidden = !text; };
function tip(event,text) {
  const el = $('tooltip'); el.textContent = text; el.hidden = false;
  el.style.left = `${Math.max(5,Math.min(event.clientX+12,innerWidth-el.offsetWidth-12))}px`;
  el.style.top = `${Math.max(5,Math.min(event.clientY+12,innerHeight-el.offsetHeight-12))}px`;
}
function interactive(el,text,action) {
  el.classList.add('bar'); el.addEventListener('pointermove',e => tip(e,text));
  el.addEventListener('pointerleave',() => { $('tooltip').hidden = true; });
  el.addEventListener('click',e => { e.stopPropagation(); action?.(); });
}
async function get(url) { const res = await fetch(url); const body = await res.json(); if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`); return body; }
async function loadRuns() {
  const result = await get('/api/runs');
  runList = result.runs;
  const before = $('run').value;
  $('run').replaceChildren(...runList.map(r => { const option = plain('option',`${r.snapshot ? '快照' : '日志'} · ${r.label}`); option.value = r.key; return option; }));
  if (runList.some(r => r.key === before)) $('run').value = before;
  if (result.skipped) setError(`有 ${result.skipped} 个目录无法读取；其余运行仍可查看。`);
  return before !== $('run').value;
}
async function refresh(relist = false) {
  if (loading) return;
  loading = true; $('refresh').disabled = true; const ticket = epoch;
  try {
    setError('');
    const changed = relist ? await loadRuns() : false;
    const key = $('run').value;
    if (!key) { data = undefined; render(); return; }
    const next = await get(`/api/run?run=${encodeURIComponent(key)}`);
    if (ticket !== epoch || key !== $('run').value) return;
    const same = data?.runId === next.runId && !changed;
    data = next;
    if (!same || !data.requests.some(r => r.id === selected)) {
      const requested = new URLSearchParams(location.search).get('request');
      selected = data.requests.find(r => r.id === requested)?.id ?? data.requests[0]?.id;
      planFilter = ''; page = 0; selectedDetail = undefined; lastWindow = 'full';
    }
    if (lastWindow === 'full') fit('full',false);
    render();
  } catch (error) { setError(`${error.message} 已显示的数据保留。`); }
  finally { loading = false; $('refresh').disabled = false; if (ticket !== epoch) void refresh(); }
}
function selectRequest(id) {
  selected = id; page = 0; planFilter = ''; selectedDetail = undefined;
  const url = new URL(location.href); url.searchParams.set('request',id); history.replaceState(null,'',url);
  fit('full',false); render();
}
function fit(mode,paint = true) {
  const r = request(); if (!r) return;
  const base = r.received ?? r.observedStart, token0 = r.tokens.find(t => t.index === 0), token1 = r.tokens.find(t => t.index === 1);
  let a = base, b = r.observedEnd;
  if (mode === 'ttft') b = token0?.published ?? b;
  if (mode === 'handoff') { a = token0?.published ?? base; b = token1?.published ?? Math.min(r.observedEnd,a+1000); a -= Math.max(2,(b-a)*0.15); }
  windowStart = Math.max(r.observedStart,a); windowEnd = Math.max(windowStart+0.1,b);
  lastWindow = mode; if (paint) renderTimeline();
}
function render() {
  const r = request();
  $('workspace').hidden = !r; $('empty').hidden = !!r;
  renderRequests(); if (!r) return;
  $('request-id').textContent = r.id; $('status').textContent = states[r.status];
  $('run-label').textContent = `${runList.find(v => v.key === $('run').value)?.snapshot ? 'SAVED EXPERIMENT' : 'EVENT LOG'} / ${data.runId.slice(0,8)}`;
  $('capture').textContent = `读取于 ${new Date(data.diagnostics.capturedAt).toLocaleTimeString()}`;
  const cards = [
    ['首 token · Service',fmt(r.ttftMs),'request.received → token #0 发布'],
    ['首 token · HTTP',fmt(r.httpTtftMs),'服务端 write，不是客户端收到'],
    ['首 → 第二 token',fmt(r.firstToSecondMs),'单独观察 P → D 交接'],
    ['生成总时长',fmt(r.generationMs),'请求进入 → 生成终态'],
    ['Token 间隔 · P95',fmt(r.itlP95),`P50 ${fmt(r.itlP50)} · Service 发布`],
    ['观察到的 token',`${r.observedTokens}`,`${r.promptTokens ?? '—'} prompt · ${r.planCount} plans`],
  ];
  $('metrics').replaceChildren(...cards.map(([label,value,note]) => { const el=plain('div',undefined,'metric'); el.append(plain('label',label),plain('strong',value),plain('small',note)); return el; }));
  renderIntegrity(); renderTimeline(); renderBreakdown(); renderTokens(); renderGaps(); renderEvents();
  if (selectedDetail) showDetail(selectedDetail.label,selectedDetail.value,false);
  else { $('detail-label').textContent='点击时间线、token 或原始事件。'; $('detail').textContent='选择一个区间，查看各阶段耗时、页号和事件字段。'; }
  $('clear-plan').hidden = !planFilter;
}
function renderRequests() {
  const query = $('request-search').value.toLowerCase();
  $('request-count').textContent = data?.requests.length ?? 0;
  const rs = data?.requests.filter(r => r.id.toLowerCase().includes(query)) ?? [];
  $('requests').replaceChildren(...rs.slice(-200).reverse().map(r => {
    const el = plain('button',undefined,`request-item ${r.id === selected ? 'selected' : ''}`);
    const title = plain('span',undefined,'request-name'); title.append(plain('span',r.id.slice(0,12)),plain('i',undefined,`state-dot ${r.status}`));
    el.append(title,plain('span',`${states[r.status]} · ${r.observedTokens} tokens\nTTFT ${fmt(r.ttftMs)}`,'request-meta'));
    el.title=r.id; el.setAttribute('aria-pressed',String(r.id===selected)); el.onclick=()=>selectRequest(r.id); return el;
  }));
  if (rs.length>200) $('requests').append(plain('p','显示最近 200 个匹配请求；可按 ID 搜索更早记录。','mono'));
  if (!rs.length) $('requests').append(plain('p','未找到请求。','mono'));
}
function renderIntegrity() {
  const d=data.diagnostics,r=request(); const issues=[];
  if (d.losses.length) issues.push(`${d.losses.length} 条丢失报告（运行范围，未必属于本请求）`);
  if (d.sequenceGaps) issues.push(`进程序号内部缺口 ${d.sequenceGaps}，可能仍在传送`);
  if (d.rotatedFiles) issues.push(`${d.rotatedFiles} 个旧文件已轮转`);
  if (d.invalidLines) issues.push(`${d.invalidLines} 条损坏 / 非法记录`);
  if (d.partialLines) issues.push(`${d.partialLines} 条末尾未写完记录`);
  if (d.unavailableFiles) issues.push(`${d.unavailableFiles} 个文件读取时不可用`);
  if (d.evictedEvents) issues.push(`仅保留最近 ${d.eventLimit} 条事件，省略 ${d.evictedEvents} 条`);
  if (d.truncated && !d.evictedEvents) issues.push(`读取达到 ${Math.round(d.byteLimit/1048576)} MiB 上限`);
  if (d.duplicateEvents) issues.push(`${d.duplicateEvents} 条重复事件已去重`);
  if (d.invalidPairs) issues.push(`${d.invalidPairs} 处无法配对或时间倒置`);
  if (r.incompletePlans) issues.push(`本请求 ${r.incompletePlans} 个 plan 缺少阶段记录`);
  if (r.missingStart) issues.push('本请求缺少接收事件，TTFT 不可计算');
  if (!r.terminalSeen) issues.push('本请求未见生成终态，可能仍运行或记录已截断');
  const snapshot=runList.find(v=>v.key===$('run').value)?.snapshot;
  $('integrity').classList.toggle('warn',issues.length>0);
  $('integrity-summary').textContent = issues.length ? `记录有边界 · ${issues.join('；')}`
    : `未发现已报告的丢失或缺失 plan · ${snapshot ? '保存的实验快照' : '本次读取'} · ${r.eventCount} 条请求事件`;
  const paragraphs = [
    d.cleanShutdown ? 'Collector 已正常收尾。正常收尾不保证生产者从未丢事件。' : 'Collector 没有正常收尾标记：可能仍在运行、已异常退出，或这是运行中的截取；不能据此判断服务在线。',
    '所有耗时来自同一台主机的单调时钟。未埋点区间不自动归因；原 CPU / 单卡路径目前仅覆盖 API 和 Service。',
    `本次读到 ${data.events.length} 条运行事件；runId ${data.runId}。${r.resourcesReleased ? '已观察到本请求资源释放。' : '未观察到本请求资源释放；不据此断言泄漏。'}`,
  ];
  $('integrity-content').replaceChildren(...paragraphs.map(t=>plain('p',t)));
}
function renderTimeline() {
  const r=request(); if (!r) return; const root=$('timeline'); root.replaceChildren();
  const width=Math.max(650,root.clientWidth), left=132,right=16,plot=width-left-right,step=29,top=35;
  root.setAttribute('viewBox',`0 0 ${width} 360`);
  const x=t=>left+(t-windowStart)/(windowEnd-windowStart)*plot;
  timelineGeometry={left,right,width,plot};
  for (let i=0;i<=5;i++) { const xx=left+plot*i/5,t=windowStart+(windowEnd-windowStart)*i/5;
    root.append(svg('line',{x1:xx,x2:xx,y1:25,y2:335,stroke:'#2b3c36','stroke-dasharray':'3 4'}),svg('text',{x:xx,y:15,'text-anchor':i===5?'end':'start'},fmt(relative(t)))); }
  const lanes=[['api','API / 文本'],['admission','准入队列'],['wait','调度等待'],['dispatch','下发 IPC'],['prefill','Prefill Worker'],['decode','Decode / KV'],['return','回执 IPC'],['http','HTTP 背压'],['published','Token 发布'],['written','HTTP 写出']];
  lanes.forEach(([key,label],i)=>{const y=top+i*step;root.append(svg('text',{x:5,y:y+14},label),svg('line',{x1:left,x2:width-right,y1:y+24,y2:y+24,stroke:'#21332a'}));});
  const others=$('others').checked;
  const shown=data.spans.filter(s=>(s.requestId===selected || (others && ['prefill','decode'].includes(s.lane))) && s.end>=windowStart && s.start<=windowEnd);
  // Context first; selected request stays on top and retains its actual timing.
  shown.sort((a,b)=>Number(a.requestId===selected)-Number(b.requestId===selected));
  for (const s of shown) {
    const lane=lanes.findIndex(l=>l[0]===s.lane); if (lane<0) continue;
    const a=Math.max(left,x(s.start)),b=Math.min(width-right,x(s.end));
    const rect=svg('rect',{x:a,y:top+lane*step+3,width:Math.max(1,b-a),height:18,rx:3,fill:colors[s.kind]??'#8d9e92',opacity:s.requestId===selected?(s.incomplete ? .4 : .92):.16});
    if (s.planId) rect.setAttribute('data-plan',s.planId);
    if (s.incomplete) {rect.setAttribute('stroke','#dbbd74');rect.setAttribute('stroke-dasharray','3 2');}
    interactive(rect,`${s.label}\n${fmt(s.end-s.start)}${s.incomplete?' · 区间未闭合，不计入耗时统计':''}\nrequest ${s.requestId.slice(0,12)}`,()=>{
      if(s.requestId!==selected) selectRequest(s.requestId);
      if(s.planId) selectPlan(s.planId); else showDetail(s.label,s);
    }); root.append(rect);
    if (b-a>65) root.append(svg('text',{x:a+5,y:top+lane*step+16,fill:'#0b1a11','pointer-events':'none',style:'fill:#102019;font-size:9px'},s.planId??s.label));
  }
  for(const [field,lane] of [['published',8],['written',9]]) for(const t of r.tokens) {
    const at=t[field];if(at===undefined||at<windowStart||at>windowEnd)continue;
    const mark=svg('rect',{x:Math.max(left,Math.min(width-right-2,x(at))),y:top+lane*step+5,width:t.index===0?4:2,height:14,rx:1,fill:field==='published'?'#c5dbb1':'#8caed1'});
    interactive(mark,`Token #${t.index} · ${field}\n请求 + ${fmt(relative(at))}`,()=>showDetail(`Token #${t.index}`,t));root.append(mark);
  }
  const zero=r.tokens.find(t=>t.index===0)?.published;
  if(zero!==undefined&&zero>=windowStart&&zero<=windowEnd){const xx=x(zero);root.append(svg('line',{x1:xx,x2:xx,y1:25,y2:335,stroke:'#e6d2a5','stroke-dasharray':'4 4','pointer-events':'none'}),svg('text',{x:Math.min(xx+4,width-105),y:350},'首 token · Service'));}
  $('window-label').textContent=`请求 + ${fmt(relative(windowStart))} → ${fmt(relative(windowEnd))}`;
  const full=Math.max(.1,r.observedEnd-r.observedStart),range=windowEnd-windowStart;
  $('pan').disabled=range>=full;
  $('pan').value=String(Math.max(0,Math.min(1000,(windowStart-r.observedStart)/Math.max(.001,full-range)*1000)));
  for(const mode of ['full','ttft','handoff'])$(mode).classList.toggle('active',lastWindow===mode);
}
function showDetail(label,value,remember=true){$('detail-label').textContent=label;$('detail').textContent=JSON.stringify(value,null,2);if(remember)selectedDetail={label,value};}
function selectPlan(id){const p=data.plans.find(p=>p.id===id&&p.requestId===selected);if(!p)return;planFilter=id;page=0;$('clear-plan').hidden=false;showDetail(`${id} · ${p.phase}`,p);renderEvents();}
function renderBreakdown(){
  const r=request(),rows=$('breakdown-mode').value==='ttft'?r.ttftSegments:r.breakdown;
  const max=Math.max(1,...rows.map(v=>v.ms));
  $('breakdown').replaceChildren(...rows.map(v=>{const row=plain('div',undefined,'breakdown-row'),bg=plain('div',undefined,'bar-bg'),bar=plain('div',undefined,'bar-fill');bar.style.width=`${v.ms/max*100}%`;bar.style.background=colors[v.kind];bg.append(bar);row.append(plain('span',names[v.kind]),bg,plain('span',fmt(v.ms),'amount'));return row;}));
  if(!rows.length)$('breakdown').append(plain('p','缺少接收或首 token 事件，无法拆分 TTFT。','mono'));
  const own=data.plans.filter(p=>p.requestId===selected&&p.executionMs!==null).sort((a,b)=>b.executionMs-a.executionMs);
  const findings=[];
  if(own[0])findings.push(`最长 Worker 计划：${own[0].id}，${fmt(own[0].executionMs)}。点击时间线核对该计划。`);
  if(r.hasKvCuda)findings.push(`KV CUDA Event 合计 ${fmt(r.kvCudaMs)}；它只测已埋点的拷贝，不含全部排队 / 控制开销。`);
  findings.push(`输出缓冲峰值 ${r.maxBufferedEvents} events；socket 缓冲峰值 ${r.maxWritableBytes} bytes。`);
  if(!r.planCount)findings.push('未记录 Worker 计划：当前后端覆盖有限，或计划事件已丢失，不能据此判断计算耗时。');
  $('findings').replaceChildren(...findings.map(t=>plain('p',t)));
}
function renderTokens(){
  const r=request(),series=$('token-mode').value==='published'?r.itl:r.writtenItl,root=$('tokens');root.replaceChildren();
  const width=Math.max(280,root.clientWidth-24),height=230,left=48,right=10,bottom=28,top=20,pw=width-left-right,ph=height-bottom-top;
  root.setAttribute('viewBox',`0 0 ${width} ${height}`);
  if(!series.length){root.append(svg('text',{x:20,y:95},'没有可计算的相邻 token 事件。'));$('token-summary').textContent='JSON 响应不会逐 token 写出；缺失记录也可能导致没有间隔数据。';return;}
  const max=Math.max(1,...series.map(t=>t.ms))*1.1,last=Math.max(1,...series.map(t=>t.index));
  for(let i=0;i<=3;i++){const y=top+ph*i/3;root.append(svg('line',{x1:left,x2:width-right,y1:y,y2:y,stroke:'#2b3c36'}),svg('text',{x:left-5,y:y+4,'text-anchor':'end'},(max*(1-i/3)).toFixed(0)));}
  root.append(svg('text',{x:8,y:12},'ms'),svg('text',{x:left,y:height-8},'#1'),svg('text',{x:width-right,y:height-8,'text-anchor':'end'},`#${last}`));
  for(const t of series){const x=left+(t.index-1)/last*pw,w=Math.max(1,pw/last*.7),h=Math.max(1,t.ms/max*ph);
    const bar=svg('rect',{x,y:top+ph-h,width:w,height:h,rx:1,fill:t.index===1?'#ed895c':'#83b899'});
    interactive(bar,`Token #${t.index-1} → #${t.index}\n${fmt(t.ms)}`,()=>{const padding=Math.max(2,t.ms*.2);windowStart=Math.max(r.observedStart,t.at-t.ms-padding);windowEnd=Math.min(r.observedEnd,t.at+padding);lastWindow='custom';renderTimeline();showDetail(`Token #${t.index-1} → #${t.index}`,t);});root.append(bar);
  }
  const maxGap=series.reduce((a,b)=>a.ms>b.ms?a:b);
  $('token-summary').textContent=`${series.length} 个相邻间隔 · 最大 ${fmt(maxGap.ms)}（#${maxGap.index-1} → #${maxGap.index}）· 橘色是首 → 第二 token`;
}
function renderGaps(){const r=request();const gaps=data.gaps.filter(g=>g.end>=r.observedStart&&g.start<=r.observedEnd).slice(0,5);
  $('gaps').replaceChildren(...gaps.map(g=>{const row=plain('tr');for(const t of [g.worker,fmt(relative(g.start)),fmt(g.ms),g.nextPlanId])row.append(plain('td',t));row.onclick=()=>{windowStart=Math.max(r.observedStart,g.start-2);windowEnd=Math.min(r.observedEnd,g.end+2);lastWindow='custom';renderTimeline();showDetail('运行范围的相邻 Worker 执行间隙',g);};return row;}));
  if(!gaps.length){const row=plain('tr'),cell=plain('td','没有可配对的相邻 Worker 计划。');cell.colSpan=4;row.append(cell);$('gaps').append(row);}
}
function renderEvents(){const r=request();if(!r)return;const query=$('event-search').value.toLowerCase();
  const es=data.events.filter(e=>e.requestId===selected&&(!planFilter||e.planId===planFilter)&&(!query||JSON.stringify(e).toLowerCase().includes(query)));
  const pages=Math.max(1,Math.ceil(es.length/100));page=Math.min(page,pages-1);
  $('event-count').textContent=`${es.length} 条${planFilter?` · plan ${planFilter}`:''} · 时间相对 request.received（缺失时相对第一条记录）`;
  $('events').replaceChildren(...es.slice(page*100,(page+1)*100).map(e=>{const row=plain('tr');for(const t of [relative(e.at).toFixed(3),`${e.source} / ${e.pid}`,e.name,e.planId??'—',JSON.stringify(e.data??{})])row.append(plain('td',t));row.onclick=()=>showDetail(e.name,e);return row;}));
  $('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;
}
function zoom(factor,center){const r=request();if(!r)return;const full=Math.max(.1,r.observedEnd-r.observedStart),old=windowEnd-windowStart,range=Math.min(full,Math.max(.1,old*factor)),at=center??(windowStart+windowEnd)/2;const fraction=(at-windowStart)/old;windowStart=Math.max(r.observedStart,Math.min(r.observedEnd-range,at-range*fraction));windowEnd=windowStart+range;lastWindow='custom';renderTimeline();}
$('run').onchange=()=>{epoch++;data=undefined;selected=undefined;setTimeout(()=>refresh(),0);};
$('refresh').onclick=()=>refresh(true);
$('request-search').oninput=renderRequests;
for(const mode of ['full','ttft','handoff'])$(mode).onclick=()=>fit(mode);
$('zoom-in').onclick=()=>zoom(.5);$('zoom-out').onclick=()=>zoom(2);
$('others').onchange=renderTimeline;
$('pan').oninput=()=>{const r=request();if(!r)return;const range=windowEnd-windowStart;windowStart=r.observedStart+(r.observedEnd-r.observedStart-range)*Number($('pan').value)/1000;windowEnd=windowStart+range;lastWindow='custom';renderTimeline();};
$('breakdown-mode').onchange=renderBreakdown;$('token-mode').onchange=renderTokens;
$('event-search').oninput=()=>{page=0;renderEvents();};$('prev').onclick=()=>{page--;renderEvents();};$('next').onclick=()=>{page++;renderEvents();};
$('clear-plan').onclick=()=>{planFilter='';page=0;$('clear-plan').hidden=true;renderEvents();};
$('export').onclick=()=>{const r=request();if(!r)return;const body={runId:data.runId,requestId:r.id,summary:r,events:data.events.filter(e=>e.requestId===r.id),plans:data.plans.filter(p=>p.requestId===r.id),diagnostics:data.diagnostics};const url=URL.createObjectURL(new Blob([JSON.stringify(body,null,2)],{type:'application/json'}));const a=plain('a');a.href=url;a.download=`sprout-${r.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
let drag;
function timeAt(event){const g=timelineGeometry,box=$('timeline').getBoundingClientRect();if(!g)return;const px=(event.clientX-box.left)*g.width/box.width;return windowStart+Math.max(0,Math.min(1,(px-g.left)/g.plot))*(windowEnd-windowStart);}
$('timeline').addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,start:timeAt(e),end:timeAt(e)};});
$('timeline').addEventListener('pointermove',e=>{if(!drag)return;drag.end=timeAt(e);$('timeline').querySelector('#selection')?.remove();if(Math.abs(e.clientX-drag.x)>5){$('tooltip').hidden=true;const g=timelineGeometry,x=t=>g.left+(t-windowStart)/(windowEnd-windowStart)*g.plot;const mark=svg('rect',{id:'selection',x:x(Math.min(drag.start,drag.end)),y:24,width:Math.abs(x(drag.end)-x(drag.start)),height:310,fill:'#edf2e9',opacity:.12,'pointer-events':'none'});$('timeline').append(mark);}});
window.addEventListener('pointerup',e=>{if(!drag)return;const d=drag;drag=undefined;$('timeline').querySelector('#selection')?.remove();if(Math.abs(e.clientX-d.x)>5&&Math.abs(d.end-d.start)>.1){windowStart=Math.min(d.start,d.end);windowEnd=Math.max(d.start,d.end);lastWindow='custom';renderTimeline();}});
$('timeline').addEventListener('wheel',e=>{if(!request()||(!e.ctrlKey&&!e.metaKey))return;e.preventDefault();zoom(e.deltaY>0?1.3:1/1.3,timeAt(e));},{passive:false});
let resize;window.addEventListener('resize',()=>{clearTimeout(resize);resize=setTimeout(()=>{if(request()){renderTimeline();renderTokens();}},100);});
setInterval(()=>{if($('auto').checked&&!document.hidden)void refresh(true);},3000);
void refresh(true);
