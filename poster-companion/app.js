'use strict';
(() => {
  const D = window.POSTER_DATA;
  if (!D) { document.querySelector('#load-error').hidden = false; return; }
  const V = window.CaveVisuals;
  const $ = (s) => document.querySelector(s);
  const escape = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct = (n, digits=1) => (n*100).toFixed(digits)+'%';
  const score = n => (n*1000).toFixed(2);
  const exact = n => String(Number(n));
  const signed = n => (n>=0?'+':'−')+Math.abs(n).toFixed(6);
  const relative = (a,b) => (b/a-1);
  const groups = {all:'All products',new:'New in the period',warm:'Established products',head:'Established · head',tail:'Established · tail'};
  const names = {P1_ranker:'Original recency lists',P1ranker_on_P3:'Mixed · no retraining',P3_sizematched_ranker:'Mixed · retrained, same size',P3_ranker:'Mixed · retrained, full data',mlp_tt:'MLP · two-tower lists',din_tt:'DIN · two-tower lists',mlp_rrf:'MLP · mixed lists',din_rrf:'DIN · mixed lists',S0:'Recency order · no ranker',S1:'Original residual · recency / T0',S2:'Same model · Aug 2021 statistics',S3:'Same model · Test-start statistics',S4:'Recency · monthly text',S5:'Recency · monthly co-review',S6:'Mixed · T0 text',S7:'Mixed · monthly text',S8:'Mixed · monthly co-review',T0:'Jul 2020 · original',T1:'Aug 2021 · intermediate',T2:'Jul 2022 · Test start',X1_known:'Only “has reviews” flag',X2_known_degree:'Flag + review count',X3_all_common:'All common statistics',X6_old_products:'Only established-at-T0 items',X7_products_released_in_B:'Only items first seen in B'};
  const label = s => names[s] || (s.includes('/') ? s.replace('armC_mlp/','MLP · ').replace('din/','DIN · ').replace('tt200','two-tower 200').replace('tt500','two-tower 500').replace('fresh','fresh retriever').replace('mixed','mixed lists') : s.startsWith('lag') ? s.slice(3)+' month'+(s==='lag1'?'':'s')+' lag' : s);
  const section = {retrieval:'rq1',k:'rq2',ranker:'rq3',frozen:'r1',retrain:'r2',timing:'r3',final:'r4',ann:'takeaways'};
  const state = {k:200,step:3,view:'guide',explorer:'retrieval',method:'query'};
  const currentRows = {};
  const row = (study,name,group='all') => D.studies[study].rows.find(r=>r.label===name&&r.group===group);
  const contrast = (study,ref,cand,group='all') => D.studies[study].contrasts?.find(r=>r.reference===ref&&r.candidate===cand&&r.group===group);
  function ciText(c) {
    return c ? `Paired Δ ${signed(c.diff)}; 95% CI [${signed(c.ci_low)}, ${signed(c.ci_high)}].` : 'Paired confidence interval unavailable for this comparison.';
  }
  function barChart(target, rows, options={}) {
    const el=typeof target==='string'?$(target):target;
    const width=Math.max(280,Math.min(850,el.clientWidth||600));
    const left=width<400?Math.round(width*.41):Math.min(230,Math.round(width*.36));
    const right=options.percent?60:70, plot=Math.max(60,width-left-right), top=12, step=56, height=rows.length*step+48;
    const maximum=options.maximum||Math.max(...rows.map(r=>r.value),.001)*1.12;
    const display=v=>options.percent?pct(v):score(v);
    const wrap = (text,limit) => {
      const words=text.split(' '), lines=[''];
      for(const word of words){const i=lines.length-1;if((lines[i]+' '+word).trim().length>limit&&lines[i])lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}
      return lines;
    };
    let svg=`<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(options.title||'Measured results')}"><title>${escape((options.title||'Measured results')+'. '+rows.map(r=>`${label(r.label)}: ${exact(r.value)}`).join('; '))}</title>`;
    for(let i=0;i<=3;i++){
      const x=left+plot*i/3;
      svg+=`<line class="gridline" x1="${x}" x2="${x}" y1="4" y2="${height-38}"/><text class="axis-label" x="${x}" y="${height-16}" text-anchor="middle">${options.percent?(maximum*i/3*100).toFixed(1)+'%':(maximum*i/3*1000).toFixed(1)}</text>`;
    }
    rows.forEach((r,i)=>{
      const y=top+i*step, text=options.labelFormatter?options.labelFormatter(r):label(r.label);
      const lines=wrap(text,Math.max(15,Math.floor(left/6.5)));
      svg+=`<text x="0" y="${y+15-(lines.length>1?7:0)}">${lines.map((line,j)=>`<tspan x="0" dy="${j?15:0}">${escape(line)}</tspan>`).join('')}</text>`;
      let cls=options.selected===r.label?'accent selected':options.accent===r.label?'accent':r.tone||'';
      if(options.selected!==undefined&&options.selected!==r.label)cls='dim';
      svg+=`<rect class="bar ${cls}" x="${left}" y="${y}" width="${Math.max(0,r.value/maximum*plot)}" height="22" rx="1"><title>${escape(text)}: ${exact(r.value)}</title></rect><text class="value-label" x="${left+plot+10}" y="${y+16}">${display(r.value)}</text>`;
    });
    svg+='</svg>';
    el.innerHTML=`<p class="chart-title">${escape(options.title||'nDCG@10 × 10⁻³ · higher is better')}</p>${svg}`;
  }
  function table(rows, title='Exact saved values') {
    const withK=rows.some(r=>r.recall!==undefined);
    return `<div class="table-scroll"><table><caption>${escape(title)} · scores shown unscaled</caption><thead><tr><th>Condition</th><th>Group</th><th>Metric</th><th>Value</th>${withK?'<th>Recall@K</th>':''}</tr></thead><tbody>${rows.map(r=>`<tr><th>${escape(label(r.label))}</th><td>${escape(groups[r.group]||r.group)}</td><td>${escape(r.metric)}</td><td>${exact(r.value)}</td>${withK?`<td>${r.recall!==undefined?exact(r.recall):'—'}</td>`:''}</tr>`).join('')}</tbody></table></div>`;
  }
  function evidenceHtml(key, rows) {
    const s=D.studies[key];
    const cis=(s.contrasts||[]).filter(c=>rows.some(r=>r.label===c.reference&&r.group===c.group)&&rows.some(r=>r.label===c.candidate&&r.group===c.group));
    const kIntervals=key==='k'?`<div class="table-scroll"><table><caption>Paired nDCG@10 differences versus K=100 · unscaled</caption><thead><tr><th>K</th><th>Δ</th><th>95% CI</th></tr></thead><tbody>${rows.map(r=>`<tr><th>${r.k}</th><td>${r.diff===null?'Reference':signed(r.diff)}</td><td>${r.ci_low===null?'—':`[${signed(r.ci_low)}, ${signed(r.ci_high)}]`}</td></tr>`).join('')}</tbody></table></div>`:'';
    return `<div class="chart-context">${escape(s.context)}</div><div class="evidence-tools"><a href="#explorer/${key}">Explore this comparison ↗</a><a href="#methods">How we evaluated →</a></div><details class="evidence"><summary>Read exact values & evaluation details</summary><p>${escape(s.note)}</p>${table(rows)}${kIntervals}${cis.length?`<div class="table-scroll"><table><caption>Available paired differences · candidate minus reference · unscaled nDCG@10</caption><thead><tr><th>Comparison</th><th>Group</th><th>Δ</th><th>95% CI</th></tr></thead><tbody>${cis.map(c=>`<tr><th>${escape(label(c.candidate))}<br>− ${escape(label(c.reference))}</th><td>${escape(groups[c.group])}</td><td>${signed(c.diff)}</td><td>[${signed(c.ci_low)}, ${signed(c.ci_high)}]</td></tr>`).join('')}</tbody></table></div>`:key==='k'?'':'<p>Paired interval unavailable for the displayed comparison. Seed standard deviations are not confidence intervals.</p>'}<details class="source-details"><summary>Source records</summary>${s.sources.map(id=>`<div class="source-entry">${escape(D.sources[id].path)}</div>`).join('')}</details></details>`;
  }
  function evidence(key,rows){currentRows[key]=rows;document.querySelectorAll(`[data-evidence="${key}"]`).forEach(el=>{const was=el.querySelector('details')?.open;el.innerHTML=evidenceHtml(key,rows);if(was)el.querySelector('details').open=true;});}
  let trainingHit=true;
  function renderMethod(){
    document.querySelectorAll('[data-method]').forEach(b=>{const active=b.dataset.method===state.method;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',active);});
    const illustration='<div class="eyebrow">ILLUSTRATION · NOT ACTUAL REVIEW RECORDS</div>';
    const scenes={
      query:`${illustration}<div class="query-track"><div><span>PAST</span><b>Case → Charger → Cable</b><small>Only reviews strictly before time t</small></div><span class="query-cut">│ t</span><div class="target-card"><span>TARGET</span><b>Next reviewed product</b><small>Hidden from the model’s inputs</small></div></div><h3>Each review with earlier history becomes a query.</h3><p>Remove duplicate user–product reviews, keeping the earliest. Use the most recent 50 earlier reviews as model history; skip queries with empty history. Reviews at the target’s exact timestamp are not part of its history. All star ratings count as observed interactions.</p><p class="fine">Earlier reviews from the same evaluation period can enter a later query’s history. They are past observations at that later time; they are not future target information.</p>`,
      train:`${illustration}<div class="training-switch"><button data-training-hit="yes" aria-pressed="${trainingHit}" class="${trainingHit?'selected':''}">Target retrieved</button><button data-training-hit="no" aria-pressed="${!trainingHit}" class="${!trainingHit?'selected':''}">Target missed</button></div><div class="candidate-example">${Array.from({length:6},(_,i)=>`<span class="${trainingHit&&i===2?'example-target':''}">${trainingHit&&i===2?'Target':'Item'}</span>`).join('')}</div><div class="training-status"><b>${trainingHit?'Included in natural-only training':'Excluded from natural-only training'}</b><p>${trainingHit?'The shortlist naturally contains the next-reviewed product. In the real recipe, train on that positive and 50 sampled non-target candidates.':'The shortlist does not contain the target. Do not append it for the natural-only training study. This query still counts in end-to-end evaluation.'}</p></div><p class="fine">Six tiles are a schematic of a 200-item shortlist. “Natural-positive” means a query whose target was already retrieved. Sampled non-targets are unobserved, not verified dislikes.</p>`,
      evaluate:`<div class="outcome-flow"><article><span class="outcome-icon">⊘</span><b>Retrieval miss</b><p>Target absent from the shortlist.</p><strong>nDCG@10 = 0</strong></article><article><span class="outcome-icon">↓</span><b>Ranking miss</b><p>Target retrieved, but below rank 10.</p><strong>nDCG@10 = 0</strong></article><article><span class="outcome-icon">✓</span><b>Top-ten hit</b><p>Target retrieved and ranked 1–10.</p><strong>Higher position → more credit</strong></article></div><h3>Evaluate every eligible query, including retrieval misses.</h3><p>Retrieve from the eligible product catalogue, filtering all products in the user’s full earlier history and products whose first review is later than the query. End-to-end evaluation never force-inserts a missed target.</p><p class="fine">Full-catalogue candidate search differs from the number of queries evaluated. Main Validation/Test results use their complete query populations; the separate ANN benchmark uses 1,000 fixed Validation queries. First review is an availability proxy, not a verified launch date.</p>`,
      average:`${illustration}<div class="macro-example"><div><b>Example user A</b><span class="event-dots">●</span><span>1 hit / 1 query = 100%</span></div><div><b>Example user B</b><span class="event-dots">● ○ ○</span><span>1 hit / 3 queries = 33.3%</span></div></div><div class="average-results"><div><b>66.7%</b><span>Per-user macro: (100% + 33.3%) / 2</span></div><div><b>50.0%</b><span>Event micro: 2 hits / 4 queries</span></div></div><h3>Give each user equal weight.</h3><p>Main reported ranking metrics average a user’s query scores first, then average across users. Otherwise frequent reviewers dominate. Subgroup scores use the users and queries within that subgroup.</p><p class="fine">This toy Recall@10 example explains averaging, not a project result. Macro recall and event-micro coverage have different denominators; their difference does not prove different candidate pools.</p>`,
      uncertainty:`<div class="bootstrap-flow"><div><b>3 trained seeds</b><small>Same aligned queries</small></div><span>→</span><div><b>Per-query seed mean</b><small>Compare A and B on the same query</small></div><span>→</span><div><b>Per-user difference</b><small>Average each user’s queries</small></div><span>→</span><div><b>1,000 user resamples</b><small>Keep both systems paired</small></div></div><h3>Ask whether the measured difference is stable across users.</h3><p>The interval is the middle 95% of the resampled differences. Use paired intervals for differences between systems, not as error bars around individual system scores. An interval including zero does not establish that the systems are equivalent.</p><p class="fine">These intervals are conditional on the fitted seeds and unadjusted for the many comparisons. They do not capture all model-training uncertainty or remove the limits of repeated Validation/Test exposure.</p>`
    };
    $('#method-scene').innerHTML=scenes[state.method];
  }
  function showStage(stage){
    document.querySelectorAll('[data-stage]').forEach(b=>{const selected=b.dataset.stage===stage;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',selected);});
    const texts={catalogue:'<b>The search space.</b> There are 111,480 products in the catalogue. For each query, exclude products not yet observed and products the user already reviewed. The target is the next reviewed product.',retriever:'<b>The first chance to fail.</b> A retriever selects up to 200 eligible products. If the target is absent, it is a retrieval miss: even a perfect ranker cannot recover it. Recall@200 measures this stage’s coverage.',ranker:'<b>The second chance to fail.</b> The ranker scores the retrieved products and shows ten. A retrieved target ranked below ten is a ranking miss. nDCG@10 rewards hits near the top, with retrieval misses counted as zero.'};
    $('#stage-explanation').innerHTML=texts[stage];
  }
  function renderMetric(){
    const rank=Number($('#metric-rank').value), found=rank<=20;
    $('#metric-rank-output').textContent=found?rank:'Not retrieved';
    $('#metric-rank').setAttribute('aria-valuetext',found?`Target at rank ${rank}`:'Target not retrieved');
    $('#metric-tiles').innerHTML=Array.from({length:20},(_,i)=>`<span class="${i<10?'top10 ':''}${i+1===rank?'target':''}"></span>`).join('');
    const ndcg=rank<=10?1/Math.log2(rank+1):0;
    $('#metric-values').innerHTML=`<div><span>Recall@20 · this one example</span><b>${found?'1':'0'}</b></div><div><span>nDCG@10 · this one example</span><b>${ndcg.toFixed(3)}</b></div><small>${found?(rank<=10?'A hit, discounted by its position.':'Retrieved, but below the ten shown.'):'A retrieval miss. Both scores are zero.'}</small>`;
  }
  function renderRetrieval(){
    const group=$('#retrieval-group').value;
    const rows=D.studies.retrieval.rows.filter(r=>r.group===group).sort((a,b)=>a.value-b.value);
    barChart('#retrieval-chart',rows,{percent:true,title:'Recall@200 · higher is better',accent:'Mixed lists'});
    const mix=row('retrieval','Mixed lists',group), tt=row('retrieval','Two-tower',group), rec=row('retrieval','Recency',group);
    $('#retrieval-insight').innerHTML=`Mixed recall: <strong>${pct(mix.value)}</strong>. Two-tower: ${pct(tt.value)}; recency: ${pct(rec.value)}. ${group==='new'?'For new products, recency alone retrieves more targets than this mixed pool. Mixing is a trade-off.':'Complementary sources can recover targets missed by a single list.'}`;
    $('#overlap-number').textContent=`${D.overlap.toFixed(1)} / 200`;
    V.pool('#pool-visual',D.studies.retrieval.mixing);
    evidence('retrieval',rows);
  }
  function renderK(){
    const rows=D.studies.k.rows, selected=String(state.k), r=rows.find(r=>r.k===state.k);
    $('#k-controls').innerHTML=rows.map(r=>`<button data-k="${r.k}" class="${r.k===state.k?'selected':''}" aria-pressed="${r.k===state.k}">K = ${r.k}</button>`).join('');
    V.trend('#k-recall-chart',[{label:'Candidate recall',points:rows.map(r=>({x:r.k,value:r.recall}))}],{percent:true,selected:state.k,title:'Recall@K · fixed retriever'});
    V.trend('#k-quality-chart',[{label:'Final top-10 quality',points:rows.map(r=>({x:r.k,value:r.value}))}],{selected:state.k,title:'nDCG@10 × 10⁻³ · frozen MLP'});
    const base=rows.find(r=>r.k===200);
    const rel=relative(base.value,r.value);
    $('#k-insight').innerHTML=`At <b>K=${state.k}</b>, recall is <b>${pct(r.recall)}</b> and top-10 quality is <b>${score(r.value)} × 10⁻³</b>. ${state.k===200?'This is the candidate count used in training.':`Compared with K=200, ranking quality changes by ${rel>=0?'+':'−'}${pct(Math.abs(rel))}.`}<br><span class="contrast">${r.diff===null?'K=100 is the reference for the saved paired K intervals.':`Versus K=100: ${ciText(r)}`}</span>`;
    renderExpansion();
    evidence('k',rows);
  }
  function renderExpansion(){
    const [low,high]=$('#expansion-range').value.split('-').map(Number);
    V.waterfall('#competition-chart',D.studies.k.expansion.find(r=>r.low===low&&r.high===high));
  }
  function renderRanker(){
    const pool=$('#ranker-pool').value, group=$('#ranker-group').value;
    const rows=['mlp_'+pool,'din_'+pool].map(n=>row('ranker',n,group));
    barChart('#ranker-chart',rows,{accent:rows[1].label,title:'nDCG@10 × 10⁻³ · same candidate pool'});
    const change=relative(rows[0].value,rows[1].value),c=contrast('ranker',rows[0].label,rows[1].label,group);
    $('#ranker-insight').innerHTML=`DIN changes quality by <strong>${change>=0?'+':'−'}${pct(Math.abs(change))}</strong> against MLP on these lists.<br><span class="contrast">${ciText(c)} ${pool==='rrf'?'Uses corrected, training-consistent score normalisation.':''}</span>`;
    evidence('ranker',rows);
  }
  function renderFrozen(){
    const rows=D.studies.frozen.rows.filter(r=>r.group==='all');
    const get=(ranker,list)=>rows.find(r=>r.label===ranker+'/'+list);
    const titles={tt200:'Original two-tower · 200',tt500:'Original two-tower · 500',fresh:'Fresh two-tower · 200',mixed:'Mixed lists · 200'};
    const quality=(ranker,list)=>{const r=get(ranker,list),delta=relative(get(ranker,'tt200').value,r.value);return score(r.value)+(list==='tt200'?'':` (${delta>=0?'+':'−'}${pct(Math.abs(delta))})`);};
    $('#frozen-table').innerHTML=`<div class="table-scroll"><table><caption>Test · frozen October rankers · nDCG@10 × 10⁻³</caption><thead><tr><th>Candidate list</th><th>Macro recall</th><th>MLP</th><th>DIN</th></tr></thead><tbody>${Object.entries(titles).map(([key,title])=>`<tr><th>${escape(title)}</th><td>${pct(get('armC_mlp',key).recall,3)}</td><td>${quality('armC_mlp',key)}</td><td>${quality('din',key)}</td></tr>`).join('')}</tbody></table></div>`;
    evidence('frozen',rows);
  }
  const steps=['P1_ranker','P1ranker_on_P3','P3_sizematched_ranker','P3_ranker'];
  const stepDescriptions=['Recency candidates at training and scoring. The reference ranker sees 20,676 natural-positive training lists.','Keep the recency-trained ranker frozen and score mixed lists. This tests the changed inputs without retraining.','Retrain on 20,676 mixed-pool examples: the same count as the reference. This controls training-set size, not every aspect of exposure or optimisation.','Retrain on all 38,861 natural-positive mixed-pool lists. More retrieved targets also provide more training examples.'];
  function renderRetrain(){
    const group=$('#retrain-group').value, rows=steps.map(n=>row('retrain',n,group)), r=rows[state.step];
    V.controls('#retrain-matrix',state.step);
    $('#retrain-controls').innerHTML=steps.map((n,i)=>`<button data-step="${i}" class="${state.step===i?'selected':''}" aria-pressed="${state.step===i}"><span>0${i+1}</span>${escape(label(n))}</button>`).join('');
    barChart('#retrain-chart',rows,{selected:r.label,title:'Validation · nDCG@10 × 10⁻³'});
    const c=contrast('retrain',steps[0],r.label,group);
    $('#retrain-insight').innerHTML=`<strong>Condition ${state.step+1}.</strong> ${escape(stepDescriptions[state.step])}<br><span class="contrast">${state.step===0?'Reference condition.':`${relative(rows[0].value,r.value)>=0?'+':'−'}${pct(Math.abs(relative(rows[0].value,r.value)))} versus the original. ${ciText(c)}`}</span>`;
    const a=row('final','S1'),b=row('final','S6');
    $('#retrain-test').textContent=`The full mixed-pool ranker improves ${score(a.value)} → ${score(b.value)} × 10⁻³ (${pct(relative(a.value,b.value),0)} overall) on Test. This is a separate confirmation, not the four-condition Validation chart. ${ciText(contrast('final','S1','S6'))}`;
    evidence('retrain',rows);
  }
  function timingRows(group,mode){
    if(mode==='snapshot')return ['T0','T1','T2'].map(n=>row('timing',n,group));
    const ids=mode==='lag'?['lag0','lag1','lag3','lag6','lag9','lag12']:['T0','X1_known','X2_known_degree','X6_old_products','X7_products_released_in_B','T2'];
    return ids.map(n=>row('timing',n,group));
  }
  function renderTiming(){
    const group=$('#timing-group').value,mode=$('#timing-mode').value,rows=timingRows(group,mode);
    if(mode==='lag'){
      const series=['new','warm',...(group==='all'?['all']:[])].map(g=>({label:groups[g],points:timingRows(g,'lag').map(r=>({x:Number(r.label.slice(3)),value:r.value}))}));
      V.trend('#timing-chart',series,{title:'Test · nDCG@10 × 10⁻³ · frozen ranker',xLabel:'Months before each query month'});
    }else barChart('#timing-chart',rows,{accent:'T2',title:'Test · nDCG@10 × 10⁻³ · frozen ranker'});
    let text;
    if(mode==='snapshot'){
      const c=contrast('timing','T0','T2',group);
      text=`Same weights, candidates and training recipe. Updating from T0 to the Test-start snapshot changes quality by <strong>+${pct(relative(rows[0].value,rows[2].value))}</strong> for this group.<br><span class="contrast">${ciText(c)}</span>`;
    }else if(mode==='lag'){
      const best=rows.reduce((a,b)=>b.value>a.value?b:a);
      text=`For this frozen model and group, the highest measured point is <strong>${escape(label(best.label))}</strong>. New products and established products prefer different ages; this does not mean stale data is generally better.<br><span class="contrast">${best.label==='lag0'?'Lag 0 is the reference condition.':ciText(contrast('timing','lag0',best.label,group))}</span>`;
    }else text='These interventions change only the named statistics. The learned review-presence cue explains part of the timing response; it is not a complete causal account of every feature or pipeline.';
    $('#timing-insight').innerHTML=text;
    const tr=D.studies.timing.training_cue, orig=D.studies.timing.age.find(r=>r.label==='T0'), end=D.studies.timing.age.find(r=>r.label==='T2');
    $('#cue-chart').innerHTML=[['Training · T0',tr.target,tr.other],['Test · original T0',orig.target_unknown,orig.other_unknown],['Test · period start',end.target_unknown,end.other_unknown]].map(([name,target,other])=>`<p class="fine" style="margin:20px 0 0">${name}</p><div class="cue-row"><span>Target</span><span class="cue-bar"><i style="width:${target*100}%"></i></span><b>${pct(target,0)}</b></div><div class="cue-row other"><span>Other</span><span class="cue-bar"><i style="width:${other*100}%"></i></span><b>${pct(other,0)}</b></div>`).join('');
    evidence('timing',rows);
    V.reviewTimeline('#review-timeline',Number($('#snapshot-example').value));
  }
  function renderFinal(){
    const baseline=$('#final-baseline').value;
    // The separate RQ3 export has head/tail slices, not a combined warm macro.
    const warmOption=$('#final-group option[value="warm"]');warmOption.disabled=baseline==='original';
    if(baseline==='original'&&$('#final-group').value==='warm')$('#final-group').value='all';
    const group=$('#final-group').value;
    let rows, text;
    if(baseline==='matched'){
      rows=['S1','S6','S7','S8','S5'].map(n=>row('final',n,group));
      const a=rows[0],b=row('final','S8',group),rel=relative(a.value,b.value);
      text=`Combined system versus its own residual baseline: <strong>${rel>=0?'+':'−'}${pct(Math.abs(rel))}</strong> (${score(a.value)} → ${score(b.value)} × 10⁻³).<br><span class="contrast">${ciText(contrast('final','S1','S8',group))} This is a combined-system effect, not one isolated intervention.</span>`;
    }else{
      const a=row('ranker','mlp_tt',group),b=row('final','S8',group);
      rows=[a,row('ranker','din_tt',group),row('ranker','din_rrf',group),b];
      text=`Against the original planned two-tower + MLP pipeline, the final system scores <strong>${(b.value/a.value).toFixed(2)}×</strong> as high. This crosses study families and changes multiple components.<br><span class="contrast">No cross-study paired interval is supplied. Mixed-DIN score provenance remains unresolved. A combined established-product macro is not available for the original pipeline, so that group is disabled here.</span>`;
    }
    barChart('#final-chart',rows,{accent:'S8',title:'Test · nDCG@10 × 10⁻³'});
    $('#final-insight').innerHTML=text;
    evidence('final',rows);
    renderEffects('#effects-forest',group);
    if(baseline==='original'){
      const el=$('[data-evidence="final"]');
      el.querySelector('.source-details').insertAdjacentHTML('beforeend',`<p class="fine">Original-pipeline and DIN rows additionally use these RQ3 sources:</p>${D.studies.ranker.sources.map(id=>`<div class="source-entry">${escape(D.sources[id].path)}</div>`).join('')}`);
    }
  }
  function renderEffects(target,group){
    const specs=[['S1','S3','Period-start statistics','Same model · T2 vs T0'],['S1','S6','Mixed-pool system','Retrained mixed vs recency'],['S4','S5','Monthly co-review features','Co-review vs text · monthly'],['S1','S8','Combined system','Mixed / monthly / co-review vs original']];
    V.forest(target,specs.map(([ref,cand,title,referenceText])=>({...contrast('final',ref,cand,group),title,referenceText})),groups[group]);
  }
  function renderAnn(){
    const rows=D.studies.ann.rows;
    barChart('#ann-chart',rows,{accent:'Adaptive HNSW',title:'1,000-query Validation benchmark · nDCG@10 × 10⁻³'});
    const el=$('#ann-chart');el.insertAdjacentHTML('beforeend',`<div class="table-scroll"><table><caption>Measured candidate overlap and end-to-end latency</caption><thead><tr><th>Index</th><th>Overlap vs exact</th><th>Median ms</th><th>nDCG vs exact</th></tr></thead><tbody>${rows.map(r=>`<tr><th>${escape(r.label)}</th><td>${pct(r.overlap,2)}</td><td>${r.latency.toFixed(2)}</td><td>${pct(relative(rows[0].value,r.value))}</td></tr>`).join('')}</tbody></table></div>`);evidence('ann',rows);
  }
  function explorerRows(key,group){
    if(key==='timing')return timingRows(group,'snapshot');
    if(key==='retrieval')return D.studies[key].rows.filter(r=>r.group===group).sort((a,b)=>a.value-b.value);
    return D.studies[key].rows.filter(r=>r.group===group);
  }
  function renderExplorer(resetGroup=false){
    const key=$('#explorer-study').value||state.explorer;state.explorer=key;
    const available=[...new Set(D.studies[key].rows.map(r=>r.group))];
    const previous=resetGroup?'all':$('#explorer-group').value;
    $('#explorer-group').innerHTML=available.map(g=>`<option value="${g}">${escape(groups[g]||g)}</option>`).join('');
    $('#explorer-group').value=available.includes(previous)?previous:available[0];
    const rows=explorerRows(key,$('#explorer-group').value);
    $('#explorer-heading').textContent=D.studies[key].title;
    barChart('#explorer-chart',rows,{percent:key==='retrieval',title:key==='retrieval'?'Recall@200 · per-user macro':'nDCG@10 × 10⁻³ · per-user macro',accent:rows.at(-1).label});
    $('#explorer-extra').innerHTML=(key==='k'?'<p class="explorer-note">Use the guided RQ2 view to see candidate recall beside top-10 quality.</p>':'')+(key==='timing'?'<p class="explorer-note">Use guided R3 to explore the measured lag and feature-swap controls.</p>':'');
    $('#explorer-evidence').innerHTML=evidenceHtml(key,rows);
    if(key==='final'){
      $('#explorer-extra').insertAdjacentHTML('beforeend','<div class="visual-panel" id="explorer-forest"></div>');
      renderEffects('#explorer-forest',$('#explorer-group').value);
    }
    if(key==='retrain'){
      $('#explorer-extra').insertAdjacentHTML('beforeend','<div class="visual-panel" id="explorer-controls-matrix"></div>');
      V.controls('#explorer-controls-matrix',state.step);
    }
    $('#explorer-story-link').href='#'+section[key];
  }
  function setView(view){
    state.view=view;const exploring=view==='explore';
    $('#guide-view').hidden=exploring;$('#explorer-view').hidden=!exploring;
    document.body.classList.toggle('exploring',exploring);
    $('#guide-button').classList.toggle('active',!exploring);$('#guide-button').setAttribute('aria-pressed',!exploring);
    $('#explore-button').classList.toggle('active',exploring);$('#explore-button').setAttribute('aria-pressed',exploring);
  }
  function navigate(){
    const hash=location.hash.slice(1)||'intro';
    if(hash.startsWith('explorer')){
      const requested=hash.split('/')[1];
      if(requested&&D.studies[requested])$('#explorer-study').value=requested;
      setView('explore');renderExplorer();window.scrollTo({top:0,behavior:'instant'});
    }else{
      setView('guide');
      if($('#mobile-chapter').querySelector(`option[value="${CSS.escape(hash)}"]`))$('#mobile-chapter').value=hash;
      const el=document.getElementById(hash);if(el)requestAnimationFrame(()=>el.scrollIntoView({behavior:'instant',block:'start'}));
    }
  }
  let zoom=1;
  function setZoom(z){
    zoom=Math.max(1,Math.min(3,z));$('#poster-image').style.width=(zoom*100)+'%';
    $('#zoom-level').textContent=zoom===1?'Fit':Math.round(zoom*100)+'%';
    $('#zoom-out').disabled=zoom===1;$('#zoom-in').disabled=zoom===3;
  }
  document.addEventListener('click',e=>{
    const method=e.target.closest('[data-method]');if(method){state.method=method.dataset.method;renderMethod();}
    const training=e.target.closest('[data-training-hit]');if(training){trainingHit=training.dataset.trainingHit==='yes';renderMethod();$('#method-scene').querySelector(`[data-training-hit="${trainingHit?'yes':'no'}"]`).focus({preventScroll:true});}
    const stage=e.target.closest('[data-stage]');if(stage)showStage(stage.dataset.stage);
    const k=e.target.closest('[data-k]');if(k){state.k=Number(k.dataset.k);renderK();$('#k-controls').querySelector(`[data-k="${state.k}"]`).focus({preventScroll:true});}
    const step=e.target.closest('[data-step]');if(step){state.step=Number(step.dataset.step);renderRetrain();$('#retrain-controls').querySelector(`[data-step="${state.step}"]`).focus({preventScroll:true});}
    if(e.target.closest('[data-open-poster]')){$('#poster-dialog').showModal();setZoom(1);$('#close-poster').focus();}
  });
  $('#metric-rank').addEventListener('input',renderMetric);
  $('#expansion-range').addEventListener('change',renderExpansion);
  $('#snapshot-example').addEventListener('change',()=>V.reviewTimeline('#review-timeline',Number($('#snapshot-example').value)));
  $('#coreview-cutoff').addEventListener('change',()=>V.coreview('#coreview-visual',Number($('#coreview-cutoff').value)));
  $('#coreview-details').addEventListener('toggle',()=>{if($('#coreview-details').open)V.coreview('#coreview-visual',Number($('#coreview-cutoff').value));});
  $('#mobile-chapter').addEventListener('change',()=>{location.hash=$('#mobile-chapter').value;});
  $('#retrieval-group').addEventListener('change',renderRetrieval);
  $('#ranker-pool').addEventListener('change',renderRanker);$('#ranker-group').addEventListener('change',renderRanker);
  $('#retrain-group').addEventListener('change',renderRetrain);
  $('#timing-group').addEventListener('change',renderTiming);$('#timing-mode').addEventListener('change',renderTiming);
  $('#final-group').addEventListener('change',renderFinal);$('#final-baseline').addEventListener('change',renderFinal);
  $('#ann-details').addEventListener('toggle',()=>{if($('#ann-details').open)renderAnn();});
  $('#explorer-study').innerHTML=Object.entries(D.studies).map(([k,s])=>`<option value="${k}">${escape(s.title)}</option>`).join('');
  $('#explorer-study').addEventListener('change',()=>{history.replaceState(null,'','#explorer/'+$('#explorer-study').value);renderExplorer(true);});
  $('#explorer-group').addEventListener('change',()=>renderExplorer());
  $('#guide-button').addEventListener('click',()=>{location.hash='intro';});$('#explore-button').addEventListener('click',()=>{location.hash='explorer/'+state.explorer;});
  $('#close-poster').addEventListener('click',()=>$('#poster-dialog').close());
  $('#zoom-in').addEventListener('click',()=>setZoom(zoom+.5));$('#zoom-out').addEventListener('click',()=>setZoom(zoom-.5));$('#zoom-fit').addEventListener('click',()=>setZoom(1));
  window.addEventListener('hashchange',navigate);
  const observer=new IntersectionObserver(entries=>{
    if(state.view!=='guide')return;
    for(const entry of entries)if(entry.isIntersecting){$('#mobile-chapter').value=entry.target.id;document.querySelectorAll('#chapter-nav a').forEach(a=>{const selected=a.hash==='#'+entry.target.id;a.classList.toggle('current',selected);if(selected)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}
  },{rootMargin:'-15% 0px -65% 0px'});
  document.querySelectorAll('.chapter').forEach(el=>observer.observe(el));
  showStage('catalogue');renderMethod();renderMetric();renderRetrieval();renderK();renderRanker();renderFrozen();renderRetrain();renderTiming();renderFinal();renderAnn();
  let resizing;window.addEventListener('resize',()=>{clearTimeout(resizing);resizing=setTimeout(()=>{renderRetrieval();renderK();renderRanker();renderRetrain();renderTiming();renderFinal();renderAnn();if($('#coreview-details').open)V.coreview('#coreview-visual',Number($('#coreview-cutoff').value));if(state.view==='explore')renderExplorer();},150);});
  navigate();
})();
