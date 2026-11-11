'use strict';
/* SVG explanations. Illustrations and measured results are labelled separately. */
(() => {
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colour = ['#b4541c','#347673','#8b7f6c'];
  const scaled = n => (n*1000).toFixed(3);
  function confidenceVerdict(c) { return c.ci_low>0?'Positive difference':c.ci_high<0?'Negative difference':'Interval includes zero'; }
  function expansionNet(d) { return d.gained+d.competition; }
  function linearPosition(n,min,max,start,end) { return start+(n-min)/(max-min||1)*(end-start); }
  function host(id) { const el=typeof id==='string'?document.querySelector(id):id;return [el,Math.max(280,Math.min(850,el.clientWidth||600))]; }
  function svgStart(w,h,title) { return `<svg class="chart-svg teaching-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>`; }
  function text(x,y,s,cls='',anchor='start') { return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`; }
  function trend(id,series,{title,xLabel='Candidates K',percent=false,selected=null}={}) {
    const [el,w]=host(id),h=270,left=48,right=20,top=20,bottom=215;
    const points=series.flatMap(s=>s.points),minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x));
    const maxY=Math.max(...points.map(p=>p.value)) *1.15;
    const x=n=>linearPosition(n,minX,maxX,left,w-right),y=n=>linearPosition(n,0,maxY,bottom,top);
    let svg=svgStart(w,h,title);
    for(let i=0;i<=3;i++){const n=maxY*i/3,yy=y(n);svg+=`<line class="gridline" x1="${left}" x2="${w-right}" y1="${yy}" y2="${yy}"/>`+text(left-8,yy+3,percent?(n*100).toFixed(1)+'%':(n*1000).toFixed(1),'axis-label','end');}
    [...new Set(points.map(p=>p.x))].forEach(n=>{svg+=text(x(n),bottom+20,n,'axis-label','middle');});
    if(selected!==null)svg+=`<line x1="${x(selected)}" x2="${x(selected)}" y1="${top}" y2="${bottom}" stroke="#aaa08f" stroke-dasharray="3 4"/>`;
    series.forEach((s,i)=>{
      svg+=`<polyline points="${s.points.map(p=>`${x(p.x)},${y(p.value)}`).join(' ')}" fill="none" stroke="${colour[i]}" stroke-width="2"/>`;
      for(const p of s.points)svg+=`<circle cx="${x(p.x)}" cy="${y(p.value)}" r="${p.x===selected?5.5:3.5}" fill="${colour[i]}" stroke="#fffdf9" stroke-width="1.5"><title>${esc(s.label)} · ${xLabel}=${p.x}: ${p.value}</title></circle>`;
    });
    svg+=text((left+w-right)/2,h-12,xLabel,'axis-label','middle')+'</svg>';
    el.innerHTML=`<p class="chart-title">${esc(title)}</p>${svg}<div class="visual-legend">${series.map((s,i)=>`<span><i style="background:${colour[i]}"></i>${esc(s.label)}</span>`).join('')}</div><p class="fine">Points are measured settings. Straight segments only connect those points; intermediate values were not measured.</p>`;
  }
  function waterfall(id,d) {
    const [el,w]=host(id),h=260,left=42,right=15,top=20,bottom=190;
    const net=expansionNet(d),min=Math.min(0,net)*1.4,max=d.gained*1.3;
    const y=n=>linearPosition(n,min,max,bottom,top),zero=y(0),step=(w-left-right)/3,bar=Math.min(75,step*.55);
    let svg=svgStart(w,h,`Measured K${d.low} to K${d.high} accounting: newly retrieved ${d.gained}, already retrieved ${d.competition}, net ${net}.`);
    for(const n of [...new Set([min,0,max])]){
      if(n!==0&&Math.abs(y(n)-zero)<20)continue;
      svg+=`<line x1="${left}" x2="${w-right}" y1="${y(n)}" y2="${y(n)}" class="gridline"/>`+text(left-5,y(n)+3,(n*1000).toFixed(1),'axis-label','end');
    }
    const entries=[{name:'Newly found',a:0,b:d.gained,value:d.gained},{name:'Already retrieved',a:d.gained,b:net,value:d.competition},{name:'Net change',a:0,b:net,value:net}];
    entries.forEach((r,i)=>{const xx=left+step*i+(step-bar)/2,yy=Math.min(y(r.a),y(r.b));svg+=`<rect x="${xx}" y="${yy}" width="${bar}" height="${Math.abs(y(r.a)-y(r.b))}" fill="${i===0?colour[1]:i===1?colour[2]:colour[0]}"/><title>${esc(r.name)}: ${r.value}</title>`+text(xx+bar/2,yy-8,(r.value>=0?'+':'−')+scaled(Math.abs(r.value)),'value-label','middle');if(w<400){const words=r.name.split(' ');svg+=text(xx+bar/2,216,words[0],'axis-label','middle')+text(xx+bar/2,230,words.slice(1).join(' '),'axis-label','middle');}else svg+=text(xx+bar/2,220,r.name,'axis-label','middle');if(i===0)svg+=`<line x1="${xx+bar}" x2="${left+step+(step-bar)/2}" y1="${y(d.gained)}" y2="${y(d.gained)}" stroke="#8c8375" stroke-dasharray="3 3"/>`;});
    svg+=text(left,250,'Change in nDCG@10 × 10⁻³','axis-label')+'</svg>';
    el.innerHTML=`<p class="chart-title">MEASURED ACCOUNTING · K${d.low} → K${d.high} · VALIDATION</p>${svg}<p class="fine">All products · per-user macro · three-seed mean. The two target groups partition the observed change for the same frozen MLP. This is an accounting identity, not an isolated causal decomposition.</p>`;
  }
  function forest(id,contrasts,groupLabel='All products') {
    const [el,w]=host(id),small=w<550,left=small?20:230,right=25,top=small?45:25,step=small?100:70,h=contrasts.length*step+45;
    const min=Math.min(0,...contrasts.map(c=>c.ci_low)),max=Math.max(0,...contrasts.map(c=>c.ci_high)),pad=(max-min)*.12;
    const x=n=>linearPosition(n,min-pad,max+pad,left,w-right);
    let svg=svgStart(w,h,'Paired differences with 95% user-bootstrap intervals. Different rows have explicitly named reference systems.');
    svg+=`<line x1="${x(0)}" x2="${x(0)}" y1="5" y2="${h-35}" stroke="#8e8374" stroke-dasharray="3 4"/>`;
    contrasts.forEach((c,i)=>{const y=top+i*step,yy=small?y+25:y;
      svg+=text(small?left:0,small?y-4:y-6,c.title,'forest-title')+text(small?left:0,small?y+10:y+12,c.referenceText,'axis-label');
      svg+=`<line x1="${x(c.ci_low)}" x2="${x(c.ci_high)}" y1="${yy}" y2="${yy}" stroke="${colour[0]}" stroke-width="2.5"/><line x1="${x(c.ci_low)}" x2="${x(c.ci_low)}" y1="${yy-5}" y2="${yy+5}" stroke="${colour[0]}"/><line x1="${x(c.ci_high)}" x2="${x(c.ci_high)}" y1="${yy-5}" y2="${yy+5}" stroke="${colour[0]}"/><circle cx="${x(c.diff)}" cy="${yy}" r="4" fill="${colour[0]}"><title>${esc(c.title)}: ${c.diff}, 95% CI [${c.ci_low}, ${c.ci_high}]. ${confidenceVerdict(c)}</title></circle>`;
      svg+=text(small?left:w-right,small?yy+21:yy+26,`${c.diff>=0?'+':'−'}${scaled(Math.abs(c.diff))} [${scaled(c.ci_low)}, ${scaled(c.ci_high)}]`,'axis-label',small?'start':'end');
    });
    for(const n of [...new Set([min,max,0])])svg+=text(x(n),h-16,(n*1000).toFixed(1),'axis-label','middle');
    el.innerHTML=`<p class="chart-title">PAIRED DIFFERENCE · RESIDUAL STUDY · TEST · ${esc(groupLabel)}</p>${svg}</svg><p class="fine">nDCG@10 × 10⁻³ · per-user macro · three-seed means. Dot: average difference. Line: paired 95% interval from 1,000 user resamples. Crossing zero means the direction is uncertain under this comparison; it does not establish equivalence. Rows have different controls and must not be added together.</p>`;
  }
  function pool(id,data) {
    const [el,w]=host(id),c=data.composition;
    el.innerHTML=`<div class="mix-sources"><div><span class="source-disc recency-disc">200</span><b>Recency list</b></div><div class="overlap-label"><b>${data.overlap.toFixed(1)}</b><small>shared on average</small><span>⇢</span></div><div><span class="source-disc tower-disc">200</span><b>Two-tower list</b></div></div><div class="mix-selection"><span>Alternate sources</span><span>→ Skip duplicates</span><span>→ Stop at 200</span></div><h3>The resulting 200-item mixed pool</h3><div class="composition-bar" role="img" aria-label="Average mixed-pool composition: ${c.recency_only} recency-only, ${c.both_sources} shared, ${c.two_tower_only} two-tower-only products."><span style="width:${c.recency_only/2}%" class="recency-fill"></span><span style="width:${c.both_sources/2}%" class="shared-fill"></span><span style="width:${c.two_tower_only/2}%" class="tower-fill"></span></div><div class="composition-labels"><span><i class="recency-fill"></i>${c.recency_only.toFixed(2)} recency only</span><span><i class="shared-fill"></i>${c.both_sources.toFixed(2)} shared</span><span><i class="tower-fill"></i>${c.two_tower_only.toFixed(2)} two-tower only</span></div><p class="fine">Measured full-Validation averages; fractional counts are averages across lists. The original lists share ${data.overlap.toFixed(1)} items on average, while ${c.both_sources.toFixed(2)} shared items survive selection into the mixed pool. Diagram is schematic; circle area does not encode set size or overlap.</p>`;
  }
  function controls(id,selected) {
    const [el]=host(id),rows=[['Recency','Recency','Original fit','20,676'],['Recency','Mixed','Reuse original','20,676'],['Mixed','Mixed','New fit','20,676'],['Mixed','Mixed','New fit','38,861']];
    el.innerHTML=`<div class="eyebrow">READ THE EXPERIMENT · VALIDATION</div><div class="table-scroll"><table class="control-matrix"><caption>What changed between the four conditions?</caption><thead><tr><th>Condition</th><th>Training lists</th><th>Scoring lists</th><th>Model weights</th><th>Training queries</th></tr></thead><tbody>${rows.map((r,i)=>`<tr class="${i===selected?'active-row':''}"><th>${i+1}</th>${r.map((v,j)=>`<td class="${i>0&&v!==rows[0][j]?'changed-cell':''}">${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="fine">Highlighted cells differ from condition 1. A training query is “natural-positive” when its next-reviewed target is already in the shortlist. Same-size retraining controls the example count, not every feature-scaling, exposure or optimisation difference.</p>`;
  }
  function reviewTimeline(id,cut) {
    const [el,w]=host(id),events=[{year:2020,name:'Snapshot T0'},{year:2021,name:'First review'},{year:2022,name:'Next review'}];
    const h=150,left=28,right=w-28,x=n=>linearPosition(n,2020,2022,left,right);
    let svg=svgStart(w,h,'Conceptual review timeline. A product first reviewed in 2021 has no known reviews in a 2020 snapshot; a 2022 snapshot includes its earlier review.');
    svg+=`<line x1="${left}" x2="${right}" y1="65" y2="65" stroke="#b7ae9d" stroke-width="2"/>`;
    events.forEach((r,i)=>{const xx=x(r.year);svg+=`<circle cx="${xx}" cy="65" r="6" fill="${i===1?colour[0]:colour[2]}"/>`+text(xx,40,r.year,'axis-label',i===0?'start':i===2?'end':'middle')+text(xx,95,r.name,'axis-label',i===0?'start':i===2?'end':'middle');});
    const xx=x(cut===0?2020:2022);svg+=`<line x1="${xx}" x2="${xx}" y1="15" y2="115" stroke="${colour[0]}" stroke-dasharray="4 3"/>`;
    el.innerHTML=`<p class="chart-title">CONCEPTUAL EXAMPLE · NOT AN ACTUAL PRODUCT</p>${svg}</svg><div class="snapshot-reading"><b>${cut===0?'2020 snapshot: 0 known reviews':'2022 snapshot: earlier review is known'}</b><span>${cut===0?'The model can read “no reviews yet” even though the product has since received a review.':'The same candidate now has historical support. We changed the information cutoff, not the model weights.'}</span></div>`;
  }
  function coreview(id,cut) {
    const [el,w]=host(id),h=210,left=40,right=w-40,mid=w/2;
    let svg=svgStart(w,h,'Conceptual co-review graph: two past reviewers connect a history product and a candidate. Only reviews before the cutoff contribute.');
    const ys=[60,150];
    ys.forEach((y,i)=>{const active=i===0||cut===1;svg+=`<path d="M${left+18} 105L${mid-15} ${y}M${mid+15} ${y}L${right-18} 105" stroke="${active?colour[0]:'#c8c0b3'}" stroke-width="2" ${active?'':'stroke-dasharray="5 4"'}/><circle cx="${mid}" cy="${y}" r="15" fill="${active?'#f2dcc6':'#e7e1d7'}" stroke="${active?colour[0]:'#bdb4a5'}"/>`+text(mid,y+4,i===0?'A':'B','','middle')+text(mid,y+(i===0?-24:33),i===0?'Reviews before 2020':'Reviews during 2021','axis-label','middle');});
    svg+=`<rect x="${left-17}" y="86" width="34" height="38" rx="4" fill="#347673"/><rect x="${right-17}" y="86" width="34" height="38" rx="4" fill="#b4541c"/>`+text(left,145,'History','axis-label','middle')+text(right,145,'Candidate','axis-label','middle');
    el.innerHTML=`<p class="chart-title">CONCEPTUAL GRAPH · LETTERS REPRESENT OTHER REVIEWERS</p>${svg}</svg><p class="snapshot-reading"><b>${cut===0?'1':'2'} shared reviewer${cut===0?'':'s'} before this cutoff</b><span>Solid edges are counted; dashed edges occur later and are excluded. This explains a support signal, not an actual model attribution or a claim of buying together.</span></p>`;
  }
  const API={trend,waterfall,forest,pool,controls,reviewTimeline,coreview,confidenceVerdict,expansionNet,linearPosition};
  globalThis.CaveVisuals=API;
  if(typeof module!=='undefined')module.exports=API;
})();
