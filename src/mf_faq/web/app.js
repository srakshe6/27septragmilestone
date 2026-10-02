/* MF facts assistant - client.
   No framework, no build step, no network beyond our own /ask.
   Every value that came from the server or the user goes through esc() before
   it reaches innerHTML. That is the whole XSS story: the answer text is
   model- or template-generated, the query is user-typed, and the citation is a
   URL that ends up in an href. */
'use strict';

const log = document.getElementById('log');
const form = document.getElementById('f');
const box = document.getElementById('q');
const sendBtn = form.querySelector('button[type=submit]');
const clearBtn = document.getElementById('clear');
let busy = false;

/* ------------------------------------------------------------------ helpers */
function esc(s){
  return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
// esc() stops a value breaking out of href="...", but it does not stop a
// javascript: URL being navigated to. Citations are allow-listed server-side,
// so this is defence in depth rather than a live hole.
function safeUrl(u){
  return /^(https?:\/\/|\/)/i.test(String(u||''))?String(u):'';
}
function num(n){
  return Number.isFinite(+n)?String(+n):'';
}
// The fusion score is unbounded - a confident TER match scores ~1.39 - so it is
// shown as the raw number rather than dressed up as a 0-1 probability.
function confClass(c){
  c = +c;
  if (!Number.isFinite(c) || c <= 0) return '';
  return c >= 1 ? 'ok' : (c >= 0.8 ? '' : 'warnt');
}

/* ------------------------------------------------------------------ examples */
document.querySelectorAll('.ex').forEach(b=>b.onclick=()=>{
  box.value=b.textContent;
  box.focus();
  form.requestSubmit();
});

/* ------------------------------------------------------------------ scope */
(function scope(){
  const dl = document.getElementById('scope');
  if (!dl) return;
  const rows = [
    ['Schemes', '5 HDFC direct-growth'],
    ['Facts', 'expense ratio, NAV, AUM'],
    ['', 'fund manager, exit load'],
    ['', 'SIP / lumpsum minimums'],
    ['', 'benchmark, lock-in, ISIN'],
    ['', 'launch date, RTA'],
    ['Not covered', 'returns, holdings'],
    ['', 'sector or market-cap split'],
    ['Refused', 'advice, PAN, email, folio']
  ];
  dl.innerHTML = rows.map(([k,v])=>'<div>'+(k?'<b>'+esc(k)+'</b>':'')+'<span>'+esc(v)+'</span></div>').join('');
})();

/* ------------------------------------------------------------------ render */
function turn(cls, html){
  const el = document.createElement('div');
  el.className = 'turn '+cls;
  el.innerHTML = html;
  log.appendChild(el);
  el.scrollIntoView({behavior:'smooth',block:'nearest'});
  return el;
}

function evidenceHtml(ev){
  if (!ev || !ev.length) return '';
  const items = ev.map(e=>{
    const kind = e.kind ? '<span class="tag">'+esc(e.kind)+'</span> ' : '';
    const sc = num(e.score);
    return '<li>'+kind+esc(e.text||'')+(sc?' <span class="score">('+esc(sc)+')</span>':'')+'</li>';
  }).join('');
  return '<details class="srcs"><summary>Evidence - '+ev.length+
         ' retrieved chunk'+(ev.length===1?'':'s')+'</summary><ul>'+items+'</ul></details>';
}

function metaHtml(d){
  const bits = [];
  // Two separate dates, never one substituted for the other: `prepared_on` is
  // when this answer was produced, `last_updated` is what the source said.
  // Printing only the source date made a week-old answer read as current.
  bits.push('<span><span class="k">Answer prepared on:</span> '+esc(d.prepared_on||'-')+'</span>');
  if (d.freshness_from_source && d.last_updated){
    bits.push('<span><span class="k">Last updated from sources:</span> '+esc(d.last_updated)+'</span>');
  }
  bits.push('<span class="tag stage">'+esc(d.stage||'-')+'</span>');
  bits.push('<span class="tag '+confClass(d.confidence)+'">confidence '+esc(num(d.confidence)||'0')+'</span>');
  if (d.timings_ms) bits.push('<span><span class="k">'+esc(num(d.timings_ms.total_ms))+' ms</span></span>');
  if (d.guards && d.guards.length){
    bits.push('<span>'+d.guards.map(g=>'<span class="tag">'+esc(g)+'</span>').join(' ')+'</span>');
  }
  if (d.citation){
    const u = safeUrl(d.citation);
    bits.push('<span class="src">Source: '+
      (u?'<a href="'+esc(u)+'" target="_blank" rel="noopener">'+esc(d.citation)+'</a>'
         :esc(d.citation))+'</span>');
  }
  return '<div class="meta">'+bits.join('')+'</div>';
}

function answerHtml(d){
  const refusal = d.kind==='answer'?'':'refusal';
  // esc() already maps null/undefined to '', so the values go in unguarded --
  // a redundant `|| ''` here would only obscure the one place XSS is handled.
  const body = '<div class="body"><p class="ans '+refusal+'">'+esc(d.text)+'</p>'+
               evidenceHtml(d.evidence)+'</div>';
  return body + metaHtml(d);
}

function errorHtml(msg){
  return '<div class="body"><p class="ans refusal">'+esc(msg)+
         '</p><p class="mut">The server did not return an answer. Check that it is still '+
         'running, then try again.</p></div>';
}

/* ------------------------------------------------------------------ ask */
async function ask(text){
  if (busy) return;
  const query = String(text||'').trim();
  if (!query) return;
  busy = true;
  sendBtn.disabled = true;
  box.value = '';

  turn('q','<div class="body"><p class="qtext">'+esc(query)+'</p></div>');
  const pending = turn('a','<div class="thinking">Looking up sources&hellip;</div>');

  let d;
  try{
    const r = await fetch('/ask',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q:query})
    });
    if (!r.ok && r.status >= 500){
      let detail = '';
      try{ detail = (await r.json()).error || ''; }catch(_){}
      throw new Error(detail || ('server error '+r.status));
    }
    d = await r.json();
  }catch(err){
    pending.innerHTML = errorHtml('Could not get an answer: '+err.message);
    busy = false;
    sendBtn.disabled = false;
    box.focus();
    return;
  }
  pending.innerHTML = answerHtml(d);
  busy = false;
  sendBtn.disabled = false;
  box.focus();
}

form.onsubmit = e=>{ e.preventDefault(); ask(box.value); };

// Enter sends, Shift+Enter inserts a newline. Growing the textarea with its
// content keeps a long question from being a one-line scroller.
box.addEventListener('keydown', e=>{
  if (e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    form.requestSubmit();
  }
});
box.addEventListener('input', ()=>{
  box.style.height='auto';
  box.style.height=Math.min(box.scrollHeight, window.innerHeight*0.4)+'px';
});

clearBtn.onclick = ()=>{
  log.innerHTML='';
  box.value='';
  box.focus();
};
