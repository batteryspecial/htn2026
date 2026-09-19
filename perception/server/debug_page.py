"""A dev-only page for driving the pipeline before anything else exists.

Not the operator UI. That lives in frontend/ and is Kevin's. This is here so
the service can be watched and retasked from a browser with no orchestrator,
no controller and no car, which is also what makes a solo end-to-end test
possible.
"""

DEBUG_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>perception</title>
<style>
 :root { color-scheme: dark; --bg:#111; --fg:#e8e8e8; --dim:#888; --line:#2a2a2a; }
 body { background:var(--bg); color:var(--fg); font:14px ui-monospace,SFMono-Regular,Menlo,monospace;
        margin:0; padding:16px; display:grid; gap:16px; grid-template-columns:minmax(0,2fr) minmax(280px,1fr); }
 @media (max-width:800px){ body{grid-template-columns:1fr} }
 img { width:100%; border:1px solid var(--line); border-radius:6px; display:block; }
 section { border:1px solid var(--line); border-radius:6px; padding:12px; }
 h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim); margin:0 0 8px; }
 textarea { width:100%; height:150px; background:#0b0b0b; color:var(--fg); border:1px solid var(--line);
            border-radius:4px; padding:8px; font:inherit; box-sizing:border-box; }
 button { background:#1d1d1d; color:var(--fg); border:1px solid var(--line); border-radius:4px;
          padding:7px 12px; font:inherit; cursor:pointer; }
 button:hover { background:#262626; }
 #phase { font-size:22px; font-weight:600; }
 #log { height:190px; overflow:auto; white-space:pre-wrap; color:var(--dim); font-size:12px; }
 .row { display:flex; gap:8px; margin-top:8px; }
 table { width:100%; border-collapse:collapse; } td { padding:2px 0; }
 td:first-child { color:var(--dim); width:45%; }
</style></head><body>
<div><img src="/video" alt="annotated feed"></div>
<div style="display:grid;gap:16px;align-content:start">
  <section><h2>phase</h2><div id="phase">—</div>
    <table id="state"></table></section>
  <section><h2>send a spec</h2>
    <textarea id="spec"></textarea>
    <div class="row"><button onclick="send()">POST /spec</button>
      <button onclick="clearSpec()">DELETE /spec</button>
      <button onclick="loadModels()">GET /models</button></div></section>
  <section><h2>events</h2><div id="log"></div></section>
</div>
<script>
const $ = id => document.getElementById(id);
$('spec').value = JSON.stringify({instruction_id:"dev-1", spec:{spec_id:"s1", model:"yoloe",
  mode:"follow", targets:[{ref:"t1", detect:["person"], select:"largest"}]}}, null, 2);

function log(line){ const l=$('log'); l.textContent += line+"\\n"; l.scrollTop = l.scrollHeight; }

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
function connect(path, onmsg){
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.onmessage = e => onmsg(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => connect(path, onmsg), 1000);
}
connect('/ws/target', s => {
  $('phase').textContent = s.phase;
  $('phase').style.color = s.phase === 'TRACKING' ? '#5c5' : s.phase === 'FAULT' ? '#e55' : '#ca5';
  $('state').innerHTML = Object.entries({visible:s.visible, label:s.label, track:s.track_id,
    cx:s.cx?.toFixed(3), cy:s.cy?.toFixed(3), area:s.area?.toFixed(4), conf:s.conf?.toFixed(2),
    spec:s.spec_id, model:s.model, target:s.target_ref})
    .map(([k,v]) => `<tr><td>${k}</td><td>${v ?? '—'}</td></tr>`).join('');
});
connect('/ws/events', e => log(`${e.stage}${e.detail ? ' — '+e.detail : ''}`));

async function post(url, body){
  const r = await fetch(url, {method: body?'POST':'DELETE',
    headers:{'content-type':'application/json'}, body: body||undefined});
  log(`${r.status} ${url} ${JSON.stringify(await r.json())}`);
}
const send = () => post('/spec', $('spec').value);
const clearSpec = () => post('/spec', null);
const loadModels = async () => {
  const r = await fetch('/models');
  const d = await r.json();
  log('models: ' + d.models.map(m =>
    `${m.name}${m.loaded?'':'(not loaded)'}${m.available?'':' UNAVAILABLE'}`).join(', '));
};
</script></body></html>
"""
