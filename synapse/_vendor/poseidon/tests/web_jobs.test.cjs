/* Run the shipped workbench in a small DOM with controlled HTTP responses.
 * No model, checkpoint, browser install, or background experiment is needed. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {execFileSync} = require('node:child_process');
const {test} = require('node:test');

const web = path.join(__dirname, '..', 'poseidon', 'web');
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const jobId = 'a'.repeat(32);
const receipt = {schema: 'poseidon-mco-v1', experiment_id: 'restored-receipt', summary: {}, paired: {}, limits: ['Synthetic benchmark only.']};
const listedJob = state => ({id: jobId, kind: 'mco-experiment', state, settings: {},
  created_at: '2026-10-21T12:00:00+00:00', finished_at: state === 'completed' ? '2026-10-21T12:01:00+00:00' : null,
  progress: {}, error: null});

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this.style = {};
    this.value = ''; this.textContent = ''; this.hidden = false; this.disabled = false;
    const classes = new Set();
    this.classList = {add: value => classes.add(value), remove: value => classes.delete(value),
      contains: value => classes.has(value), toggle: (value, on) => on ? classes.add(value) : classes.delete(value)};
  }
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children = items; this.textContent = '';}
  setAttribute(key, value) {this.attributes[key] = String(value);}
  removeAttribute(key) {delete this.attributes[key];}
  addEventListener() {}
  querySelectorAll() {return [];}
  checkValidity() {return true;}
  reportValidity() {return true;}
  getContext() {return new Proxy({}, {get: () => () => {}});}
  scrollIntoView() {}
}

async function workbench(route) {
  const elements = new Map([...html.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"/g)].map(match => [match[2], new Element(match[1])]));
  for (const match of html.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>([^<]*)<\/\1>/g)) {
    elements.get(match[2]).textContent = match[3];
  }
  const calls = [], timers = new Map(), downloads=[];
  let timerId = 0;
  const document = {
    querySelector(selector) {return selector.startsWith('#') ? elements.get(selector.slice(1)) || null : null;},
    querySelectorAll() {return [];}, createElement: tag => new Element(tag),
  };
  elements.get('experimentType').value = 'cognitive';
  elements.get('experimentSeed').value = '90000000';
  elements.get('episodes').value = '1'; elements.get('maxSteps').value = '32'; elements.get('scarcity').value = '2.5';
  elements.get('experimentForm').querySelectorAll = () => ['experimentType', 'experimentSeed', 'episodes', 'maxSteps', 'scarcity'].map(id => elements.get(id));
  class BrowserURL extends URL {}
  BrowserURL.createObjectURL=blob=>{downloads.push(blob);return 'blob:test';};
  BrowserURL.revokeObjectURL=()=>{};
  Element.prototype.click=function(){};
  const context = vm.createContext({document, console, URL:BrowserURL, Blob, setTimeout:fn=>{fn();return 0;}, location: {href: 'http://127.0.0.1:8787/'},
    matchMedia: () => ({matches: true}), requestAnimationFrame: () => 0, cancelAnimationFrame() {},
    setInterval(fn, ms) {const id = ++timerId; timers.set(id, {fn, ms}); return id;},
    clearInterval(id) {timers.delete(id);},
    async fetch(url, options = {}) {
      const call = {url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : undefined};
      calls.push(call);
      const response = await route(call);
      return {ok: response?.status >= 200 && response.status < 300, status: response?.status || 404,
        statusText: response?.statusText || 'Unavailable', json: async () => response?.data || {error: 'Unavailable'},
        text:async()=>response?.raw || JSON.stringify(response?.data || {error:'Unavailable'})};
    },
  });
  vm.runInContext(source, context, {filename: 'app.js'});
  const flush = async () => {for (let i = 0; i < 20; i++) await Promise.resolve();};
  await flush();
  return {elements, calls, context, flush, downloads,
    async tick() {for (const {fn, ms} of [...timers.values()]) if (ms < 15000) await fn(); await flush();},
    buttons() {return elements.get('jobHistory').children.flatMap(item => item.children).filter(item => item.tagName === 'BUTTON');},
  };
}

test('generated reasoning audit displays undefined risk for zero accepted answers', async () => {
  const metrics={total:14,correct:0,wrong:0,errors:0,correct_refusals:8,missed_answerable:6,coverage:0,selective_risk:null};
  const ui=await workbench(({url,body})=>url==='/api/reasoning/audit'?{status:200,data:{
    config:body,metrics,by_family:{unsupported:metrics},report_sha256:'abc',source_hashes:{},limits:[],rows:[]}}:null);
  await ui.elements.get('runReasoningAudit').onclick();await ui.flush();
  assert.equal(ui.elements.get('reasoningAuditResults').hidden,false);
  assert.match(ui.elements.get('reasoningAuditFamilies').children[0].children[5].textContent,/Undefined/);
  assert.match(ui.elements.get('reasoningAuditStatus').textContent,/6 missed answerable/);
  assert.match(ui.elements.get('reasoningAuditScope').textContent,/No calibrated risk guarantee/);
  assert.deepEqual(ui.calls.find(call=>call.url==='/api/reasoning/audit').body,{seed:20261030,samples:8});
  assert.equal(ui.elements.get('runReasoningAudit').disabled,false);
});

test('reasoning audit errors hide stale results and allow retry', async () => {
  const ui=await workbench(({url})=>url==='/api/reasoning/audit'?{status:409,data:{error:'Restart needed'}}:null);
  ui.elements.get('reasoningAuditResults').hidden=false;
  await ui.elements.get('runReasoningAudit').onclick();await ui.flush();
  assert.equal(ui.elements.get('reasoningAuditResults').hidden,true);
  assert.match(ui.elements.get('reasoningAuditStatus').textContent,/Restart needed/);
  assert.equal(ui.elements.get('runReasoningAudit').disabled,false);
});

test('reasoning audit validates the requested budget before sending', async () => {
  const ui=await workbench(()=>null);
  ui.elements.get('reasoningAuditSamples').value='33';
  await ui.elements.get('runReasoningAudit').onclick();await ui.flush();
  assert.equal(ui.calls.filter(call=>call.url==='/api/reasoning/audit').length,0);
  assert.match(ui.elements.get('reasoningAuditStatus').textContent,/1–32/);
});

test('metamorphic audit separates consistent relations from correct answers', async () => {
  const metrics={groups:5,total:10,accepted:10,correct:0,wrong:10,errors:0,refusals:0,coverage:1,selective_risk:1,relation_checked:5,relation_passed:5,relation_violations:0,relation_unavailable:0,joint_correct_groups:0};
  const ui=await workbench(({url,body})=>url==='/api/reasoning/metamorphic-audit'?{status:200,data:{config:body,metrics,by_family:{arithmetic:metrics},groups:[],limits:[]}}:null);
  await ui.elements.get('runMetamorphicAudit').onclick();await ui.flush();
  assert.equal(ui.elements.get('metamorphicAuditResults').hidden,false);
  assert.match(ui.elements.get('metamorphicAuditStatus').textContent,/0.*5.*correct/);
  assert.match(ui.elements.get('metamorphicAuditScope').textContent,/both be wrong/);
  assert.equal(ui.elements.get('runMetamorphicAudit').disabled,false);
});

test('episode risk upload errors clear old results and permit retry', async () => {
  const ui=await workbench(()=>null);
  ui.elements.get('episodeRiskResults').hidden=false;
  ui.elements.get('episodeRiskFile').files=[];
  await ui.elements.get('runEpisodeRisk').onclick();await ui.flush();
  assert.equal(ui.elements.get('episodeRiskResults').hidden,true);
  assert.match(ui.elements.get('episodeRiskStatus').textContent,/Choose/);
  assert.equal(ui.elements.get('runEpisodeRisk').disabled,false);
});

test('episode risk displays per-condition bounds without controller promotion', async () => {
  const ui=await workbench(({url})=>url==='/api/research/episode-risk'?{status:200,data:{eligible:false,conditions:{scarcity:{episodes:10,adverse_episodes:0,observed_risk:0,risk_upper_bound:0.31,passed:false}},protocol:{max_episode_risk:0.1,confidence:0.95}}}:null);
  ui.elements.get('episodeRiskFile').files=[{size:100,text:async()=>'{"protocol":{},"outcomes_by_condition":{}}'}];
  await ui.elements.get('runEpisodeRisk').onclick();await ui.flush();
  assert.equal(ui.elements.get('episodeRiskResults').hidden,false);
  assert.match(ui.elements.get('episodeRiskStatus').textContent,/Insufficient/);
  assert.match(ui.elements.get('episodeRiskScope').textContent,/independent/);
  assert.match(ui.elements.get('episodeRiskScope').textContent,/per-action/);
});

test('audit export preserves original float forms and 64-bit episode identifiers', async () => {
  const raw='{"eligible":true,"conditions":{"id":{"episodes":1,"adverse_episodes":0,"observed_risk":0.0,"risk_upper_bound":0.5,"passed":true}},"protocol":{"max_episode_risk":0.5,"confidence":0.95,"calibration_episode_ids":[9007199254740993]}}';
  const ui=await workbench(({url})=>url==='/api/research/episode-risk'?{status:200,raw}:null);
  ui.elements.get('episodeRiskFile').files=[{size:2,text:async()=>'{}'}];
  await ui.elements.get('runEpisodeRisk').onclick();await ui.flush();
  ui.elements.get('exportEpisodeRisk').onclick();
  assert.equal(await ui.downloads[0].text(),raw+'\n');
  assert.match(ui.elements.get('episodeRiskDetails').textContent,/9007199254740993/);
});

test('completed history fetches its full result only when opened after reload', async () => {
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('completed')]}}
    : url === '/api/jobs/' + jobId ? {status: 200, data: {...listedJob('completed'), result: receipt}} : null);
  assert.equal(ui.calls.filter(call => call.url === '/api/jobs/' + jobId).length, 0, 'history must remain lightweight');
  const button = ui.buttons().find(item => /view result/i.test(item.textContent));
  assert.ok(button, 'completed history must let the user reopen its result');
  await button.onclick(); await ui.flush();
  assert.match(ui.elements.get('experimentIdentity').textContent, /restored-receipt/);
  assert.equal(ui.elements.get('experimentResults').hidden, false);
});

test('running history reconnects monitoring without submitting duplicate work', async () => {
  let complete = false;
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('running')]}}
    : url === '/api/jobs/' + jobId ? {status: 200, data: {...listedJob(complete ? 'completed' : 'running'), result: complete ? receipt : null}} : null);
  const button = ui.buttons().find(item => /monitor|reconnect|resume/i.test(item.textContent));
  assert.ok(button, 'a running job must be attachable after reload');
  await button.onclick(); await ui.flush();
  assert.equal(ui.elements.get('send').disabled, true);
  assert.equal(ui.elements.get('cancelJob').hidden, false);
  complete = true; await ui.tick();
  assert.match(ui.elements.get('experimentIdentity').textContent, /restored-receipt/);
  assert.equal(ui.elements.get('send').disabled, false);
  assert.equal(ui.calls.filter(call => call.method === 'POST').length, 0);
});

test('MCO submission follows the kind-only server contract', async () => {
  let accepted = false;
  const ui = await workbench(call => {
    if (call.url === '/api/jobs' && call.method === 'GET') return {status: 200, data: {jobs: []}};
    if (call.url === '/api/jobs' && call.method === 'POST') {
      accepted = JSON.stringify(call.body) === '{"kind":"mco-experiment"}';
      return accepted ? {status: 202, data: listedJob('queued')} : {status: 400, data: {error: 'Unknown job setting.'}};
    }
    if (call.url === '/api/jobs/' + jobId) return {status: 200, data: {...listedJob('completed'), result: receipt}};
    return null;
  });
  ui.elements.get('experimentType').value = 'mco';
  await ui.elements.get('experimentForm').onsubmit({preventDefault() {}}); await ui.tick();
  assert.ok(accepted, 'the actual form must submit only the accepted MCO kind');
  assert.match(ui.elements.get('experimentIdentity').textContent, /restored-receipt/);
  assert.equal(ui.elements.get('send').disabled, false);
});

test('status remains connected with optional candidate cards absent and keeps Atlas unavailable', async () => {
  const ui = await workbench(({url}) => url === '/api/status' ? {status: 200, data: {
    core_ready: false, reports: {}, atlas: {ready: false}, contrast: {ready: false}, horizon: {ready: false},
    odyssey: {ready: false}, helm: {ready: false}, odysseus: {ready: false}, cognitive: {ready: true},
    aura: {ready: true}, tessera: {ready: true}, metamorph: {ready: true}, chimera: {ready: true},
    hyperion: {ready: true}, titan: {ready: true}, trident: {ready: true}, mnemorph: {ready: false},
  }} : url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
  assert.match(ui.elements.get('connection').textContent, /Connected/);
  ui.elements.get('experimentType').value = 'atlas';
  ui.elements.get('experimentType').onchange();
  assert.equal(ui.elements.get('runExperiment').disabled, true, 'available composites must not mark missing Atlas fitted');
});

for (const protocol of ['tessera', 'mnemorph']) test(protocol + ' uses its fixed benchmark contract', async () => {
  let accepted = false;
  const ui = await workbench(call => {
    if (call.url === '/api/jobs' && call.method === 'GET') return {status: 200, data: {jobs: []}};
    if (call.url === '/api/jobs' && call.method === 'POST') {
      accepted = JSON.stringify(call.body) === JSON.stringify({kind: protocol + '-experiment'});
      return accepted ? {status: 202, data: listedJob('queued')} : {status: 400, data: {error: 'Unknown job setting.'}};
    }
    if (call.url === '/api/jobs/' + jobId) return {status: 200, data: {...listedJob('completed'), result: receipt}};
    return null;
  });
  ui.elements.get('experimentType').value = protocol;
  ui.elements.get('experimentType').onchange();
  assert.equal(ui.elements.get('maxSteps').disabled, true, 'fixed benchmark controls must not imply adjustable settings');
  await ui.elements.get('experimentForm').onsubmit({preventDefault() {}}); await ui.tick();
  assert.ok(accepted);
  assert.match(ui.elements.get('experimentIdentity').textContent, /restored-receipt/);
});

test('METAMORPH uses a cancellable durable job', async () => {
  const ui = await workbench(call => {
    if (call.url === '/api/status') return {status: 200, data: {core_ready: true, metamorph: {ready: true}}};
    if (call.url === '/api/jobs' && call.method === 'GET') return {status: 200, data: {jobs: []}};
    if (call.url === '/api/jobs' && call.method === 'POST' && call.body.kind === 'metamorph-experiment') return {status: 202, data: listedJob('queued')};
    if (call.url === '/api/jobs/' + jobId) return {status: 200, data: {...listedJob('completed'), result: receipt}};
    return null;
  });
  ui.elements.get('experimentType').value = 'metamorph';
  await ui.elements.get('experimentForm').onsubmit({preventDefault() {}}); await ui.tick();
  assert.match(ui.elements.get('experimentIdentity').textContent, /restored-receipt/);
  assert.equal(ui.calls.some(call => call.url === '/api/metamorph-experiment'), false);
});

test('unsupported paired protocol never falls through to an Atlas run', async () => {
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
  vm.runInContext("coreReady=true;atlasReady=true", ui.context);
  ui.elements.get('experimentType').value = 'nexus_omega';
  ui.elements.get('experimentType').onchange();
  assert.equal(ui.elements.get('runExperiment').disabled, true);
  await ui.elements.get('experimentForm').onsubmit({preventDefault() {}});
  assert.equal(ui.calls.filter(call => call.method === 'POST').length, 0);
});

test('cognitive availability follows returned readiness', async () => {
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
  vm.runInContext("setCandidateStatus('cognitive',{ready:false,error:'No core checkpoint'});updateProtocol()", ui.context);
  assert.equal(ui.elements.get('runExperiment').disabled, true);
  assert.match(ui.elements.get('cognitiveNote').textContent, /unavailable|No core checkpoint/i);
});

test('polling reports temporary API errors and recovers to a terminal failure', async () => {
  let status = 503;
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('running')]}}
    : url === '/api/jobs/' + jobId ? status === 503 ? {status: 503, data: {error: 'Temporary maintenance'}}
      : {status: 200, data: {...listedJob('failed'), error: {type: 'JobTimeout', message: 'The absolute experiment deadline elapsed'}}} : null);
  await ui.buttons().find(item => /monitor/i.test(item.textContent)).onclick();
  assert.match(ui.elements.get('jobStatus').textContent, /Temporary maintenance/);
  assert.equal(ui.elements.get('send').disabled, true, 'a retry must retain the selected job');
  status = 200; await ui.tick();
  assert.equal(ui.elements.get('send').disabled, false);
  assert.equal(ui.elements.get('cancelJob').hidden, true);
  assert.match(ui.elements.get('experimentStatus').textContent, /deadline elapsed/);
  assert.equal(ui.elements.get('experimentStatus').classList.contains('error'), true);
});

test('cancellation errors remain visible and a successful retry releases the controls', async () => {
  let canCancel = false;
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('queued')]}}
    : url === '/api/jobs/' + jobId ? {status: 200, data: listedJob('queued')}
    : url === '/api/jobs/' + jobId + '/cancel' ? canCancel ? {status: 200, data: {...listedJob('cancelled'), error: {type: 'JobCancelled', message: 'Queued job cancelled'}}}
      : {status: 409, data: {error: 'Cancellation temporarily unavailable'}} : null);
  await ui.buttons().find(item => /monitor/i.test(item.textContent)).onclick();
  await ui.elements.get('cancelJob').onclick();
  assert.match(ui.elements.get('jobStatus').textContent, /Cancellation temporarily unavailable/);
  assert.equal(ui.elements.get('cancelJob').disabled, false);
  assert.equal(ui.elements.get('send').disabled, true);
  canCancel = true; await ui.elements.get('cancelJob').onclick();
  assert.equal(ui.elements.get('send').disabled, false);
  assert.equal(ui.elements.get('cancelJob').hidden, true);
  assert.match(ui.elements.get('experimentStatus').textContent, /cancelled/);
  assert.equal(ui.elements.get('experimentSummary').children.length, 0, 'cancelled work cannot be shown as completed evidence');
});

test('a disappeared job ends monitoring and leaves a useful error', async () => {
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('running')]}}
    : url === '/api/jobs/' + jobId ? {status: 404, data: {error: 'Unknown experiment job.'}} : null);
  await ui.buttons().find(item => /monitor/i.test(item.textContent)).onclick();
  assert.equal(ui.elements.get('send').disabled, false);
  assert.equal(ui.elements.get('cancelJob').hidden, true);
  assert.match(ui.elements.get('experimentStatus').textContent, /no longer available/);
});

test('lazy result errors are shown instead of disappearing silently', async () => {
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('completed')]}}
    : url === '/api/jobs/' + jobId ? {status: 500, data: {error: 'Saved job could not be read'}} : null);
  const button = ui.buttons().find(item => /view result/i.test(item.textContent));
  await button.onclick();
  assert.match(ui.elements.get('jobStatus').textContent, /Saved job could not be read/);
  assert.equal(button.disabled, false);
});

test('local status diagnostics remain visible while the server stays connected', async () => {
  let issues = {'reports.core': 'Invalid JSON: <unclosed object>'};
  const ui = await workbench(({url}) => url === '/api/status' ? {status: 200, data: {reports: {}, status_errors: issues}}
    : url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
  assert.match(ui.elements.get('connection').textContent, /Connected/);
  assert.ok(ui.elements.get('statusDiagnostics'), 'the diagnostics must have a visible place in the page');
  assert.equal(ui.elements.get('statusDiagnostics').hidden, false);
  assert.equal(ui.elements.get('statusErrors').children[0].textContent, 'reports.core: Invalid JSON: <unclosed object>');
  issues = {}; await vm.runInContext('refreshStatus()', ui.context); await ui.flush();
  assert.equal(ui.elements.get('statusDiagnostics').hidden, true, 'resolved diagnostics should clear on refresh');
});

for (const kind of ['cognitive', 'titan', 'trident', 'hyperion', 'chimera', 'metamorph', 'aura']) {
  for (const [coreReady, candidateReady] of [[false, true], [true, false]]) {
    test(kind + ' requires both a checkpoint and candidate readiness (' + coreReady + ', ' + candidateReady + ')', async () => {
      const ui = await workbench(({url}) => url === '/api/status' ? {status: 200, data: {core_ready: coreReady, [kind]: {ready: candidateReady}}}
        : url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
      ui.elements.get('experimentType').value = kind;
      ui.elements.get('experimentType').onchange();
      assert.equal(ui.elements.get('runExperiment').disabled, true);
      await ui.elements.get('experimentForm').onsubmit({preventDefault() {}});
      assert.equal(ui.calls.filter(call => call.method === 'POST').length, 0);
    });
  }
}

function renderedText(element) {
  return [element.textContent, element.innerHTML || '', ...element.children.filter(item => typeof item !== 'string').map(renderedText)].join(' ');
}

let failedPoincareReceipt;
function realFailedPoincareReceipt() {
  if (!failedPoincareReceipt) failedPoincareReceipt = JSON.parse(execFileSync(process.env.PYTHON || 'python', ['-c',
    'import json; from poseidon.poincare_engine import PoincareEngine; print(json.dumps(PoincareEngine().synthesize(iterations=17, observation_16d=[0.0]*16)))'],
    {cwd: path.join(__dirname, '..'), encoding: 'utf8'}));
  return failedPoincareReceipt;
}

test('POINCARE chat renders the actual failed numerical diagnostics without certifying them', async () => {
  const result = realFailedPoincareReceipt();
  assert.equal(result.verified, false);
  const ui = await workbench(() => null);
  ui.context.result = result;
  vm.runInContext("message('assistant','Bounded diagnostic result','POINCARE',result)", ui.context);
  const text = renderedText(ui.elements.get('messages'));
  assert.match(text, /DIAGNOSTICS FAILED/);
  assert.ok(text.includes('Lyapunov λ1: ' + result.poincare_return_map.lambda_1.toFixed(4)));
  assert.ok(text.includes('Gromov δ: ' + result.hyperbolic_poincare_disk.delta_hyperbolicity.toFixed(4)));
  assert.ok(text.includes('Synthetic quartet action: ' + result.unified_quartet_coordinator.final_dispatched_action));
  assert.match(text, /finite orbit.*curvature relaxation surrogate.*synthetic fixtures/i);
  assert.doesNotMatch(text, /POINCARÉ Certified|undefined|Geodesic d: 0/);
});

test('POINCARE missing diagnostics remain unavailable instead of becoming zero or a chaotic regime', async () => {
  const ui = await workbench(() => null);
  ui.context.result = {project: 'POINCARE', verified: false};
  vm.runInContext("message('assistant','Incomplete diagnostic result','POINCARE',result)", ui.context);
  const text = renderedText(ui.elements.get('messages'));
  assert.match(text, /Lyapunov λ1: Unavailable/);
  assert.match(text, /Gromov δ: Unavailable/);
  assert.match(text, /Synthetic quartet action: Unavailable/);
  assert.doesNotMatch(text, /undefined|chaotic|λ1: 0\.0000|δ: 0\.0000/);
});

test('POINCARE synthesis button preserves failed diagnostics when readiness is later polled', async () => {
  const result = realFailedPoincareReceipt();
  const ui = await workbench(({url}) => url === '/api/poincare/synthesize' ? {status: 200, data: result}
    : url === '/api/telemetry/memory-influence' ? {status: 200, data: {}}
      : url === '/api/poincare/status' ? {status: 200, data: {ready: true, total_verified_domains: null}}
        : null);
  await ui.elements.get('synthesizePoincareBtn').onclick();
  assert.equal(ui.elements.get('poincareTelemetryStatus').textContent, 'DIAGNOSTICS FAILED');
  assert.ok(ui.elements.get('poincareTelemetryText').textContent.includes('Lyapunov λ1: ' + result.poincare_return_map.lambda_1.toFixed(4)));
  assert.ok(ui.elements.get('poincareTelemetryText').textContent.includes('Synthetic quartet action: ' + result.unified_quartet_coordinator.final_dispatched_action));
  await ui.elements.get('pollTelemetryBtn').onclick();
  assert.equal(ui.elements.get('poincareTelemetryStatus').textContent, 'DIAGNOSTICS FAILED');
});

test('POINCARE readiness polling reports availability without inventing numerical evidence', async () => {
  const ui = await workbench(({url}) => url === '/api/telemetry/memory-influence' ? {status: 200, data: {}}
    : url === '/api/poincare/status' ? {status: 200, data: {ready: true, total_verified_domains: 100,
      telemetry_summary: {sample_lyapunov: .124, sample_gromov_delta: .052, poincare_duality_satisfied: true, ricci_entropy_monotonic: true}}}
      : url === '/api/status' ? {status: 200, data: {poincare: {ready: true}}} : null);
  await ui.elements.get('pollTelemetryBtn').onclick();
  assert.match(ui.elements.get('poincareTelemetryStatus').textContent, /AVAILABLE.*NOT RUN/);
  assert.match(ui.elements.get('poincareState').textContent, /AVAILABLE/);
  assert.match(ui.elements.get('poincareTelemetryText').textContent, /Not run/i);
  assert.doesNotMatch(ui.elements.get('poincareTelemetryText').textContent, /0\.124|100\/100/);
  assert.match(ui.elements.get('poincareNote').textContent, /finite orbit.*curvature relaxation surrogate.*synthetic fixtures/i);
});

test('initial telemetry cards show not-run placeholders rather than fabricated metrics or certifications', async () => {
  const ui = await workbench(() => null);
  const grid = html.split('id="telemetryGrid"')[1].split('</section>')[0];
  const ids = [...grid.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.ok(ids.length > 40);
  for (const id of ids) {
    const value = ui.elements.get(id).textContent;
    if (id.endsWith('Status')) assert.equal(value, 'NOT RUN', id);
    else assert.match(value, /Not run|No .*recorded/i, id);
  }
});

test('an unsupported Omni answer labels its fixed safety proof as a demonstration', async () => {
  const ui = await workbench(() => null);
  ui.context.result = {
    verified: false, plan: ['Inspect assumptions'],
    verification: {answer: {verified: false, scope: 'unsupported_query', query_bound: false}},
    formal_proof: {verdict: 'PROVEN_SAFE', contract_satisfied: true, query_bound: false, certificate_sha256: 'b'.repeat(64)},
    thought_steps: [{step: 1, phase: 'safety', content: 'Check a fixed survival example'}],
  };
  vm.runInContext("message('assistant','A general suggestion','omni',result)", ui.context);
  const text = renderedText(ui.elements.get('messages'));
  assert.match(text, /Answer unverified/i);
  assert.match(text, /Synthetic safety demonstration/i);
  assert.doesNotMatch(text, /LTL Proof Certified|Verified Solution Plan/i);
});

test('a checked solver answer keeps parsed-model and generated-prose scopes separate', async () => {
  const ui = await workbench(() => null);
  ui.context.result = {verified: true, verification: {
    answer: {verified: true, scope: 'parsed_problem_model', query_bound: true, input_interpretation_verified: false},
    language_synthesis: {present: true, verified: false},
  }};
  vm.runInContext("message('assistant','Computed result and generated explanation','omni',result)", ui.context);
  const text = renderedText(ui.elements.get('messages'));
  assert.match(text, /Parsed-model answer verified/i);
  assert.match(text, /generated prose.*unverified/i);
});

test('checksum-only receipt checks do not claim replay verification', async () => {
  const result = {...receipt, verification: {verified: true, scope: 'checksum_only'}};
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [listedJob('completed')]}}
    : url === '/api/jobs/' + jobId ? {status: 200, data: {...listedJob('completed'), result}} : null);
  await ui.buttons().find(item => /view result/i.test(item.textContent)).onclick();
  const text = ui.elements.get('experimentIdentity').textContent;
  assert.match(text, /receipt checks passed/i);
  assert.match(text, /checksum_only/);
  assert.doesNotMatch(text, /replay verified|independently/i);
});

for (const protocol of ['mco', 'tessera', 'mnemorph']) test(protocol + ' submission describes its fixed protocol without invented counts', async () => {
  let accept;
  const ui = await workbench(call => {
    if (call.url === '/api/jobs' && call.method === 'GET') return {status: 200, data: {jobs: []}};
    if (call.url === '/api/jobs' && call.method === 'POST') return new Promise(resolve => {accept = resolve;});
    if (call.url === '/api/jobs/' + jobId) return {status: 200, data: {...listedJob('completed'), result: receipt}};
    return null;
  });
  ui.elements.get('experimentType').value = protocol;
  const submission = ui.elements.get('experimentForm').onsubmit({preventDefault() {}});
  await ui.flush();
  const text = ui.elements.get('experimentStatus').textContent;
  assert.match(text, /fixed.*protocol|fixed.*benchmark/i);
  assert.doesNotMatch(text, /controllers|paired episodes|steps/i);
  accept({status: 202, data: listedJob('queued')});
  await submission;
});

test('MCO setup follows receipt availability while the benchmark remains runnable', async () => {
  let receiptReady = false;
  const ui = await workbench(({url}) => url === '/api/status' ? {status: 200, data: {
    core_ready: false, mco: {ready: true, receipt_ready: receiptReady},
  }} : url === '/api/jobs' ? {status: 200, data: {jobs: []}} : null);
  ui.elements.get('experimentType').value = 'mco';
  ui.elements.get('experimentType').onchange();
  assert.equal(ui.elements.get('mcoSetup').hidden, false);
  assert.equal(ui.elements.get('runExperiment').disabled, false);
  receiptReady = true; await vm.runInContext('refreshStatus()', ui.context); await ui.flush();
  assert.equal(ui.elements.get('mcoSetup').hidden, true);
  assert.equal(ui.elements.get('runExperiment').disabled, false);
});

test('a delayed saved result cannot replace a newly monitored job', async () => {
  const savedId = 'b'.repeat(32);
  let deliverSavedResult;
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [
    {...listedJob('completed'), id: savedId}, listedJob('running'),
  ]}} : url === '/api/jobs/' + savedId ? new Promise(resolve => {deliverSavedResult = resolve;})
    : url === '/api/jobs/' + jobId ? {status: 200, data: listedJob('running')} : null);
  const opening = ui.buttons().find(item => /view result/i.test(item.textContent)).onclick();
  await ui.flush();
  await ui.buttons().find(item => /monitor/i.test(item.textContent)).onclick();
  deliverSavedResult({status: 200, data: {...listedJob('completed'), id: savedId, result: receipt}});
  await opening; await ui.flush();
  assert.match(ui.elements.get('jobIdentity').textContent, new RegExp(jobId + '.*running'));
  assert.equal(ui.elements.get('experimentResults').hidden, true);
  assert.equal(ui.elements.get('send').disabled, true);
});

test('the latest saved-result selection owns the display when reads finish out of order', async () => {
  const olderId = 'b'.repeat(32), newerId = 'c'.repeat(32);
  let deliverOlder;
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [
    {...listedJob('completed'), id: olderId}, {...listedJob('completed'), id: newerId},
  ]}} : url === '/api/jobs/' + olderId ? new Promise(resolve => {deliverOlder = resolve;})
    : url === '/api/jobs/' + newerId ? {status: 200, data: {...listedJob('completed'), id: newerId, result: {...receipt, experiment_id: 'newest-selection'}}} : null);
  const opening = ui.buttons()[0].onclick(); await ui.flush();
  await ui.buttons()[1].onclick();
  deliverOlder({status: 200, data: {...listedJob('completed'), id: olderId, result: receipt}});
  await opening; await ui.flush();
  assert.match(ui.elements.get('experimentIdentity').textContent, /newest-selection/);
  assert.match(ui.elements.get('jobIdentity').textContent, new RegExp(newerId));
});

test('MCO history renders its actual retrieval metrics and restores controller views when switched', async () => {
  const controllerId = 'b'.repeat(32);
  const conditions = {
    intact: {trials: 256, mean_hit_at_1: .5, mean_hit_at_3: .75, mean_mrr: .625, mean_grounded_score: .51953125, mean_retrieval_ms: 60.35},
    shuffled: {trials: 256, mean_hit_at_1: .25, mean_hit_at_3: .5, mean_mrr: .375, mean_grounded_score: .3, mean_retrieval_ms: 59},
    irrelevant: {trials: 256, mean_hit_at_1: 0, mean_hit_at_3: 0, mean_mrr: 0, mean_grounded_score: 0, mean_retrieval_ms: 55},
    erased: {trials: 256, mean_hit_at_1: 0, mean_hit_at_3: 0, mean_mrr: 0, mean_grounded_score: 0, mean_retrieval_ms: 3},
  };
  const mco = {schema: 'poseidon-mco-experiment-v1', experiment_id: 'mco-fixture',
    best_subset: 'episodic+body+habitat+social', condition_comparison: conditions,
    carrier_contributions: {episodic: {carrier: 'episodic', mean_with: .625, mean_without: .5, marginal_grounded_delta: .125}},
    verification: {verified: true}, artifact_url: '/artifacts/mco_experiments/RECEIPT.json'};
  const controller = {schema: 'poseidon-titan-experiment-v2', experiment_id: 'controller-fixture', summary: {
    policy: {episodes: 1, survival_rate: 1, mean_steps: 32, mean_reward: 2},
  }, paired: {}};
  const ui = await workbench(({url}) => url === '/api/jobs' ? {status: 200, data: {jobs: [
    listedJob('completed'), {...listedJob('completed'), id: controllerId},
  ]}} : url === '/api/jobs/' + jobId ? {status: 200, data: {...listedJob('completed'), result: mco}}
    : url === '/api/jobs/' + controllerId ? {status: 200, data: {...listedJob('completed'), id: controllerId, result: controller}} : null);
  await ui.buttons()[0].onclick();
  await ui.flush();
  assert.ok(ui.elements.get('mcoResults'), 'MCO must have a retrieval-results panel');
  assert.equal(ui.elements.get('mcoResults').hidden, false);
  assert.equal(ui.elements.get('mcoConditions').children.length, 4);
  assert.match(renderedText(ui.elements.get('mcoConditions')), /intact.*256.*50\.0%.*75\.0%.*0\.625.*0\.5195.*60\.35/);
  assert.match(renderedText(ui.elements.get('mcoCarriers')), /episodic.*0\.625.*0\.500.*\+0\.125/);
  assert.match(ui.elements.get('mcoBestSubset').textContent, /episodic.*body.*habitat.*social/);
  assert.equal(ui.elements.get('controllerResults').hidden, true);
  assert.equal(ui.elements.get('pairedResults').hidden, true);
  assert.match(ui.elements.get('resultCaution').textContent, /retrieval benchmark/i);
  await ui.buttons()[1].onclick();
  await ui.flush();
  assert.equal(ui.elements.get('mcoResults').hidden, true);
  assert.equal(ui.elements.get('controllerResults').hidden, false);
  assert.equal(ui.elements.get('pairedResults').hidden, false);
  assert.equal(ui.elements.get('experimentSummary').children.length, 1);
  assert.match(ui.elements.get('comparisonTitle').textContent, /Controller comparison/);
});
