// Run with: node tests/check_viewer_credit.cjs
// Exercise the actual formatter and DOM text renderer without a WebGL browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../graftbrain/web/index.html'), 'utf8');
const utilities = html.slice(html.indexOf('const $ ='), html.indexOf('TASKS.forEach'));
const renderer = html.slice(html.indexOf('function updatePlasticity('), html.indexOf('function onSnap('));

function render(input) {
  const node = { children: [], replaceChildren(...children) { this.children = children; } };
  const context = {
    input,
    document: { getElementById: () => node, createElement: () => ({ textContent: '' }) },
  };
  vm.runInNewContext(utilities + renderer + '\nupdatePlasticity(input);', context);
  const pairs = [];
  for (let i = 0; i < node.children.length; i += 2) {
    pairs.push([node.children[i].textContent, node.children[i + 1].textContent]);
  }
  return Object.fromEntries(pairs);
}

const rows = render({
  mode: 'adaptive', candidates: 100, active: 60, dormant: 40, growth_rule: 'gradient',
  growth_init: 'zero', min_age: 20, updates: 35, rewires: 2, swaps: 12,
  protected_active: 20, mature_active: 40, last_rewire_update: 30,
  last_event: { update: 30, swaps: 6, eligible_active: 40, protected_active: 20,
    growth_rule: 'gradient', selection_score_mean: .25 },
});
assert.match(rows['last event swaps / active'], /^6 \(10[.,]0%\)$/);
assert.equal(rows['last event eligible / protected'], '40 / 20');
assert.equal(rows['last event growth rule'], 'gradient');
assert.equal(rows['last recruitment score'], '2.50e-1');
assert.equal(rows['growth rule / initialization'], 'gradient / zero');
assert.equal(rows['minimum active age'], '20 updates');
const old = render({ mode: 'adaptive', candidates: 100, active: 60,
  last_event: { update: 10, swaps: 6 }, rewires: 1 });
assert.equal(old['growth rule / initialization'], 'unavailable · older runtime');
assert.equal(old['last event growth rule'], 'not reported');
assert.equal(old['last recruitment score'], 'not reported');
assert.equal(render(null).telemetry, 'unavailable in this snapshot');
console.log('Viewer credit rendering: known event, old runtime, and absent telemetry passed.');
