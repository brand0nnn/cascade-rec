const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const V = require('./visuals.js');
const D = JSON.parse(fs.readFileSync(path.join(__dirname,'data/evidence.json'),'utf8'));

test('Measured K intervals keep their actual numerical spacing',()=>{
  assert.equal(V.linearPosition(100,50,500,0,450),50);
  assert.equal(V.linearPosition(200,50,500,0,450),150);
});
test('A paired interval crossing or touching zero is not called a positive result',()=>{
  assert.equal(V.confidenceVerdict({ci_low:-.001,ci_high:.002}),'Interval includes zero');
  assert.equal(V.confidenceVerdict({ci_low:0,ci_high:.002}),'Interval includes zero');
  assert.equal(V.confidenceVerdict({ci_low:.001,ci_high:.002}),'Positive difference');
  assert.equal(V.confidenceVerdict({ci_low:-.002,ci_high:-.001}),'Negative difference');
});
test('Every displayed waterfall net equals the saved model score difference',()=>{
  const rows=D.studies.k.rows;
  for(const d of D.studies.k.expansion){
    const a=rows.find(r=>r.k===d.low),b=rows.find(r=>r.k===d.high);
    assert.ok(Math.abs(V.expansionNet(d)-(b.value-a.value))<1e-12);
  }
});
