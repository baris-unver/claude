/* In-browser localisation: the same pipeline the server runs, ported to onnxruntime-web.
   GitHub Pages cannot execute the model, so the model is shipped to the visitor instead.

   Fidelity notes, both measured rather than assumed:
   * the encoder is fp32 (86 MB). int8 was built and rejected -- it kept only 72.9% of the top-1
     cells and shifted p90 position by 1.4 km on the demo queries.
   * the image-refine step is NOT included: it needs the 97,834-image index (191 MB). This runs
     "prototypes + aerial + refine", which the evaluation put ~1 point of R@100m below the full
     method. Pre-selected queries show their FULL-method numbers, computed server-side; an upload
     is labelled as the browser pipeline so the two are never silently mixed. */
export const R_EARTH = 6371008.8;

let ORT = null, session = null, META = null, CELLS = null, CENTERS = null;

function f16to32(buf){                     // no portable Float16Array yet; decode by hand
  const u = new Uint16Array(buf), out = new Float32Array(u.length);
  for(let i = 0; i < u.length; i++){
    const h = u[i], s = (h & 0x8000) >> 15, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
    out[i] = e === 0 ? (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
           : e === 31 ? (f ? NaN : (s ? -Infinity : Infinity))
           : (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

async function fetchProgress(url, onProgress){
  const r = await fetch(url);
  if(!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +(r.headers.get('content-length') || 0);
  const reader = r.body.getReader(); const chunks = []; let got = 0;
  for(;;){
    const {done, value} = await reader.read();
    if(done) break;
    chunks.push(value); got += value.length;
    onProgress && onProgress(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for(const c of chunks){ out.set(c, o); o += c.length; }
  return out.buffer;
}

export async function load(onProgress){
  ORT = window.ort;
  ORT.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  META = await (await fetch('model/meta.json')).json();
  const step = (label, frac) => (g, t) => onProgress &&
    onProgress(label, g, t, frac[0] + (t ? g / t : 0) * (frac[1] - frac[0]));

  const cells = await fetchProgress('model/cells.bin', step('cell database', [0, 0.2]));
  CELLS = f16to32(cells);
  CENTERS = new Float32Array(await fetchProgress('model/centers.bin', step('cell centres', [0.2, 0.21])));
  const enc = await fetchProgress('model/encoder.onnx', step('encoder', [0.21, 0.99]));
  session = await ORT.InferenceSession.create(enc, {executionProviders: ['wasm']});
  onProgress && onProgress('ready', 1, 1, 1);
  return META;
}

/* torchvision eval_transform: Resize(shortest side -> S) then CenterCrop(S), then ImageNet
   normalisation. Done on a canvas here; drawImage's filtering is not bit-identical to PIL's
   bilinear, which is why the port is checked end-to-end against the server's own predictions
   rather than trusted. */
export function preprocess(img){
  const S = META.image_size;
  const scale = S / Math.min(img.width, img.height);
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const c = document.createElement('canvas'); c.width = S; c.height = S;
  const g = c.getContext('2d', {willReadFrequently: true});
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(img, Math.round((w - S) / 2) * -1, Math.round((h - S) / 2) * -1, w, h);
  const d = g.getImageData(0, 0, S, S).data;
  const t = new Float32Array(3 * S * S), m = META.mean, sd = META.std;
  for(let i = 0, n = S * S; i < n; i++){
    t[i]         = (d[i*4]     / 255 - m[0]) / sd[0];
    t[n + i]     = (d[i*4 + 1] / 255 - m[1]) / sd[1];
    t[2*n + i]   = (d[i*4 + 2] / 255 - m[2]) / sd[2];
  }
  return new ORT.Tensor('float32', t, [1, 3, S, S]);
}

export async function embed(img){
  const out = await session.run({image: preprocess(img)});
  return out[session.outputNames[0]].data;      // already unit-norm out of the encoder
}

const toUnit = (lat, lon) => {
  const a = lat * Math.PI / 180, b = lon * Math.PI / 180;
  return [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
};
const toLatLon = ([x, y, z]) => {
  const n = Math.hypot(x, y, z) || 1e-12;
  return [Math.asin(Math.max(-1, Math.min(1, z / n))) * 180 / Math.PI,
          Math.atan2(y / n, x / n) * 180 / Math.PI];
};
export const haversine = (a1, o1, a2, o2) => {
  const r = Math.PI / 180, dLat = (a2 - a1) * r, dLon = (o2 - o1) * r;   // `do` is reserved
  const h = Math.sin(dLat/2)**2 + Math.cos(a1*r)*Math.cos(a2*r)*Math.sin(dLon/2)**2;
  return 2 * R_EARTH * Math.asin(Math.sqrt(Math.min(1, h)));
};

/* localize(): mirrors geoloc_tr/localize.py -- score every cell, take top_k, keep those within
   refine_radius_m of the best, softmax their scores at refine_temperature, and take the weighted
   centroid on the sphere. */
export function localize(q){
  const N = META.n_cells, D = META.dim, K = Math.min(META.top_k, N);
  const sc = new Float32Array(N);
  for(let c = 0; c < N; c++){
    let s = 0, o = c * D;
    for(let d = 0; d < D; d++) s += q[d] * CELLS[o + d];
    sc[c] = s;
  }
  const order = Array.from(sc.keys()).sort((a, b) => sc[b] - sc[a]).slice(0, K);
  const best = order[0];
  const bu = toUnit(CENTERS[best*2], CENTERS[best*2 + 1]);
  const keep = order.filter(i => {
    const u = toUnit(CENTERS[i*2], CENTERS[i*2 + 1]);
    return Math.hypot(u[0]-bu[0], u[1]-bu[1], u[2]-bu[2]) * R_EARTH <= META.refine_radius_m;
  });
  const top = sc[keep[0]];
  const w = keep.map(i => Math.exp((sc[i] - top) / META.refine_temperature));
  const wt = w.reduce((a, b) => a + b, 0);
  const acc = [0, 0, 0];
  keep.forEach((i, j) => {
    const u = toUnit(CENTERS[i*2], CENTERS[i*2 + 1]);
    for(let k = 0; k < 3; k++) acc[k] += (w[j] / wt) * u[k];
  });
  const [lat, lon] = toLatLon(acc);
  const wmax = Math.max(...order.map(i => Math.exp((sc[i] - top) / META.refine_temperature)));
  return {lat, lon, top_score: sc[best],
          candidates: order.slice(0, 20).map(i => ({
            lat: CENTERS[i*2], lon: CENTERS[i*2 + 1], score: sc[i],
            weight: Math.exp((sc[i] - top) / META.refine_temperature) / (wmax || 1)}))};
}
