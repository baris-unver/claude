/* In-browser localisation: the same pipeline the server runs, ported to onnxruntime-web.
   GitHub Pages cannot execute the model, so the model is shipped to the visitor instead.

   Two pipelines share this code:
   * `ground`   (model/):          street-level photos, the paper's setting.
   * `overhead` (model_overhead/): satellite / aerial photos as queries (geoloc_tr/overhead.py).
     Its 278,932-cell database is shipped as an uncentred PCA to 128 dims (proj.bin, applied to the
     query here) with int8 rows and a per-row scale -- 35 MB instead of 272 MB. Measured cost on the
     2000-query test sets: about one point of R@100m. Queries are averaged over 4 rotations.

   Fidelity notes, both measured rather than assumed:
   * the encoders are fp32 (86 MB each). int8 was built and rejected for the ground model -- it kept
     only 72.9% of the top-1 cells and shifted p90 position by 1.4 km on the demo queries.
   * the image-refine step is NOT included: it needs the 97,834-image index (191 MB) for the ground
     model and the full 279k x 512 code matrix for the overhead one. This runs "prototypes + aerial +
     refine", which the evaluation put ~1 point of R@100m below the full method. Pre-selected queries
     show their FULL-method numbers, computed server-side; an upload is labelled as the browser
     pipeline so the two are never silently mixed. */
export const R_EARTH = 6371008.8;

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

export class Pipeline {
  constructor(dir){ this.dir = dir; this.session = null; this.META = null; }

  async load(onProgress){
    const ORT = window.ort;
    ORT.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    const M = this.META = await (await fetch(`${this.dir}/meta.json`)).json();
    const step = (label, frac) => (g, t) => onProgress &&
      onProgress(label, g, t, frac[0] + (t ? g / t : 0) * (frac[1] - frac[0]));
    const cells = await fetchProgress(`${this.dir}/cells.bin`, step('cell database', [0, 0.2]));
    if(M.dtype === 'int8'){
      this.CELLS = new Int8Array(cells);
      this.SCALES = new Float32Array(await fetchProgress(`${this.dir}/scales.bin`, step('cell scales', [0.2, 0.205])));
      this.PROJ = new Float32Array(await fetchProgress(`${this.dir}/proj.bin`, step('projection', [0.205, 0.21])));
      this.D = M.pca_dim;
    } else {
      this.CELLS = f16to32(cells); this.SCALES = null; this.PROJ = null; this.D = M.dim;
    }
    this.CENTERS = new Float32Array(await fetchProgress(`${this.dir}/centers.bin`, step('cell centres', [0.21, 0.215])));
    const enc = await fetchProgress(`${this.dir}/encoder.onnx`, step('encoder', [0.215, 0.99]));
    this.session = await ORT.InferenceSession.create(enc, {executionProviders: ['wasm']});
    onProgress && onProgress('ready', 1, 1, 1);
    return M;
  }

  /* torchvision eval_transform: Resize(shortest side -> S) then CenterCrop(S), then ImageNet
     normalisation, on a canvas. `cropSide` (source pixels) takes a centred square of that size
     instead of the whole shortest side -- the overhead pipeline's metres-per-pixel option; `rotation`
     (degrees) rotates about the centre for rotation TTA. drawImage's filtering is not bit-identical to
     PIL's bilinear, which is why the port is checked end-to-end against the server's own predictions
     rather than trusted. */
  preprocess(img, {rotation = 0, cropSide = null} = {}, into = null, offset = 0){
    const S = this.META.image_size;
    const side = Math.min(cropSide || Infinity, img.width, img.height);
    const c = document.createElement('canvas'); c.width = S; c.height = S;
    const g = c.getContext('2d', {willReadFrequently: true});
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.translate(S / 2, S / 2); g.rotate(rotation * Math.PI / 180);
    // draw the whole image scaled so that the chosen centre square maps onto the S x S canvas;
    // pixels outside the square then fill the corners a rotation would otherwise leave black
    const k = S / side;
    g.drawImage(img, -img.width / 2 * k, -img.height / 2 * k, img.width * k, img.height * k);
    const d = g.getImageData(0, 0, S, S).data;
    const n = S * S, t = into || new Float32Array(3 * n), m = this.META.mean, sd = this.META.std, o = offset;
    for(let i = 0; i < n; i++){
      t[o + i]       = (d[i*4]     / 255 - m[0]) / sd[0];
      t[o + n + i]   = (d[i*4 + 1] / 255 - m[1]) / sd[1];
      t[o + 2*n + i] = (d[i*4 + 2] / 255 - m[2]) / sd[2];
    }
    return t;
  }

  /* unit embedding of one image; `tta` > 1 averages the encoder output over that many rotations
     (one batched run), for overhead queries of unknown heading */
  async embed(img, {tta = 1, cropSide = null} = {}){
    const S = this.META.image_size, n = 3 * S * S;
    const t = new Float32Array(tta * n);
    for(let k = 0; k < tta; k++) this.preprocess(img, {rotation: 360 * k / tta, cropSide}, t, k * n);
    const out = await this.session.run({image: new window.ort.Tensor('float32', t, [tta, 3, S, S])});
    const e = out[this.session.outputNames[0]].data, D = this.META.dim, q = new Float32Array(D);
    for(let k = 0; k < tta; k++) for(let d = 0; d < D; d++) q[d] += e[k * D + d];
    let nrm = 0; for(let d = 0; d < D; d++) nrm += q[d] * q[d];
    nrm = Math.sqrt(nrm) || 1e-12; for(let d = 0; d < D; d++) q[d] /= nrm;
    return q;
  }

  /* localize(): mirrors geoloc_tr/localize.py -- score every cell (through the PCA projection and
     int8 scales when the database is compressed), take top_k, keep those within refine_radius_m of
     the best, softmax their scores at refine_temperature, and take the weighted centroid on the
     sphere. */
  localize(q){
    const M = this.META, N = M.n_cells, D = this.D, K = Math.min(M.top_k, N);
    let qq = q;
    if(this.PROJ){                          // q (dim) -> q P (pca_dim); proj.bin is row-major (dim, pca_dim)
      qq = new Float32Array(D);
      for(let i = 0; i < M.dim; i++){ const qi = q[i], o = i * D; if(qi) for(let j = 0; j < D; j++) qq[j] += qi * this.PROJ[o + j]; }
    }
    const sc = new Float32Array(N), C = this.CELLS, SC = this.SCALES;
    for(let c = 0; c < N; c++){
      let s = 0; const o = c * D;
      for(let d = 0; d < D; d++) s += qq[d] * C[o + d];
      sc[c] = SC ? s * SC[c] : s;
    }
    // top-K by a linear scan (a full sort of 279k indices is the slow part otherwise)
    const order = [];
    for(let c = 0; c < N; c++){
      if(order.length < K || sc[c] > sc[order[order.length - 1]]){
        let i = order.length; order.push(c);
        while(i > 0 && sc[order[i - 1]] < sc[c]){ order[i] = order[i - 1]; i--; }
        order[i] = c;
        if(order.length > K) order.pop();
      }
    }
    const best = order[0];
    const bu = toUnit(this.CENTERS[best*2], this.CENTERS[best*2 + 1]);
    const keep = order.filter(i => {
      const u = toUnit(this.CENTERS[i*2], this.CENTERS[i*2 + 1]);
      return Math.hypot(u[0]-bu[0], u[1]-bu[1], u[2]-bu[2]) * R_EARTH <= M.refine_radius_m;
    });
    const top = sc[keep[0]];
    const w = keep.map(i => Math.exp((sc[i] - top) / M.refine_temperature));
    const wt = w.reduce((a, b) => a + b, 0);
    const acc = [0, 0, 0];
    keep.forEach((i, j) => {
      const u = toUnit(this.CENTERS[i*2], this.CENTERS[i*2 + 1]);
      for(let k = 0; k < 3; k++) acc[k] += (w[j] / wt) * u[k];
    });
    const [lat, lon] = toLatLon(acc);
    const wmax = Math.max(...order.map(i => Math.exp((sc[i] - top) / M.refine_temperature)));
    return {lat, lon, top_score: sc[best],
            candidates: order.slice(0, 20).map(i => ({
              lat: this.CENTERS[i*2], lon: this.CENTERS[i*2 + 1], score: sc[i],
              weight: Math.exp((sc[i] - top) / M.refine_temperature) / (wmax || 1)}))};
  }
}

export const ground = new Pipeline('model');
export const overhead = new Pipeline('model_overhead');

/* the original single-model API, kept for the street-level page code */
export const load = onProgress => ground.load(onProgress);
export const embed = img => ground.embed(img);
export const localize = q => ground.localize(q);
export const preprocess = img => ground.preprocess(img);
