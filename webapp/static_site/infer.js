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
    const py = M.pyramid || null;
    // the main set: cells.bin (+ *_scales / *_proj when int8) and centers.bin
    const loadSet = async (stem, centers, coarse, frac) => {
      const cells = await fetchProgress(`${this.dir}/${stem}.bin`, step(`cell database ${stem}`, frac));
      const set = {stem};
      if(M.dtype === 'int8'){
        set.CELLS = new Int8Array(cells);
        const sc = `${this.dir}/${stem}_scales.bin`, pr = `${this.dir}/${stem}_proj.bin`;
        set.SCALES = new Float32Array(await fetchProgress(sc));
        set.PROJ = new Float32Array(await fetchProgress(pr));
        set.D = M.pca_dim;
      } else {
        set.CELLS = f16to32(cells); set.SCALES = null; set.PROJ = null; set.D = M.dim;
      }
      set.CENTERS = new Float32Array(await fetchProgress(`${this.dir}/${centers}`));
      set.N = set.CENTERS.length / 2;
      if(coarse) set.COARSE = new Int32Array(await fetchProgress(`${this.dir}/${coarse}`));
      return set;
    };
    const legacy = !py && !(await fetch(`${this.dir}/cells_scales.bin`, {method: 'HEAD'})).ok;
    this.sets = [];
    if(legacy){   // single-set export before the pyramid: scales.bin / proj.bin
      const cells = await fetchProgress(`${this.dir}/cells.bin`, step('cell database', [0, 0.2]));
      const set = {stem: 'cells'};
      if(M.dtype === 'int8'){
        set.CELLS = new Int8Array(cells);
        set.SCALES = new Float32Array(await fetchProgress(`${this.dir}/scales.bin`));
        set.PROJ = new Float32Array(await fetchProgress(`${this.dir}/proj.bin`)); set.D = M.pca_dim;
      } else { set.CELLS = f16to32(cells); set.SCALES = null; set.PROJ = null; set.D = M.dim; }
      set.CENTERS = new Float32Array(await fetchProgress(`${this.dir}/centers.bin`)); set.N = set.CENTERS.length / 2;
      this.sets.push(set);
    } else if(py){
      const n = py.sets.length;
      for(let i = 0; i < n; i++){
        const d = py.sets[i];
        const set = await loadSet(d.name, d.centers, d.coarse, [0.5 * i / n, 0.5 * (i + 1) / n]);
        set.extent_m = d.extent_m; this.sets.push(set);
      }
      this.COARSE_PROTOS = f16to32(await fetchProgress(`${this.dir}/coarse.bin`, step('coarse prototypes', [0.5, 0.52])));
    } else {
      this.sets.push(await loadSet('cells', 'centers.bin', null, [0, 0.2]));
    }
    // legacy field names used by localize(): the main set
    const m = this.sets[0];
    this.CELLS = m.CELLS; this.SCALES = m.SCALES; this.PROJ = m.PROJ; this.D = m.D; this.CENTERS = m.CENTERS;
    const enc = await fetchProgress(`${this.dir}/encoder.onnx`, step('encoder', [py ? 0.52 : 0.215, 0.99]));
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
    if(this.session.outputNames.length > 1){          // scale head: mean log2(extent / reference) over the rotations
      const ls = out[this.session.outputNames[1]].data; let acc = 0;
      for(let k = 0; k < tta; k++) acc += ls[k];
      this.lastLogScale = acc / tta;
    }
    return q;
  }

  /* localize(): mirrors geoloc_tr/localize.py -- score every cell (through the PCA projection and
     int8 scales when the database is compressed), take top_k, keep those within refine_radius_m of
     the best, softmax their scores at refine_temperature, and take the weighted centroid on the
     sphere. */
  localize(q, set = null, mask = null){
    set = set || this.sets[0];
    const M = this.META, N = set.N, D = set.D, K = Math.min(M.top_k, N);
    let qq = q;
    if(set.PROJ){                           // q (dim) -> q P (pca_dim); *_proj.bin is row-major (dim, pca_dim)
      qq = new Float32Array(D);
      for(let i = 0; i < M.dim; i++){ const qi = q[i], o = i * D; if(qi) for(let j = 0; j < D; j++) qq[j] += qi * set.PROJ[o + j]; }
    }
    const sc = new Float32Array(N), C = set.CELLS, SC = set.SCALES;
    for(let c = 0; c < N; c++){
      if(mask && !mask[c]){ sc[c] = -Infinity; continue; }
      let s = 0; const o = c * D;
      for(let d = 0; d < D; d++) s += qq[d] * C[o + d];
      sc[c] = SC ? s * SC[c] : s;
    }
    this.CENTERS = set.CENTERS;             // the centroid / candidate code below reads this set's centres
    // top-K by a linear scan (a full sort of 279k indices is the slow part otherwise)
    const order = [];
    for(let c = 0; c < N; c++){
      if(sc[c] === -Infinity) continue;
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

/* Coarse-to-fine localisation of a photo of any ground extent (mirrors geoloc_tr.overhead.Pyramid):
   whole photo -> embedding + extent (scale head, or `gsd` m/px) + top-k 560 m cells from the coarse
   prototypes; wide photos are cropped to the reference extent and encoded again; the fine pass uses
   the code set nearest in scale and scores only cells inside the region. Narrow photos widen the
   region and also try everything, keeping the better score. */
Pipeline.prototype.localizePyramid = async function(img, {gsd = null, tta = 4} = {}){
  const py = this.META.pyramid;
  if(!py) return this.localize(await this.embed(img, {tta}));
  const S = this.META.image_size, ref = py.ref_extent_m;
  const q0 = await this.embed(img, {tta});
  const extent = gsd ? gsd * Math.min(img.width, img.height) : ref * Math.pow(2, this.lastLogScale || 0);
  const from = gsd ? 'gsd' : 'estimated';
  // coarse pass
  const CP = this.COARSE_PROTOS, nc = py.coarse_classes, D = this.META.dim, cl = new Float32Array(nc);
  for(let c = 0; c < nc; c++){ let s = 0; const o = c * D; for(let d = 0; d < D; d++) s += q0[d] * CP[o + d]; cl[c] = s; }
  const small = extent < py.small_extent_frac * ref, k = small ? py.small_topk : py.coarse_topk;
  const top = Array.from(cl.keys()).sort((a, b) => cl[b] - cl[a]).slice(0, k), topSet = new Set(top);
  // fine pass: crop wide photos to the reference extent
  const cropped = extent >= py.crop_above * ref;
  let q1 = q0, fineExtent = extent;
  if(cropped){ q1 = await this.embed(img, {tta, cropSide: Math.round(ref / extent * Math.min(img.width, img.height))}); fineExtent = ref; }
  const set = this.sets.reduce((best, s) => Math.abs(Math.log2(fineExtent / s.extent_m)) < Math.abs(Math.log2(fineExtent / best.extent_m)) ? s : best, this.sets[0]);
  const mask = new Uint8Array(set.N);
  for(let c = 0; c < set.N; c++) mask[c] = topSet.has(set.COARSE[c]) ? 1 : 0;
  let regionCells = 0; for(let c = 0; c < set.N; c++) regionCells += mask[c];
  let r = this.localize(q1, set, mask), picked = 'region';
  if(small){ const g = this.localize(q1, set, null); if(g.top_score > r.top_score){ r = g; picked = 'global'; } }
  return {...r, pyramid: {extent_m: Math.round(extent), extent_from: from, cropped, set: set.stem,
                          code_extent_m: Math.round(set.extent_m), region_cells: regionCells, picked}};
};

export const ground = new Pipeline('model');
export const overhead = new Pipeline('model_overhead');

/* the original single-model API, kept for the street-level page code */
export const load = onProgress => ground.load(onProgress);
export const embed = img => ground.embed(img);
export const localize = q => ground.localize(q);
export const preprocess = img => ground.preprocess(img);
