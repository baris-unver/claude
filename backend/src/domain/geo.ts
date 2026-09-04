import type { GeoPolygon, LatLng } from './types.js';

const EARTH_RADIUS_M = 6_371_008.8;
const DEG = Math.PI / 180;

export function haversineMeters(a: LatLng, b: LatLng): number {
  const dLat = (b.lat - a.lat) * DEG;
  const dLng = (b.lng - a.lng) * DEG;
  const s =
    Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * DEG) * Math.cos(b.lat * DEG) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(s)));
}

export function ringFromGeoJson(ring: number[][]): LatLng[] {
  return ring.map(([lng, lat]) => ({ lat, lng }));
}

/** Işın atma (ray casting) yöntemi ile noktanın halka içinde olup olmadığı. */
export function pointInRing(p: LatLng, ring: LatLng[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const yi = ring[i].lat;
    const xi = ring[i].lng;
    const yj = ring[j].lat;
    const xj = ring[j].lng;
    const crosses = yi > p.lat !== yj > p.lat && p.lng < ((xj - xi) * (p.lat - yi)) / (yj - yi) + xi;
    if (crosses) inside = !inside;
  }
  return inside;
}

export function pointInPolygon(p: LatLng, polygon: GeoPolygon): boolean {
  const [outer, ...holes] = polygon.coordinates;
  if (!outer || !pointInRing(p, ringFromGeoJson(outer))) return false;
  return !holes.some((h) => pointInRing(p, ringFromGeoJson(h)));
}

/** Yerel düzlem izdüşümü (küçük mesafeler için yeterli hassasiyet). */
function project(p: LatLng, ref: LatLng): { x: number; y: number } {
  const kx = Math.cos(ref.lat * DEG) * 111_320;
  const ky = 110_574;
  return { x: (p.lng - ref.lng) * kx, y: (p.lat - ref.lat) * ky };
}

export function distanceToSegmentMeters(p: LatLng, a: LatLng, b: LatLng): number {
  const pa = project(a, p);
  const pb = project(b, p);
  const dx = pb.x - pa.x;
  const dy = pb.y - pa.y;
  const len2 = dx * dx + dy * dy;
  let t = 0;
  if (len2 > 0) t = Math.max(0, Math.min(1, (-(pa.x * dx) - pa.y * dy) / len2));
  const cx = pa.x + t * dx;
  const cy = pa.y + t * dy;
  return Math.sqrt(cx * cx + cy * cy);
}

export function distanceToRingMeters(p: LatLng, ring: LatLng[]): number {
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    best = Math.min(best, distanceToSegmentMeters(p, ring[j], ring[i]));
  }
  return best;
}

export function distanceToBoundaryMeters(p: LatLng, polygon: GeoPolygon): number {
  return Math.min(...polygon.coordinates.map((r) => distanceToRingMeters(p, ringFromGeoJson(r))));
}

export type GeofenceState = 'inside' | 'outside' | 'uncertain';

/**
 * Konum doğruluğu (accuracy, metre) sınıra olan uzaklıktan büyükse karar verilemez ("uncertain").
 * Bu, GPS gürültüsünden kaynaklanan sahte ihlalleri önler.
 */
export function evaluateGeofence(p: LatLng, accuracy: number, polygon: GeoPolygon): GeofenceState {
  const inside = pointInPolygon(p, polygon);
  const d = distanceToBoundaryMeters(p, polygon);
  if (d <= accuracy) return 'uncertain';
  return inside ? 'inside' : 'outside';
}

export function validatePolygon(polygon: unknown): string | null {
  if (!polygon || typeof polygon !== 'object') return 'Poligon nesnesi bekleniyor';
  const poly = polygon as Partial<GeoPolygon>;
  if (poly.type !== 'Polygon') return "type alanı 'Polygon' olmalıdır";
  if (!Array.isArray(poly.coordinates) || poly.coordinates.length === 0) return 'coordinates boş olamaz';
  for (const ring of poly.coordinates) {
    if (!Array.isArray(ring) || ring.length < 4) return 'Her halka en az 4 nokta (kapalı) içermelidir';
    for (const pt of ring) {
      if (!Array.isArray(pt) || pt.length < 2) return 'Koordinat [lng, lat] biçiminde olmalıdır';
      const [lng, lat] = pt;
      if (typeof lng !== 'number' || typeof lat !== 'number' || Number.isNaN(lng) || Number.isNaN(lat)) {
        return 'Koordinatlar sayısal olmalıdır';
      }
      if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return 'Koordinat aralık dışı';
    }
    const first = ring[0];
    const last = ring[ring.length - 1];
    if (first[0] !== last[0] || first[1] !== last[1]) return 'Halka kapalı olmalıdır (ilk ve son nokta aynı)';
  }
  return null;
}

/** Yarıçapı verilen daireyi çokgene çevirir (yönetim panelinde hızlı bölge tanımı için). */
export function circleToPolygon(center: LatLng, radiusM: number, steps = 36): GeoPolygon {
  const ring: number[][] = [];
  const kx = Math.cos(center.lat * DEG) * 111_320;
  const ky = 110_574;
  for (let i = 0; i < steps; i++) {
    const a = (i / steps) * 2 * Math.PI;
    ring.push([center.lng + (radiusM * Math.cos(a)) / kx, center.lat + (radiusM * Math.sin(a)) / ky]);
  }
  ring.push(ring[0]);
  return { type: 'Polygon', coordinates: [ring] };
}

export function polygonCentroid(polygon: GeoPolygon): LatLng {
  const ring = polygon.coordinates[0];
  const pts = ring.slice(0, -1);
  const lat = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const lng = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  return { lat, lng };
}
