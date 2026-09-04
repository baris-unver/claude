import { evaluateGeofence, distanceToBoundaryMeters, type GeofenceState } from './geo.js';
import { isInCurfew } from './schedule.js';
import type { Curfew, Severity, ViolationType, Zone } from './types.js';

export interface Detection {
  type: ViolationType;
  severity: Severity;
  details: Record<string, unknown>;
}

export interface SampleInput {
  lat: number;
  lng: number;
  accuracy: number;
  isMock: boolean;
  recordedAt: Date;
}

export interface EvaluationContext {
  zones: Zone[];
  curfews: Curfew[];
  now: Date;
  tz: string;
  maxAccuracyMeters: number;
}

export interface SampleEvaluation {
  detections: Detection[];
  /** İzinli bölge(ler)e göre durum. İzinli bölge tanımlı değilse null. */
  geofence: GeofenceState | null;
  /** Değerlendirmeye alınmadı (düşük doğruluk). */
  skipped: boolean;
}

export function zoneActiveAt(zone: Zone, at: Date): boolean {
  if (zone.activeFrom && at < zone.activeFrom) return false;
  if (zone.activeTo && at > zone.activeTo) return false;
  return true;
}

/**
 * Tek bir konum örneğini kurallara göre değerlendirir. Saf fonksiyondur; yan etkisi yoktur.
 * Kalıcı ihlal kayıtlarının açılması/kapatılması `services/locationIngest.ts` içindedir.
 */
export function evaluateLocationSample(sample: SampleInput, ctx: EvaluationContext): SampleEvaluation {
  const detections: Detection[] = [];
  const p = { lat: sample.lat, lng: sample.lng };

  if (sample.isMock) {
    detections.push({
      type: 'MOCK_LOCATION',
      severity: 'high',
      details: { recordedAt: sample.recordedAt.toISOString(), lat: sample.lat, lng: sample.lng },
    });
  }

  if (sample.accuracy > ctx.maxAccuracyMeters) {
    return { detections, geofence: null, skipped: true };
  }

  const active = ctx.zones.filter((z) => zoneActiveAt(z, sample.recordedAt));
  const allowed = active.filter((z) => z.kind === 'allowed');
  const forbidden = active.filter((z) => z.kind === 'forbidden');
  const home = active.filter((z) => z.kind === 'home');

  let geofence: GeofenceState | null = null;
  if (allowed.length > 0) {
    const states = allowed.map((z) => evaluateGeofence(p, sample.accuracy, z.polygon));
    if (states.includes('inside')) geofence = 'inside';
    else if (states.includes('uncertain')) geofence = 'uncertain';
    else {
      geofence = 'outside';
      const distances = allowed.map((z) => ({
        zoneId: z.id,
        name: z.name,
        distance: distanceToBoundaryMeters(p, z.polygon),
      }));
      const nearest = distances.reduce((a, b) => (a.distance < b.distance ? a : b));
      detections.push({
        type: 'ZONE_EXIT',
        severity: nearest.distance > 1000 ? 'critical' : 'high',
        details: {
          zoneId: nearest.zoneId,
          zoneName: nearest.name,
          distanceMeters: Math.round(nearest.distance),
          lat: sample.lat,
          lng: sample.lng,
          accuracy: sample.accuracy,
          recordedAt: sample.recordedAt.toISOString(),
        },
      });
    }
  }

  for (const z of forbidden) {
    if (evaluateGeofence(p, sample.accuracy, z.polygon) === 'inside') {
      detections.push({
        type: 'FORBIDDEN_ZONE_ENTRY',
        severity: 'critical',
        details: {
          zoneId: z.id,
          zoneName: z.name,
          lat: sample.lat,
          lng: sample.lng,
          recordedAt: sample.recordedAt.toISOString(),
        },
      });
    }
  }

  const activeCurfews = ctx.curfews.filter((c) => isInCurfew(c, sample.recordedAt, ctx.tz));
  if (activeCurfews.length > 0 && home.length > 0) {
    const states = home.map((z) => evaluateGeofence(p, sample.accuracy, z.polygon));
    if (!states.includes('inside') && !states.includes('uncertain')) {
      detections.push({
        type: 'CURFEW_VIOLATION',
        severity: 'high',
        details: {
          curfewId: activeCurfews[0].id,
          window: `${activeCurfews[0].startTime}-${activeCurfews[0].endTime}`,
          lat: sample.lat,
          lng: sample.lng,
          recordedAt: sample.recordedAt.toISOString(),
        },
      });
    }
  }

  return { detections, geofence, skipped: false };
}

/** Konum akışına bağlı olmayan, cihaz durumundan üretilen ihlaller. */
export function evaluateDeviceStatus(status: {
  isRooted: boolean;
  locationServicesEnabled: boolean;
  backgroundPermission: boolean;
}): Detection[] {
  const out: Detection[] = [];
  if (status.isRooted) {
    out.push({ type: 'DEVICE_COMPROMISED', severity: 'high', details: { reason: 'root/jailbreak tespit edildi' } });
  }
  if (!status.locationServicesEnabled || !status.backgroundPermission) {
    out.push({
      type: 'LOCATION_SERVICES_OFF',
      severity: 'medium',
      details: {
        locationServicesEnabled: status.locationServicesEnabled,
        backgroundPermission: status.backgroundPermission,
      },
    });
  }
  return out;
}

export const VIOLATION_LABELS_TR: Record<ViolationType, string> = {
  ZONE_EXIT: 'İzinli bölge dışına çıkış',
  FORBIDDEN_ZONE_ENTRY: 'Yasak bölgeye giriş',
  CURFEW_VIOLATION: 'Ev hapsi saati ihlali',
  MOCK_LOCATION: 'Sahte konum tespiti',
  DEVICE_COMPROMISED: 'Cihaz güvenliği ihlali (root/jailbreak)',
  LOCATION_SERVICES_OFF: 'Konum servisleri kapalı',
  HEARTBEAT_MISSED: 'Cihazdan sinyal alınamıyor',
  CHECKIN_MISSED: 'Yoklama kaçırıldı',
  CHECKIN_VERIFICATION_FAILED: 'Yoklama doğrulaması başarısız',
};
