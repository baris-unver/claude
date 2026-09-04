import type { LivenessAction, LivenessChallenge } from '../domain/types.js';
import { randomToken } from './crypto.js';

export interface FaceFrame {
  action: LivenessAction;
  image: string; // base64 JPEG
}

export interface FaceVerificationResult {
  score: number; // 0..1 benzerlik
  livenessPassed: boolean;
  provider: string;
  reason?: string;
}

/**
 * Yüz doğrulama sağlayıcısı sözleşmesi.
 * Üretimde kurumun tercih ettiği (tercihen yerinde/on-prem çalışan) yüz tanıma ve canlılık SDK'sı için
 * bu arayüzü uygulayan bir adaptör yazılır; uygulamanın geri kalanı değişmez.
 */
export interface FaceVerificationProvider {
  readonly name: string;
  verify(referenceImage: string, frames: FaceFrame[], challenge: LivenessChallenge): Promise<FaceVerificationResult>;
}

const ALL_ACTIONS: LivenessAction[] = ['look_straight', 'turn_left', 'turn_right', 'blink', 'smile', 'nod'];

/** Rastgele canlılık zorluğu üretir: her zaman düz bakış + 2 rastgele hareket. */
export function issueChallenge(now: Date, ttlSeconds = 180, rng: () => number = Math.random): LivenessChallenge {
  const pool = ALL_ACTIONS.filter((a) => a !== 'look_straight');
  const picked: LivenessAction[] = [];
  while (picked.length < 2) {
    const a = pool[Math.floor(rng() * pool.length)];
    if (!picked.includes(a)) picked.push(a);
  }
  return {
    nonce: randomToken(16),
    actions: ['look_straight', ...picked],
    issuedAt: now,
    expiresAt: new Date(now.getTime() + ttlSeconds * 1000),
  };
}

/** Karelerin zorluktaki hareket sırasını eksiksiz karşılayıp karşılamadığını kontrol eder. */
export function framesMatchChallenge(frames: FaceFrame[], challenge: LivenessChallenge): boolean {
  if (frames.length !== challenge.actions.length) return false;
  return challenge.actions.every((a, i) => frames[i]?.action === a && typeof frames[i].image === 'string' && frames[i].image.length > 0);
}

/**
 * Geliştirme/test sağlayıcısı. Gerçek biyometrik karşılaştırma yapmaz; yapılandırılmış sabit skoru döner.
 * Kare içeriği "FAIL" ile başlıyorsa düşük skor üretir (test senaryoları için).
 */
export class MockFaceProvider implements FaceVerificationProvider {
  readonly name = 'mock';
  constructor(private readonly score: number) {}

  async verify(referenceImage: string, frames: FaceFrame[], challenge: LivenessChallenge): Promise<FaceVerificationResult> {
    if (!referenceImage) return { score: 0, livenessPassed: false, provider: this.name, reason: 'Referans yüz kaydı yok' };
    if (!framesMatchChallenge(frames, challenge)) {
      return { score: 0, livenessPassed: false, provider: this.name, reason: 'Kareler zorluk dizisiyle eşleşmiyor' };
    }
    const forcedFail = frames.some((f) => f.image.startsWith('FAIL'));
    return { score: forcedFail ? 0.2 : this.score, livenessPassed: !forcedFail, provider: this.name };
  }
}

export function createFaceProvider(kind: 'mock', mockScore: number): FaceVerificationProvider {
  switch (kind) {
    case 'mock':
    default:
      return new MockFaceProvider(mockScore);
  }
}
