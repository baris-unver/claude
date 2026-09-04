import { z } from 'zod';

const boolish = z
  .union([z.boolean(), z.string()])
  .transform((v) => (typeof v === 'boolean' ? v : ['1', 'true', 'yes', 'on'].includes(v.toLowerCase())));

const schema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().positive().default(3000),
  HOST: z.string().default('0.0.0.0'),
  DATABASE_URL: z.string().optional(),
  JWT_SECRET: z.string().min(32).default('dev-only-secret-do-not-use-in-production-0000'),
  ACCESS_TOKEN_TTL: z.string().default('15m'),
  REFRESH_TOKEN_TTL_DAYS: z.coerce.number().int().positive().default(30),
  ACTIVATION_CODE_TTL_HOURS: z.coerce.number().int().positive().default(72),
  SIGNATURE_MAX_SKEW_SEC: z.coerce.number().int().positive().default(300),
  LOCATION_MAX_ACCURACY_M: z.coerce.number().positive().default(100),
  HEARTBEAT_TIMEOUT_MIN: z.coerce.number().positive().default(30),
  VIOLATION_CONFIRM_SAMPLES: z.coerce.number().int().min(1).default(2),
  LOCATION_RETENTION_DAYS: z.coerce.number().int().positive().default(180),
  MONITOR_INTERVAL_SEC: z.coerce.number().int().positive().default(60),
  FACE_PROVIDER: z.enum(['mock']).default('mock'),
  FACE_MATCH_THRESHOLD: z.coerce.number().min(0).max(1).default(0.8),
  MOCK_FACE_SCORE: z.coerce.number().min(0).max(1).default(0.95),
  EXPO_PUSH_URL: z.string().url().default('https://exp.host/--/api/v2/push/send'),
  TIMEZONE: z.string().default('Europe/Istanbul'),
  SEED_DEMO: boolish.default(false),
  CORS_ORIGIN: z.string().default('*'),
});

export type Config = z.infer<typeof schema>;

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const parsed = schema.safeParse(env);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ');
    throw new Error(`Geçersiz yapılandırma: ${issues}`);
  }
  const cfg = parsed.data;
  if (cfg.NODE_ENV === 'production' && cfg.JWT_SECRET.startsWith('dev-only-secret')) {
    throw new Error('Üretim ortamında JWT_SECRET ayarlanmalıdır.');
  }
  return cfg;
}
