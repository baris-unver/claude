import type { Config } from '../config.js';
import { MemoryStore } from './memory.js';
import { PgStore } from './pg.js';
import type { Store } from './types.js';

export type { Store } from './types.js';

export function createStore(cfg: Config): Store {
  if (cfg.DATABASE_URL) return new PgStore(cfg.DATABASE_URL);
  return new MemoryStore();
}
