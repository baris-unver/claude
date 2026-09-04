import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pg from 'pg';

const here = path.dirname(fileURLToPath(import.meta.url));

/** `migrations/` klasöründeki SQL dosyalarını sırayla, bir kez uygular. */
export async function runMigrations(pool: pg.Pool): Promise<string[]> {
  await pool.query(
    'CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())',
  );
  const dir = path.join(here, 'migrations');
  const files = (await readdir(dir)).filter((f) => f.endsWith('.sql')).sort();
  const applied = new Set((await pool.query<{ name: string }>('SELECT name FROM schema_migrations')).rows.map((r) => r.name));
  const done: string[] = [];
  for (const f of files) {
    if (applied.has(f)) continue;
    const sql = await readFile(path.join(dir, f), 'utf8');
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query(sql);
      await client.query('INSERT INTO schema_migrations(name) VALUES ($1)', [f]);
      await client.query('COMMIT');
      done.push(f);
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  }
  return done;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.error('DATABASE_URL tanımlı değil');
    process.exit(1);
  }
  const pool = new pg.Pool({ connectionString: url });
  runMigrations(pool)
    .then((d) => {
      console.log(d.length ? `Uygulanan göçler: ${d.join(', ')}` : 'Şema güncel');
      return pool.end();
    })
    .catch((e) => {
      console.error(e);
      process.exit(1);
    });
}
