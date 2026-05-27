use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;

/// Run summary stored in SQLite
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub id: String,
    pub suite_name: String,
    pub model: String,
    pub timestamp: String,
    pub is_baseline: bool,
    pub total_tests: u32,
    pub passed_tests: u32,
    pub avg_score: f64,
    pub total_tokens: u64, // sum of all input+output tokens in this run
}

/// Per-test result stored in SQLite
#[derive(Debug, Clone)]
pub struct TestRecord {
    pub run_id: String,
    pub test_name: String,
    pub score: f64,
    pub passed: bool,
    pub pass_rate: f64, // fraction of n_runs that passed (1.0 when n_runs=1)
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub reason: String,
}

/// Baseline record
#[derive(Debug, Clone)]
pub struct BaselineRecord {
    pub id: String,
    pub timestamp: String,
}

pub struct Db {
    conn: Connection,
    path: PathBuf,
}

impl Db {
    pub fn open() -> Result<Self> {
        let path = data_dir().join("crucible.db");
        std::fs::create_dir_all(path.parent().unwrap())?;

        let conn = Connection::open(&path)
            .with_context(|| format!("Cannot open database at {}", path.display()))?;

        // Create tables (idempotent)
        conn.execute_batch(SCHEMA)?;

        // Migrate existing databases: add new columns if they don't exist yet.
        // SQLite ALTER TABLE ADD COLUMN errors when column already exists — we
        // intentionally ignore those errors so this is safe to run every startup.
        let _ = conn.execute(
            "ALTER TABLE runs ADD COLUMN total_tokens INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE results ADD COLUMN pass_rate REAL NOT NULL DEFAULT 1.0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE results ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE results ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0",
            [],
        );

        Ok(Self { conn, path })
    }

    pub fn path(&self) -> String {
        self.path.display().to_string()
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    pub fn insert_run(&self, r: &RunRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO runs (id, suite_name, model, timestamp, is_baseline,
                              total_tests, passed_tests, avg_score, total_tokens)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                r.id,
                r.suite_name,
                r.model,
                r.timestamp,
                r.is_baseline as i32,
                r.total_tests,
                r.passed_tests,
                r.avg_score,
                r.total_tokens,
            ],
        )?;
        Ok(())
    }

    pub fn insert_test_result(&self, t: &TestRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO results
             (run_id, test_name, score, passed, pass_rate, latency_ms,
              input_tokens, output_tokens, reason)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                t.run_id,
                t.test_name,
                t.score,
                t.passed as i32,
                t.pass_rate,
                t.latency_ms,
                t.input_tokens,
                t.output_tokens,
                t.reason,
            ],
        )?;
        Ok(())
    }

    /// Mark `run_id` as the baseline for its suite.
    /// Clears the baseline flag only for other runs in the SAME suite —
    /// other suites' baselines are unaffected.
    pub fn set_baseline(&self, run_id: &str, suite_name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE runs SET is_baseline = 0 WHERE suite_name = ?1",
            params![suite_name],
        )?;
        self.conn.execute(
            "UPDATE runs SET is_baseline = 1 WHERE id = ?1",
            params![run_id],
        )?;
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Fetch the current baseline for a specific suite.
    pub fn current_baseline(&self, suite_name: &str) -> Result<Option<BaselineRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp FROM runs
             WHERE is_baseline = 1 AND suite_name = ?1
             LIMIT 1",
        )?;
        let mut rows = stmt.query(params![suite_name])?;
        if let Some(row) = rows.next()? {
            Ok(Some(BaselineRecord {
                id: row.get(0)?,
                timestamp: row.get(1)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Look up which suite a specific run belongs to.
    pub fn suite_for_run(&self, run_id: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT suite_name FROM runs WHERE id = ?1 LIMIT 1")?;
        let mut rows = stmt.query(params![run_id])?;
        Ok(rows.next()?.map(|r| r.get(0).unwrap()))
    }

    /// Return all current baselines (one per suite).
    pub fn all_baselines(&self) -> Result<Vec<(String, BaselineRecord)>> {
        let mut stmt = self.conn.prepare(
            "SELECT suite_name, id, timestamp FROM runs WHERE is_baseline = 1 ORDER BY suite_name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                BaselineRecord {
                    id: row.get(1)?,
                    timestamp: row.get(2)?,
                },
            ))
        })?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    pub fn last_run_id(&self) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM runs ORDER BY timestamp DESC LIMIT 1")?;
        let mut rows = stmt.query([])?;
        Ok(rows.next()?.map(|r| r.get(0).unwrap()))
    }

    pub fn get_results_for_run(&self, run_id: &str) -> Result<Vec<TestRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT run_id, test_name, score, passed,
                    COALESCE(pass_rate, 1.0),
                    latency_ms,
                    COALESCE(input_tokens, 0),
                    COALESCE(output_tokens, 0),
                    reason
             FROM results WHERE run_id = ?1",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(TestRecord {
                run_id: row.get(0)?,
                test_name: row.get(1)?,
                score: row.get(2)?,
                passed: row.get::<_, i32>(3)? != 0,
                pass_rate: row.get(4)?,
                latency_ms: row.get(5)?,
                input_tokens: row.get(6)?,
                output_tokens: row.get(7)?,
                reason: row.get(8)?,
            })
        })?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    pub fn recent_runs(&self, limit: usize, suite: Option<&str>) -> Result<Vec<RunRecord>> {
        let sql = if suite.is_some() {
            "SELECT id, suite_name, model, timestamp, is_baseline,
                    total_tests, passed_tests, avg_score,
                    COALESCE(total_tokens, 0)
             FROM runs WHERE suite_name = ?1
             ORDER BY timestamp DESC LIMIT ?2"
        } else {
            "SELECT id, suite_name, model, timestamp, is_baseline,
                    total_tests, passed_tests, avg_score,
                    COALESCE(total_tokens, 0)
             FROM runs ORDER BY timestamp DESC LIMIT ?2"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![suite.unwrap_or(""), limit as i64], |row| {
            Ok(RunRecord {
                id: row.get(0)?,
                suite_name: row.get(1)?,
                model: row.get(2)?,
                timestamp: row.get(3)?,
                is_baseline: row.get::<_, i32>(4)? != 0,
                total_tests: row.get(5)?,
                passed_tests: row.get(6)?,
                avg_score: row.get(7)?,
                total_tokens: row.get(8)?,
            })
        })?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    pub fn run_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))?)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    dirs_next::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("crucible")
}

mod dirs_next {
    use std::path::PathBuf;
    pub fn data_local_dir() -> Option<PathBuf> {
        #[cfg(target_os = "macos")]
        return std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Application Support"));
        #[cfg(target_os = "linux")]
        return std::env::var("XDG_DATA_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".local/share"))
            });
        #[cfg(target_os = "windows")]
        return std::env::var("LOCALAPPDATA").ok().map(PathBuf::from);
        #[allow(unreachable_code)]
        None
    }
}

// ── Schema ────────────────────────────────────────────────────────────────────

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    suite_name    TEXT NOT NULL,
    model         TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    is_baseline   INTEGER NOT NULL DEFAULT 0,
    total_tests   INTEGER NOT NULL DEFAULT 0,
    passed_tests  INTEGER NOT NULL DEFAULT 0,
    avg_score     REAL    NOT NULL DEFAULT 0.0,
    total_tokens  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL REFERENCES runs(id),
    test_name     TEXT    NOT NULL,
    score         REAL    NOT NULL,
    passed        INTEGER NOT NULL,
    pass_rate     REAL    NOT NULL DEFAULT 1.0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reason        TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_results_run   ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_baseline ON runs(is_baseline);
CREATE INDEX IF NOT EXISTS idx_runs_suite    ON runs(suite_name);
";
