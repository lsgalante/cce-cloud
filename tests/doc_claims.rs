//! CLAUDE.md, checked against the crate it describes.
//!
//! An integration test on purpose: anything under `src/` would be counted by
//! its own scans — a `#[cfg(test)]` block added here to measure which modules
//! carry tests promptly made this module one of them, and the line-count
//! figures would drift by however long this file happens to be.
//!
//! Deliberately MIRRORED per crate rather than shared from a helper: each is
//! its own git repository that must build standalone, and this needs nothing
//! but std.

/// CLAUDE.md's `(~Nk lines)` figures, checked against the files they describe.
///
/// Those numbers exist to set expectations before opening a file — "this is
/// the big one" — and they drift in total silence, because a stale number
/// reads exactly like a fresh one. A workspace sweep on 2026-09-19 found
/// EVERY size figure in every CLAUDE.md stale, all of them undercounts, the
/// worst by 49% (cce-designer's app.rs, written ~5.4k at 8046 lines).
///
/// Tolerance is 10%: loose enough that ordinary work does not trip it, tight
/// enough that a file cannot quietly double. When it fails, write the number
/// it reports — that is the whole fix.
///
/// Deliberately MIRRORED into each crate that carries such a figure rather
/// than shared from a helper: every crate here is its own git repository and
/// must build standalone, and this needs nothing but `std`. Same call
/// `ramp.rs` makes about its cce-ui parser.
pub mod size_claims {
    use std::path::{Path, PathBuf};

    /// One `(~Nk lines)` claim: the name as CLAUDE.md spells it, the figure,
    /// and whether the claim also calls it the largest file.
    fn claims(doc: &str) -> Vec<(String, f64, bool)> {
        const TAIL: &str = " lines)";
        let mut out = Vec::new();
        let mut i = 0;
        while let Some(p) = doc[i..].find(TAIL) {
            let end = i + p;
            i = end + TAIL.len();
            let Some(open) = doc[..end].rfind('(') else { continue };
            let inner = &doc[open + 1..end];
            // "~8k", or "largest file, ~6.9k" — take the last word.
            let largest = inner.contains("largest file");
            let word = inner.rsplit([' ', ',']).next().unwrap_or("").trim();
            let digits = word.trim_start_matches('~');
            let value = match digits.strip_suffix('k') {
                Some(k) => k.parse::<f64>().ok().map(|v| v * 1000.0),
                None => digits.parse::<f64>().ok(),
            };
            // The backticked name immediately before the parenthetical.
            let before = &doc[..open];
            let name = before.rfind('`').and_then(|e| {
                before[..e].rfind('`').map(|s| before[s + 1..e].to_string())
            });
            if let (Some(v), Some(n)) = (value, name) {
                if v > 0.0 {
                    out.push((n, v, largest));
                }
            }
        }
        out
    }

    /// Every `.rs` file under `src/`, as (path, line count).
    fn sources(root: &Path) -> Vec<(PathBuf, usize)> {
        fn walk(dir: &Path, out: &mut Vec<(PathBuf, usize)>) {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                    if let Ok(s) = std::fs::read_to_string(&p) {
                        out.push((p, s.lines().count()));
                    }
                }
            }
        }
        let mut v = Vec::new();
        walk(&root.join("src"), &mut v);
        v
    }

    #[test]
    fn test_claude_md_line_counts_match_the_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let doc = std::fs::read_to_string(root.join("CLAUDE.md"))
            .expect("CLAUDE.md is missing next to Cargo.toml");
        let files = sources(root);
        let claims = claims(&doc);
        assert!(
            !claims.is_empty(),
            "no `(~N lines)` figure found in CLAUDE.md — either the syntax changed \
             and this scan needs updating, or the figures were removed and so \
             should this test"
        );

        let mut bad: Vec<String> = Vec::new();
        for (name, claimed, largest) in &claims {
            // A path relative to the crate root, else a unique basename.
            let hits: Vec<&(PathBuf, usize)> = if root.join(name).is_file() {
                files.iter().filter(|(p, _)| *p == root.join(name)).collect()
            } else {
                files
                    .iter()
                    .filter(|(p, _)| p.file_name().and_then(|x| x.to_str()) == Some(name.as_str()))
                    .collect()
            };
            let [(path, actual)] = hits[..] else {
                bad.push(format!("`{name}`: names {} files under src/, cannot check", hits.len()));
                continue;
            };
            let actual = *actual as f64;
            let drift = (actual - claimed) / claimed;
            if drift.abs() > 0.10 {
                bad.push(format!(
                    "`{name}` is documented as ~{} lines but has {} ({:+.0}%) — write ~{}",
                    round_k(*claimed),
                    actual as usize,
                    drift * 100.0,
                    round_k(actual)
                ));
            }
            if *largest {
                if let Some((big, n)) = files.iter().max_by_key(|(_, n)| *n) {
                    if big != path {
                        bad.push(format!(
                            "`{name}` is called the largest file, but {} has {n} lines",
                            big.strip_prefix(root).unwrap_or(big).display()
                        ));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "CLAUDE.md size claims are stale:\n  {}", bad.join("\n  "));
    }

    /// "~8k" for 8046, "~3.1k" for 3134, "~950" for 950 — the spelling the
    /// docs already use, so the failure message can be pasted straight in.
    fn round_k(n: f64) -> String {
        if n < 1000.0 {
            return format!("{}", n.round() as usize);
        }
        let k = n / 1000.0;
        if (k - k.round()).abs() < 0.05 {
            format!("{}k", k.round() as usize)
        } else {
            format!("{k:.1}k")
        }
    }
}
