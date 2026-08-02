"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ReplayEngine } from "@/lib/engagementEngine";
import { ReplayEvent, UserRiskState } from "@/lib/types";

const SPEED_OPTIONS = [
  { label: "50 events/tick", value: 50 },
  { label: "150 events/tick", value: 150 },
  { label: "400 events/tick", value: 400 },
  { label: "1000 events/tick", value: 1000 },
];
const TICK_MS = 200;
const FEED_SIZE = 30;

export default function ReplayPage() {
  const [events, setEvents] = useState<ReplayEvent[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [eventsPerTick, setEventsPerTick] = useState(150);
  const [cursor, setCursor] = useState(0);
  const [recentEvents, setRecentEvents] = useState<ReplayEvent[]>([]);
  const [riskStates, setRiskStates] = useState<UserRiskState[]>([]);

  const engineRef = useRef<ReplayEngine>(new ReplayEngine());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/data/replay_events.json")
      .then((r) => r.json())
      .then((data: ReplayEvent[]) => {
        setEvents(data);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!running || !events) return;
    intervalRef.current = setInterval(() => {
      setCursor((prevCursor) => {
        const engine = engineRef.current;
        const end = Math.min(prevCursor + eventsPerTick, events.length);
        const batch = events.slice(prevCursor, end);
        if (batch.length === 0) {
          setRunning(false);
          return prevCursor;
        }
        batch.forEach((ev) => engine.process(ev));
        setRecentEvents(batch.slice(-FEED_SIZE).reverse());
        setRiskStates(engine.allRiskStates());
        return end;
      });
    }, TICK_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, events, eventsPerTick]);

  function handleReset() {
    setRunning(false);
    setCursor(0);
    setRecentEvents([]);
    setRiskStates([]);
    engineRef.current.reset();
  }

  const total = events?.length ?? 0;
  const progress = total ? cursor / total : 0;
  const currentSimTime = recentEvents[0]?.t ?? null;
  const sortedRisk = [...riskStates].sort((a, b) => b.churn_probability - a.churn_probability).slice(0, 20);

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <Link href="/" className="text-sm text-secondary underline decoration-dotted underline-offset-2">
        ← Back to dashboard
      </Link>
      <h1 className="mt-3 text-3xl font-semibold">Live Replay</h1>
      <p className="mt-2 max-w-2xl text-secondary">
        Real events from 25 real Last.fm users, Jan–Jun 2009, replayed in true chronological
        order in your browser. Rolling engagement features and churn-risk scores are computed
        incrementally as each event arrives, using the actual trained model.
      </p>

      <div className="card mt-6 flex flex-wrap items-center gap-4 p-4">
        <button
          onClick={() => setRunning((r) => !r)}
          disabled={loading || cursor >= total}
          className="rounded-lg px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          style={{ background: "var(--series-1)" }}
        >
          {loading ? "Loading real events…" : running ? "Pause" : cursor >= total && total > 0 ? "Finished" : "Start"}
        </button>
        <button
          onClick={handleReset}
          className="rounded-lg border px-4 py-2 text-sm font-medium"
          style={{ borderColor: "var(--border-hairline)" }}
        >
          Reset
        </button>
        <label className="flex items-center gap-2 text-sm text-secondary">
          Speed
          <select
            value={eventsPerTick}
            onChange={(e) => setEventsPerTick(Number(e.target.value))}
            className="rounded-lg border bg-transparent px-2 py-1 text-sm"
            style={{ borderColor: "var(--border-hairline)" }}
          >
            {SPEED_OPTIONS.map((o) => (
              <option key={o.value} value={o.value} className="text-black">
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <div className="ml-auto text-sm text-muted">
          {cursor.toLocaleString()} / {total.toLocaleString()} events
        </div>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full" style={{ background: "var(--gridline)" }}>
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${progress * 100}%`, background: "var(--series-1)" }}
        />
      </div>
      {currentSimTime && (
        <div className="mt-2 text-xs text-muted">
          Simulated time: {new Date(currentSimTime).toISOString().replace("T", " ").slice(0, 19)} UTC
        </div>
      )}

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div className="card p-5">
          <div className="mb-3 text-sm font-medium">Live event feed</div>
          <div className="max-h-[420px] overflow-y-auto text-sm">
            {recentEvents.length === 0 && <div className="text-muted">Press Start to begin streaming.</div>}
            {recentEvents.map((ev, i) => (
              <div
                key={`${ev.t}-${ev.u}-${i}`}
                className="flex items-center justify-between border-b py-1.5 last:border-0"
                style={{ borderColor: "var(--gridline)" }}
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate">
                    <span className="text-muted">{ev.u}</span> — {ev.a} · {ev.tr}
                  </div>
                </div>
                <div className="ml-2 shrink-0 text-xs text-muted tabular-nums">
                  {ev.t.slice(11, 19)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <div className="mb-3 text-sm font-medium">Live churn-risk by user</div>
          <div className="max-h-[420px] overflow-y-auto">
            {sortedRisk.length === 0 && <div className="text-sm text-muted">No users scored yet.</div>}
            {sortedRisk.map((r) => (
              <div
                key={r.user_id}
                className="flex items-center gap-3 border-b py-2 last:border-0"
                style={{ borderColor: "var(--gridline)" }}
              >
                <div className="w-28 shrink-0 truncate text-sm text-secondary">{r.user_id}</div>
                <div className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: "var(--gridline)" }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(100, r.churn_probability * 100)}%`,
                      background:
                        r.churn_probability >= 0.5
                          ? "var(--status-critical)"
                          : r.churn_probability >= 0.25
                            ? "var(--status-warning)"
                            : "var(--status-good)",
                    }}
                  />
                </div>
                <div className="w-14 shrink-0 text-right text-sm tabular-nums">
                  {(r.churn_probability * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8 card p-5">
        <div className="mb-2 text-sm font-medium">What &quot;real-time&quot; means here</div>
        <p className="text-sm text-secondary">
          Every event above is a real row from the real Last.fm-1K dataset, replayed in true
          chronological order — nothing is generated. What&apos;s simulated: the pacing (events
          are processed in batches per tick so years of eventual full-dataset replay would be
          watchable — here it&apos;s a real 6-month window already), and the fact that this is a
          historical replay rather than a live production event bus. The rolling features are
          computed incrementally using trailing time windows as each event arrives — the same
          approach (and the same real train/serve approximation gap, documented in the README)
          as the Python/Streamlit version of this pipeline, ported faithfully to the browser
          instead of a server-side thread.
        </p>
      </div>
    </main>
  );
}
