import { ReplayEvent, UserFeatureVector, UserRiskState } from "./types";
import { scoreFeatureVector } from "./scoreModel";

const SESSION_GAP_MS = 30 * 60 * 1000; // 30 minutes, matches src/config.py SESSION_GAP_MINUTES
const ROLLING_WINDOW_MS = 4 * 7 * 24 * 60 * 60 * 1000; // 4 weeks, matches ROLLING_WEEKS
const ROLLING_WEEKS = 4;

interface TrackEvent {
  t: number;
  artist: string;
  track: string;
}

interface Session {
  start: number;
  end: number;
}

/**
 * Client-side port of ml/src/stream_simulator.py's UserState: maintains
 * incrementally-updated rolling engagement state per user as real events
 * arrive during the replay, using trailing time windows (documented in the
 * README as an approximation of the calendar-week-bucketed features the
 * model was trained on -- the same real train/serve skew the Python
 * Streamlit version has, ported faithfully rather than hidden).
 */
export class UserState {
  lastEventTime: number | null = null;
  lastSessionStart: number | null = null;
  priorActiveTime: number | null = null;
  trackEvents: TrackEvent[] = [];
  sessions: Session[] = [];

  private prune(now: number) {
    const cutoff = now - ROLLING_WINDOW_MS;
    while (this.trackEvents.length && this.trackEvents[0].t < cutoff) this.trackEvents.shift();
    while (this.sessions.length && this.sessions[0].end < cutoff) this.sessions.shift();
  }

  processEvent(t: number, artist: string, track: string): UserFeatureVector {
    const isNewSession =
      this.lastEventTime === null || t - this.lastEventTime > SESSION_GAP_MS;

    if (isNewSession) {
      if (this.lastSessionStart !== null) {
        this.priorActiveTime = this.lastEventTime;
      }
      this.lastSessionStart = t;
      this.sessions.push({ start: t, end: t });
    } else {
      this.sessions[this.sessions.length - 1].end = t;
    }

    this.trackEvents.push({ t, artist, track });
    this.lastEventTime = t;
    this.prune(t);

    return this.featureVector();
  }

  featureVector(): UserFeatureVector {
    const n = this.trackEvents.length;
    const artists = new Set(this.trackEvents.map((e) => e.artist));
    const tracks = this.trackEvents.map((e) => e.track);
    const uniqueTracks = new Set(tracks).size;
    const repeatRate = n ? (n - uniqueTracks) / n : 0;

    const durations = this.sessions.map((s) => (s.end - s.start) / 60000);
    const avgSessionMinutes = durations.length
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0;

    const weekAgo = (this.lastEventTime ?? 0) - 7 * 24 * 60 * 60 * 1000;
    const sessionsThisWeek = this.sessions.filter((s) => s.start >= weekAgo).length;

    let weeksSinceActive = 0;
    if (this.priorActiveTime !== null && this.lastEventTime !== null) {
      weeksSinceActive = Math.max(
        0,
        (this.lastEventTime - this.priorActiveTime) / (7 * 24 * 60 * 60 * 1000)
      );
    }

    const rollingSessions = this.sessions.length;

    return {
      sessions_count: sessionsThisWeek,
      tracks_count: n,
      unique_artists: artists.size,
      repeat_rate: repeatRate,
      avg_session_minutes: avgSessionMinutes,
      rolling_sessions: rollingSessions,
      rolling_tracks: n,
      rolling_unique_artists: artists.size,
      rolling_repeat_rate: repeatRate,
      sessions_trend: rollingSessions ? rollingSessions / ROLLING_WEEKS : 0,
      weeks_since_active: weeksSinceActive,
    };
  }
}

export class ReplayEngine {
  private userStates = new Map<string, UserState>();

  reset() {
    this.userStates = new Map();
  }

  process(event: ReplayEvent): UserRiskState {
    if (!this.userStates.has(event.u)) {
      this.userStates.set(event.u, new UserState());
    }
    const state = this.userStates.get(event.u)!;
    const t = new Date(event.t).getTime();
    const features = state.processEvent(t, event.a, event.tr);
    const churnProbability = scoreFeatureVector(features);

    return {
      user_id: event.u,
      churn_probability: churnProbability,
      last_event_time: event.t,
      ...features,
    };
  }

  allRiskStates(): UserRiskState[] {
    const out: UserRiskState[] = [];
    for (const [userId, state] of this.userStates) {
      if (state.lastEventTime === null) continue;
      const features = state.featureVector();
      out.push({
        user_id: userId,
        churn_probability: scoreFeatureVector(features),
        last_event_time: new Date(state.lastEventTime).toISOString(),
        ...features,
      });
    }
    return out;
  }
}
