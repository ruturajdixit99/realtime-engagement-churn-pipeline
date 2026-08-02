"use client";

import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { FeatureComparisonRow } from "@/lib/types";

const FEATURE_LABELS: Record<string, string> = {
  sessions_count: "Sessions this week",
  tracks_count: "Tracks played (trailing 4wk)",
  unique_artists: "Unique artists (trailing 4wk)",
  repeat_rate: "Repeat-listen rate",
  avg_session_minutes: "Avg session length",
  rolling_sessions: "Rolling session count",
  rolling_tracks: "Rolling track count",
  rolling_unique_artists: "Rolling unique artists",
  rolling_repeat_rate: "Rolling repeat rate",
  sessions_trend: "Session trend (slope)",
  weeks_since_active: "Weeks of prior silence",
};

interface Item {
  feature: string;
  label: string;
  pctOfNormal: number;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as Item;
  return (
    <div className="card px-3 py-2 text-sm shadow-lg">
      <div className="font-medium text-[var(--text-primary)]">{p.label}</div>
      <div className="text-secondary">
        {p.pctOfNormal < 100
          ? `${p.pctOfNormal.toFixed(0)}% of a retained user's level`
          : `${p.pctOfNormal.toFixed(0)}% of a retained user's level (higher)`}
      </div>
    </div>
  );
}

export default function FeatureRatioChart({ data }: { data: FeatureComparisonRow[] }) {
  const items: Item[] = data
    .filter((d) => d.feature !== "weeks_since_active" && d.retained_avg !== 0)
    .map((d) => ({
      feature: d.feature,
      label: FEATURE_LABELS[d.feature] ?? d.feature,
      pctOfNormal: (d.disengaging_avg / d.retained_avg) * 100,
    }))
    .sort((a, b) => a.pctOfNormal - b.pctOfNormal);

  return (
    <ResponsiveContainer width="100%" height={Math.max(220, items.length * 32)}>
      <BarChart data={items} layout="vertical" margin={{ left: 8, right: 32, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--gridline)" />
        <XAxis
          type="number"
          tickFormatter={(v) => `${v}%`}
          stroke="var(--axis)"
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={190}
          stroke="var(--axis)"
          tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
        />
        <ReferenceLine x={100} stroke="var(--axis)" strokeDasharray="3 3" />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
        <Bar dataKey="pctOfNormal" radius={[0, 4, 4, 0]} maxBarSize={18}>
          {items.map((_, i) => (
            <Cell key={i} fill="var(--series-1)" />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
