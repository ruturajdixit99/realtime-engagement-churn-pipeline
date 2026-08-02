import Link from "next/link";
import engagementSummary from "../../public/data/engagement_summary.json";
import modelComparison from "../../public/data/model_comparison.json";
import FeatureRatioChart from "@/components/FeatureRatioChart";
import ModelComparisonTable from "@/components/ModelComparisonTable";
import StatTile from "@/components/StatTile";
import { EngagementSummary, ModelComparison } from "@/lib/types";

const summary = engagementSummary as EngagementSummary;
const models = modelComparison as ModelComparison;

export default function DashboardPage() {
  const bestModelKey = Object.entries(models.results).reduce((a, b) =>
    b[1].roc_auc > a[1].roc_auc ? b : a
  )[0];
  const [startDate, endDate] = summary.date_range;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <header className="mb-10">
        <div className="text-xs font-medium uppercase tracking-wide text-muted">
          Real data · Real sessions · Real disengagement labels
        </div>
        <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">
          Real-Time Engagement &amp; Churn Pipeline
        </h1>
        <p className="mt-3 max-w-2xl text-secondary">
          Trained on real timestamped listening events from the{" "}
          <a
            className="underline decoration-dotted underline-offset-2"
            href="http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html"
            target="_blank"
          >
            Last.fm-1K Users dataset
          </a>{" "}
          — structurally the same signal a subscription product like Spotify would use to spot a
          subscriber about to go quiet.
        </p>
        <div className="mt-6 flex gap-3">
          <Link
            href="/replay"
            className="rounded-lg px-4 py-2 text-sm font-medium text-white"
            style={{ background: "var(--series-1)" }}
          >
            Watch the live replay →
          </Link>
          <a
            href="https://github.com/ruturajdixit99/realtime-engagement-churn-pipeline"
            target="_blank"
            className="rounded-lg border px-4 py-2 text-sm font-medium"
            style={{ borderColor: "var(--border-hairline)" }}
          >
            View source
          </a>
        </div>
      </header>

      <section className="mb-10 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="Real users" value={summary.n_users.toLocaleString()} />
        <StatTile label="Real events" value={summary.n_events.toLocaleString()} />
        <StatTile label="Real sessions" value={summary.n_sessions.toLocaleString()} />
        <StatTile
          label="Disengagement rate"
          value={`${(summary.disengagement_rate * 100).toFixed(2)}%`}
          sublabel={`of ${summary.n_labeled_user_weeks.toLocaleString()} labeled active weeks`}
        />
      </section>

      <section className="mb-10">
        <div className="card p-5 text-sm text-secondary">
          Data spans {new Date(startDate).toISOString().slice(0, 10)} to{" "}
          {new Date(endDate).toISOString().slice(0, 10)}. Best model:{" "}
          <strong className="text-[var(--text-primary)]">
            {bestModelKey.replace("_", " ")}
          </strong>{" "}
          (ROC-AUC {models.results[bestModelKey].roc_auc.toFixed(3)}).
        </div>
      </section>

      <section className="mb-10">
        <h2 className="mb-1 text-xl font-semibold">What disengagement actually looks like</h2>
        <p className="mb-4 text-sm text-secondary">
          Real average engagement in a user-week that turned out to be followed by 4+ silent
          weeks, vs. a week that stayed active — shown as % of a retained user's typical level.
        </p>
        <div className="card p-5">
          <FeatureRatioChart data={summary.feature_comparison} />
        </div>
      </section>

      <section className="mb-10">
        <h2 className="mb-1 text-xl font-semibold">Model comparison</h2>
        <p className="mb-4 text-sm text-secondary">
          Held-out real users (never seen in training) — a genuinely hard, severely imbalanced
          real-world problem.
        </p>
        <ModelComparisonTable data={models} />
      </section>

      <footer className="mt-16 border-t pt-6 text-sm text-muted" style={{ borderColor: "var(--gridline)" }}>
        Dataset: Last.fm-1K Users (real, public, UPF/Celma 2010). Pipeline: pandas · scikit-learn.
        The live replay runs the actual trained Logistic Regression model, ported exactly to
        TypeScript — see{" "}
        <Link href="/replay" className="underline decoration-dotted underline-offset-2">
          /replay
        </Link>
        .
      </footer>
    </main>
  );
}
