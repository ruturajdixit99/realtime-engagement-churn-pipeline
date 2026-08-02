import { ModelComparison } from "@/lib/types";

const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
};

export default function ModelComparisonTable({ data }: { data: ModelComparison }) {
  const rows = Object.entries(data.results);
  const bestAuc = Math.max(...rows.map(([, m]) => m.roc_auc));

  return (
    <div className="card overflow-x-auto p-5">
      <div className="mb-3 text-sm text-secondary">
        Trained on {data.n_train_users} real users ({data.n_train_rows.toLocaleString()} labeled
        weeks), evaluated on {data.n_test_users} held-out real users (
        {data.n_test_rows.toLocaleString()} weeks, {(data.disengagement_rate * 100).toFixed(2)}%
        real disengagement rate).
      </div>
      <table className="w-full text-sm" style={{ fontVariantNumeric: "tabular-nums" }}>
        <thead>
          <tr className="border-b" style={{ borderColor: "var(--gridline)" }}>
            <th className="py-2 pr-4 text-left font-medium text-muted">Model</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">ROC-AUC</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">PR-AUC</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">Precision</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">Recall</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">F1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, m]) => (
            <tr key={name} className="border-b last:border-0" style={{ borderColor: "var(--gridline)" }}>
              <td className="py-2 pr-4 font-medium">
                {MODEL_LABELS[name] ?? name}
                {m.roc_auc === bestAuc && (
                  <span
                    className="ml-2 rounded-full px-2 py-0.5 text-xs font-normal"
                    style={{ background: "var(--status-good)", color: "white" }}
                  >
                    best ROC-AUC
                  </span>
                )}
              </td>
              <td className="py-2 pr-4 text-right">{m.roc_auc.toFixed(4)}</td>
              <td className="py-2 pr-4 text-right">{m.pr_auc.toFixed(4)}</td>
              <td className="py-2 pr-4 text-right">{m.precision.toFixed(4)}</td>
              <td className="py-2 pr-4 text-right">{m.recall.toFixed(4)}</td>
              <td className="py-2 pr-4 text-right">{m.f1.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 text-xs text-muted">
        PR-AUC is the more honest headline metric here — the real no-skill baseline under this
        class imbalance is {data.disengagement_rate.toFixed(4)}, not 0.5.
      </div>
    </div>
  );
}
