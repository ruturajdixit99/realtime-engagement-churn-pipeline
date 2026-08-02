export default function StatTile({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-3xl font-semibold" style={{ fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sublabel && <div className="mt-1 text-sm text-secondary">{sublabel}</div>}
    </div>
  );
}
