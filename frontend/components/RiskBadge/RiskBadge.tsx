// components/RiskBadge/RiskBadge.tsx
// Color-coded risk label badge with optional pulse animation for critical risks

interface RiskBadgeProps {
  riskLabel: string;
  severity: string;
  large?: boolean;
}

const BADGE_STYLES: Record<string, string> = {
  Safe:           "bg-emerald-900/60 border-emerald-500 text-emerald-300",
  "Adjust Dosage":"bg-amber-900/60 border-amber-500 text-amber-300",
  Toxic:          "bg-red-900/60 border-red-500 text-red-300",
  Ineffective:    "bg-red-900/60 border-red-500 text-red-300",
};

const DOT_STYLES: Record<string, string> = {
  Safe:           "bg-emerald-400",
  "Adjust Dosage":"bg-amber-400",
  Toxic:          "bg-red-400",
  Ineffective:    "bg-red-400",
};

const PULSE_LABELS = new Set(["Toxic", "Ineffective"]);

export function RiskBadge({ riskLabel, severity, large }: RiskBadgeProps) {
  const style    = BADGE_STYLES[riskLabel] ?? "bg-slate-800 border-slate-600 text-slate-300";
  const dotStyle = DOT_STYLES[riskLabel]   ?? "bg-slate-400";
  const pulse    = PULSE_LABELS.has(riskLabel);

  return (
    <span
      className={[
        "inline-flex items-center gap-2 rounded-full border font-semibold",
        large ? "px-5 py-2 text-base" : "px-3 py-1 text-xs",
        style,
      ].join(" ")}
    >
      <span className="relative flex h-2.5 w-2.5">
        {pulse && (
          <span
            className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${dotStyle}`}
          />
        )}
        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${dotStyle}`} />
      </span>
      {riskLabel}
    </span>
  );
}
