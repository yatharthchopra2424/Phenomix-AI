"use client";
// components/DrugSelector/DrugSelector.tsx
// Multi-select combobox for supported pharmacogenomic drugs

import { useState } from "react";
import { Check, ChevronDown, X } from "lucide-react";

const SUPPORTED_DRUGS = [
  { label: "Codeine",      gene: "CYP2D6"  },
  { label: "Warfarin",     gene: "CYP2C9"  },
  { label: "Clopidogrel",  gene: "CYP2C19" },
  { label: "Simvastatin",  gene: "SLCO1B1" },
  { label: "Azathioprine", gene: "TPMT"    },
  { label: "Fluorouracil", gene: "DPYD"    },
];

interface DrugSelectorProps {
  selected: string[];
  onChange: (drugs: string[]) => void;
  disabled?: boolean;
}

export function DrugSelector({ selected, onChange, disabled }: DrugSelectorProps) {
  const [open, setOpen] = useState(false);

  const toggle = (drugLabel: string) => {
    if (selected.includes(drugLabel)) {
      onChange(selected.filter((d) => d !== drugLabel));
    } else {
      onChange([...selected, drugLabel]);
    }
  };

  const remove = (drugLabel: string) => {
    onChange(selected.filter((d) => d !== drugLabel));
  };

  return (
    <div className="relative w-full">
      <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
        Target Drug(s)
      </label>

      {/* Selected pills */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {selected.map((d) => (
            <span
              key={d}
              className="inline-flex items-center gap-1 rounded-full bg-cyan-900/60 border border-cyan-700 px-2.5 py-0.5 text-xs font-medium text-cyan-200"
            >
              {d}
              <button
                type="button"
                onClick={() => remove(d)}
                disabled={disabled}
                className="hover:text-red-300 transition-colors"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Dropdown trigger */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className={[
          "w-full flex items-center justify-between rounded-lg border px-4 py-2.5",
          "text-sm text-left transition-colors",
          disabled
            ? "cursor-not-allowed opacity-50 border-slate-700 bg-slate-800 text-slate-500"
            : "border-slate-600 bg-slate-800 text-slate-200 hover:border-cyan-600",
        ].join(" ")}
      >
        <span className={selected.length === 0 ? "text-slate-500" : ""}>
          {selected.length === 0 ? "Select drugs to analyse…" : `${selected.length} drug(s) selected`}
        </span>
        <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {/* Dropdown list */}
      {open && !disabled && (
        <ul className="absolute z-50 mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 shadow-2xl overflow-hidden">
          {SUPPORTED_DRUGS.map(({ label, gene }) => {
            const isSelected = selected.includes(label);
            return (
              <li key={label}>
                <button
                  type="button"
                  onClick={() => { toggle(label); }}
                  className={[
                    "w-full flex items-center justify-between px-4 py-2.5 text-sm",
                    "hover:bg-slate-800 transition-colors",
                    isSelected ? "text-cyan-300" : "text-slate-200",
                  ].join(" ")}
                >
                  <div>
                    <span className="font-medium">{label}</span>
                    <span className="ml-2 text-xs text-slate-500">({gene})</span>
                  </div>
                  {isSelected && <Check className="h-4 w-4 text-cyan-400" />}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {/* Click-away to close */}
      {open && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setOpen(false)}
        />
      )}
    </div>
  );
}
