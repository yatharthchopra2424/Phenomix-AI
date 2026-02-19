"use client";
// components/AccordionSection/AccordionSection.tsx
// Accessible collapsible section using Headless UI Disclosure

import { Disclosure, DisclosureButton, DisclosurePanel } from "@headlessui/react";
import { ChevronDown } from "lucide-react";
import { ReactNode } from "react";

interface AccordionSectionProps {
  title: string;
  badge?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function AccordionSection({
  title,
  badge,
  defaultOpen = false,
  children,
}: AccordionSectionProps) {
  return (
    <Disclosure defaultOpen={defaultOpen} as="div" className="border border-slate-700 rounded-lg overflow-hidden">
      <DisclosureButton className="w-full flex items-center justify-between gap-3 px-4 py-3 bg-slate-800 hover:bg-slate-750 transition-colors text-left">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">{title}</span>
          {badge}
        </div>
        <ChevronDown className="h-4 w-4 text-slate-400 ui-open:rotate-180 transition-transform flex-shrink-0" />
      </DisclosureButton>
      <DisclosurePanel className="px-4 py-3 bg-slate-900 text-sm text-slate-300 space-y-2">
        {children}
      </DisclosurePanel>
    </Disclosure>
  );
}
