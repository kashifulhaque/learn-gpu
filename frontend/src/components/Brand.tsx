/** The mark from the favicon, drawn in the accent colour. */
export function Mark({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={`shrink-0 ${className}`} aria-hidden>
      <rect width="64" height="64" rx="14" className="fill-accent" />
      <path
        d="M19 15 Q27 13 30.5 20 L46.5 50"
        fill="none"
        className="stroke-on-accent"
        strokeWidth="7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M34 28 L17.5 50" fill="none" className="stroke-on-accent" strokeWidth="7" strokeLinecap="round" />
    </svg>
  );
}
