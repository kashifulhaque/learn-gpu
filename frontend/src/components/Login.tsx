import { useEffect } from "react";
import { Mark } from "./Brand";

function portalUrl(): string {
  const portal = new URL("https://auth.ifkash.dev/");
  portal.searchParams.set("rd", window.location.href);
  portal.searchParams.set("rm", "GET");
  return portal.toString();
}

export default function Login() {
  const destination = portalUrl();

  useEffect(() => {
    window.location.assign(destination);
  }, [destination]);

  return (
    <div className="flex min-h-full flex-col bg-paper px-5">
      <div className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center py-12">
        <div className="mb-10 flex items-center gap-2.5">
          <Mark className="size-7" />
          <span className="text-[15px] font-semibold tracking-tight text-fg">learn-gpu</span>
        </div>
        <h1 className="font-serif text-[2rem] font-semibold leading-[1.15] tracking-[-0.01em] text-fg">
          GPU programming for ML
        </h1>
        <p className="mt-3 text-[14px] leading-6 text-fg-subtle">
          Sign in through the central portal to continue to this course.
        </p>
        <a
          href={destination}
          className="mt-9 inline-flex h-10 items-center justify-center gap-2 rounded-md bg-accent px-4 text-[14px] font-medium text-on-accent"
        >
          Continue to sign in
          <span aria-hidden>→</span>
        </a>
      </div>
    </div>
  );
}
