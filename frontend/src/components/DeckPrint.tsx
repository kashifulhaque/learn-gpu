import SlideFrame from "./SlideFrame";
import type { Slide } from "../lib/api";

/**
 * A deck as pages: one 16:9 slide per sheet, with every question's answer
 * showing. It prints on its own named page size (see `.slides-print` in
 * index.css), so printing the site's other pages is unaffected.
 *
 * The slide pages show on screen too, stacked, so the print preview and the
 * browser's own print dialog agree.
 */
export default function DeckPrint({ slides, label }: { slides: Slide[]; label?: string }) {
  return (
    <div className="slides-print">
      {slides.map((slide, index) => (
        <div
          key={slide.id}
          className="slide-stage print-slide"
          data-where={label ? `${label} · ${slide.id}` : slide.id}
          aria-label={`Slide ${index + 1} of ${slides.length}`}
        >
          <SlideFrame slide={slide} variant="print" />
        </div>
      ))}
    </div>
  );
}
