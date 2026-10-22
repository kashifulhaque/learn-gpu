import { Component, type ReactNode } from "react";
import { VIZ } from "./registry";

/** Keeps a figure that throws from taking the rest of the chapter with it. */
class Boundary extends Component<{ name: string; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error(`The figure ${this.props.name} failed to render`, error);
  }

  render() {
    if (this.state.failed) {
      return <div className="viz-missing">The figure “{this.props.name}” failed to render.</div>;
    }
    return this.props.children;
  }
}

/** The figure a ```viz fence names. */
export default function Viz({ name }: { name: string }) {
  const Figure = VIZ[name];
  if (!Figure) {
    return <div className="viz-missing">There is no figure named “{name}”.</div>;
  }
  return (
    <Boundary name={name}>
      <Figure />
    </Boundary>
  );
}
