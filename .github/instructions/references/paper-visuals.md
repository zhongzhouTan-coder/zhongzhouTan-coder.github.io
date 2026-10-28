# Paper Insight: Figures and Diagrams

## Diagrams

- Give every diagram one explicit teaching job and one primary reader question.
  If nearby prose or a table answers the same question equally well, omit the
  diagram.
- Search the paper extraction and captured web Markdown for relevant figures
  before creating a visual.
- Use original paper/web figures for the Big Picture, architecture, mechanism,
  ablation, result, and runtime sections whenever suitable figures exist.
- Preserve every selected source figure under the consuming page's local
  `assets/` directory; do not hotlink remote images as the only copy.
- Use Mermaid for **The Landscape**, save its editable `.mmd` source locally,
  and link that file from the page.
- Outside The Landscape, use a synthesized Mermaid, SVG/Draw.io, Excalidraw, or
  AI-generated visual only when there is no suitable source figure and the
  visual is explicitly requested or materially necessary.
- Every visual must have a caption explaining what the reader should see and
  whether it is an original source figure or a synthesized explanation.
- Source-figure captions must identify and link the paper or web source.
- Synthesized charts must cite the exact source values, preserve units and
  experimental conditions, and keep editable source or input data beside the
  rendered asset.
- Prefer numbered annotations (①, ②, ③) that are referenced in surrounding
  prose, without altering the original image.
- Place visuals before the text that explains them.
- Use descriptive alt text, expand abbreviations, label axes and units, and
  summarize the visual's takeaway in prose for readers who cannot render or
  inspect it.
- Do not add generic icons, stock or AI-generated decoration, repeated colored
  boxes, or a Mermaid diagram that merely restates the surrounding prose.
