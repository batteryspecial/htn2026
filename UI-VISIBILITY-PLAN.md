# Plan: make the agent trace and the other log sections actually visible

**For the implementing agent.** This is a frontend-only, CSS-and-two-small-components
task. Do not touch `orchestrator/`, `perception/`, `linker/`, any hook, any service,
or any contract. Do not run git commands.

Everything below is against the current tree. Line numbers are where the code sits
today; match on the selector or the code snippet, not the number.

---

## 0. The goal

The right-hand column has two stacked things: the INSTRUCTION panel on top and a
five-section accordion (Agent trace / Events / Behaviours / Spec / History)
underneath. Right now the accordion is crushed to a ~50px sliver at the very bottom
of the screen — one partial fold header, no content. The agent trace is effectively
invisible, and so is everything else in that accordion.

After this work: the instruction panel is capped, the accordion always owns the
bottom half of the column, and at least one section is open and readable on a fresh
load.

Keep the overall layout the operator asked for: **camera on the left half,
instruction + status + timer top right, logs in a collapsible accordion bottom
right.** Do not convert this to tabs. Do not move the logs into the camera column.

---

## 1. Diagnosis — why it is invisible

Three things stack up:

1. **`frontend/src/styles/layout.css:87`** — `.panel-chat { flex: 0 0 auto; }`.
   `0 0 auto` means *take your natural content height and never shrink*. The chat
   panel therefore claims whatever it wants and the accordion gets the remainder.

2. **The chat panel's natural height is enormous.** Inside it:
   - `.statusbar` ≈ 40px (`panels.css:440`)
   - `.transcript` — `min-height: 84px; max-height: 26vh` (`panels.css:487`), so up
     to ~310px on a 1200px-tall window
   - `.composer` — mic button + a `font-size: 24px` textarea at 2 rows (~100px,
     `panels.css:148`) + the attachments list + a chips row of four example buttons
     + the SEND / + IMAGE / STOP row + the `.hint` keyboard line

   Together that is roughly 750–820px of a ~1050px column.

3. **The accordion is left with nothing.** `.accordion` is `flex: 1 1 auto;
   min-height: 0; overflow: auto` (`layout.css:140`). With no space left it
   collapses to near zero. `.fold.open { flex: 1 1 0; min-height: 120px }`
   (`layout.css:160`) cannot be honoured inside a 50px box, so the accordion just
   scrolls its own headers out of view.

There is also a **fourth, separate hazard** that will bite after the layout is
fixed — see step 6.

---

## 2. Change 1 — give the accordion a guaranteed half (the core fix)

**File: `frontend/src/styles/layout.css`**

Replace the `.panel-chat` rule (currently line 87):

```css
/* Instruction sizes to its content, and gives the rest to the logs. */
.panel-chat { flex: 0 0 auto; }
```

with:

```css
/* The instruction sizes to its content but is capped, so the logs below it are
   never squeezed out. `0 1 auto` lets it shrink; the cap is in vh rather than %
   because a percentage max-height on a flex item depends on the parent having a
   definite height, and vh always resolves. */
.panel-chat {
  flex: 0 1 auto;
  min-height: 200px;
  max-height: 40vh;
}
```

Then, immediately below the existing `.accordion` rule (line 140), add a floor:

```css
/* The logs own the rest of the column and are never squeezed to a sliver. */
.accordion { flex: 1 1 auto; min-height: 240px; }
```

Do not delete the original `.accordion` rule; it carries `display: flex`,
`flex-direction: column`, `gap` and `overflow: auto`, all of which must stay. Either
add `min-height: 240px` inside the existing block or add the short block above after
it. One rule, not two conflicting ones, is cleaner — prefer editing the existing
block.

**Why `40vh`:** the top bar is ~104px plus 16px of grid padding, so the column is
about `100vh - 140px`. 40vh leaves the accordion roughly 55% of the column.

---

## 3. Change 2 — let the transcript absorb the squeeze

With the panel capped, something inside it has to give, or `.panel { overflow:
hidden }` (`layout.css:80`) will clip the SEND button off the bottom. The transcript
is the right thing to shrink; the composer must never be clipped.

**File: `frontend/src/styles/panels.css`**, the `.transcript` rule (line 487):

```css
.transcript {
  flex: 1 1 auto;
  min-height: 84px;
  max-height: 26vh;      /* ← remove */
  overflow: auto;
  ...
}
```

Change to:

```css
.transcript {
  flex: 1 1 auto;
  /* The panel above is capped now, so the transcript takes whatever is left
     inside it rather than setting its own ceiling. It must be allowed to reach
     zero, or the composer gets clipped instead. */
  min-height: 0;
  overflow: auto;
  ...
}
```

i.e. `min-height: 84px` → `min-height: 0`, and delete the `max-height: 26vh` line.
Leave everything else in that rule alone.

Verify `.composer { flex: 0 0 auto; }` (`panels.css:541`) is still present and
unchanged. That is what keeps SEND visible.

---

## 4. Change 3 — stop the five headers eating the accordion

Five closed sections currently cost about 270px of pure header (each fold is a
~40px header plus 3px borders top and bottom, plus a 10px gap between folds). That
is most of the accordion before a single row of content is drawn.

**File: `frontend/src/styles/layout.css`**

`.fold-toggle` (line 169): change

```css
  font-size: 14px;
  padding: 10px 16px;
```

to

```css
  font-size: 13px;
  padding: 7px 14px;
```

`.accordion` (line 140): change `gap: 10px` to `gap: 8px`.

`.fold.open` (line 160): change `min-height: 120px` to `min-height: 96px`, so three
sections can be open at once without the accordion needing to scroll.

`.fold-body` (line 197): change `padding: 12px 16px 14px` to `padding: 10px 14px 12px`.

These four together recover roughly 70–80px.

---

## 5. Change 4 — make sure something is actually open on load

**File: `frontend/src/components/common/Accordion.tsx`**

Line 14: `const STORE_KEY = 'retask.accordion.v1';`

The open/closed set is persisted in `localStorage`. Anyone who has already used the
console has a stored set from before this fix — very possibly everything collapsed,
or only a section that is still hard to see. They would apply all of the above and
still see nothing, and conclude the fix failed.

Bump the key so the new default applies exactly once:

```ts
const STORE_KEY = 'retask.accordion.v2';
```

**File: `frontend/src/App.tsx`**, line 216:

```tsx
initial={['trace']}
```

change to:

```tsx
initial={['trace', 'behaviors']}
```

Agent trace plus what is actually running is the pair you want side by side during a
demo. Two open sections at ~96px minimum each fits comfortably in the new budget.

---

## 6. Change 5 — the scroll hijack (do not skip this)

This is a real bug that the layout fix will *expose*, not one it causes.

**`frontend/src/components/trace/TracePanel.tsx:56-58`** and
**`frontend/src/components/chat/Transcript.tsx:8-10`** both pin to the newest entry
with:

```tsx
tail.current?.scrollIntoView({ block: 'end' });
```

`scrollIntoView` scrolls **every** scrollable ancestor, not just the nearest one.
`.accordion` has `overflow: auto`. So each time a trace entry arrives, the browser
may also scroll the accordion itself to bring the trace tail into view — dragging
the Events and Behaviours sections out of sight. During a live turn, entries arrive
every few hundred milliseconds, so the accordion will fight the operator constantly.

Fix both files by scrolling the container directly instead.

**`TracePanel.tsx`** — put the ref on the scroll box rather than a tail div:

```tsx
export function TracePanel({ entries, live }: TraceProps) {
  const box = useRef<HTMLDivElement>(null);

  // Pin to the newest entry. Set scrollTop on the box itself rather than calling
  // scrollIntoView on a tail element: scrollIntoView also scrolls every scrollable
  // ancestor, and the accordion around this panel is one of them.
  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries.length]);

  return (
    <div className="panel-scroll" ref={box}>
      ...
    </div>
  );
}
```

Delete the `<div ref={tail} />` at the end of the returned JSX (currently line 84)
and the now-unused `tail` ref.

**`Transcript.tsx`** — identical treatment. Move the ref onto the
`<div className="transcript">` element, set `scrollTop = scrollHeight` in the
effect, and delete the `<div ref={tail} />` on line 36.

Note `Transcript` has two return paths (an empty state and the populated list). Only
the populated one needs the ref; leave the empty branch alone.

---

## 7. Optional, only if there is time after the above is verified

Do these one at a time and re-check, not as a batch.

**7a. Reclaim the camera's wasted width.** The video is letterboxed inside its
panel — there are wide black bars either side. `layout.css:67-68`:
`.col-camera { flex: 1 1 50%; }` → `flex: 1 1 44%;` and
`.col-side { flex: 1 1 50%; }` → `flex: 1 1 56%;`. This is the single cheapest way
to buy the logs more room, and the camera loses nothing visible.

**7b. Shrink the composer.** `panels.css:148` `.instruction-input { font-size:
24px }` → `18px` saves about 35px. The `.hint` line (the ENTER / SHIFT+ENTER
legend at the bottom of `ChatPanel.tsx`) could drop to `font-size: 11px`.

**7c. Solo / maximise a section.** In `Accordion.tsx`, add a small `⤢` button to
each `.fold-head` next to the count that sets `open` to just that one id. During a
demo you often want the trace alone at full height. Keep the existing toggle
behaviour intact; this is an additional button, and it must call
`writeJson(STORE_KEY, next)` like `toggle` does so the choice persists.

**7d. Three columns on a very wide monitor.** The operator runs this at 2560px.
Behind `@media (min-width: 1800px)`, make `.grid` three columns — camera 42%,
instruction 29%, logs 29% — by moving the accordion out of `.col-side` into its own
`.col`. **This changes `App.tsx` structure, contradicts the layout the operator
specified, and should not be done without asking them first.** Listed only so it is
not rediscovered later.

---

## 8. Acceptance checks

Run the dev server (`cd frontend && npm run dev`, port 3000) and confirm each of
these by eye at a normal 1920×1080 or larger window:

1. The accordion occupies the bottom ~55% of the right column. All five section
   headers — AGENT TRACE, EVENTS, BEHAVIOURS, SPEC, HISTORY — are visible without
   scrolling.
2. AGENT TRACE and BEHAVIOURS are open on first load in a fresh profile (or after
   `localStorage.removeItem('retask.accordion.v1')` and a reload).
3. The open AGENT TRACE section shows at least ~5 trace rows, and its own scrollbar
   appears when there are more.
4. The SEND, + IMAGE and STOP buttons are all fully visible and clickable. Nothing
   in the instruction panel is clipped.
5. Type a long multi-line instruction and send several turns. The transcript scrolls
   internally; the instruction panel does **not** grow past its cap and the
   accordion does **not** shrink.
6. With the orchestrator running and a live turn in progress, new trace entries
   appear and the trace pins to the newest — **and the accordion's own scroll
   position does not jump.** Scroll the accordion so EVENTS is visible, then send a
   turn; EVENTS must stay where you put it.
7. Open all five sections at once. The accordion scrolls rather than crushing them;
   each open section still shows content, not just its header.
8. Narrow the window below 1100px. The stacked mobile layout from the existing
   `@media (max-width: 1100px)` block (`layout.css:127-136`) still works — the page
   scrolls and the camera keeps its 46vh.

Then:

```
cd frontend
npx tsc -p tsconfig.app.json --noEmit
npx vite build
```

Both must be clean. The build was clean before this work, so any error is yours.

---

## 9. Do not touch

- `orchestrator/**`, `perception/**`, `linker/**` — different services entirely.
- Any file under `frontend/src/hooks/` or `frontend/src/services/` — this is a
  presentation problem, and nothing about the data flow needs to change.
- `frontend/src/contracts/**`.
- The `@media (max-width: 1100px)` block in `layout.css`, except to confirm it still
  behaves.
- Git. No commits, no branches, no stashes.

## 10. Summary of files touched

| File | Change |
|---|---|
| `frontend/src/styles/layout.css` | cap `.panel-chat`; floor `.accordion`; shrink `.fold-toggle`, `.fold-body`, gap, `.fold.open` min-height |
| `frontend/src/styles/panels.css` | `.transcript` — drop `max-height`, `min-height` → 0 |
| `frontend/src/components/common/Accordion.tsx` | `STORE_KEY` → `v2` |
| `frontend/src/App.tsx` | `initial={['trace', 'behaviors']}` |
| `frontend/src/components/trace/TracePanel.tsx` | `scrollIntoView` → `scrollTop` on the box |
| `frontend/src/components/chat/Transcript.tsx` | `scrollIntoView` → `scrollTop` on the box |

Six files. Nothing else.
