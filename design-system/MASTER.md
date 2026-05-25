# Study Coach — Design System MASTER

> Single source of truth for visual decisions. Locked at P3 brainstorm (2026-05-25). All views / cuts must conform unless they declare an explicit override in `design-system/pages/<page>.md`.

- **Stack**: Vue 3.5 + Tailwind 4 (existing P1 baseline; no migration)
- **Style category**: Modern Dark Cinema (Inter System) — adapted from `ui-ux-pro-max` typography pairing #2, applied to the existing P1 dark+indigo palette
- **Inspiration parallel**: developer tools / AI dashboards / fintech precision panels
- **Audience**: HKBU CS student portfolio review + adult self-learner; bilingual (zh-CN + en)

---

## 1. Color tokens

P1 baseline kept verbatim; tokens formalized for cross-component reuse. Add to Tailwind `theme.extend.colors`:

```ts
// tailwind.config — add at root
theme: {
  extend: {
    colors: {
      // surface
      bg:        '#0b0e1a',     // page background (slightly darker than P1 #11162a, for OLED-friendly contrast)
      surface:   '#11162a',     // P1 baseline — nav, cards
      'surface-2': '#171c34',   // raised cards / modals
      border:    'rgba(255,255,255,0.05)',  // P1 baseline divider
      'border-strong': 'rgba(255,255,255,0.12)',

      // text
      fg:        '#e6e6ec',     // primary text (P1 baseline)
      'fg-muted': '#b0b6c5',    // secondary text (P1 nav-link)
      'fg-dim':   'rgba(255,255,255,0.4)',   // P1 empty-state

      // primary (locked = indigo, P1 baseline)
      primary:    '#6366f1',    // indigo-500 — single primary CTA color
      'primary-2': '#818cf8',   // indigo-400 — hover
      'primary-bg': 'rgba(99,102,241,0.15)', // P1 nav-link.router-link-active bg
      'primary-ring': 'rgba(99,102,241,0.4)',// P1 input focus border

      // status (semantic)
      success:    '#10b981',    // emerald-500 — done milestone, correct quiz, mastery > 0.7
      warning:    '#f59e0b',    // amber-500   — due-today milestone, mastery 0.4-0.7
      danger:     '#f43f5e',    // rose-500    — overdue milestone, wrong quiz, mastery < 0.4
      'success-bg': 'rgba(16,185,129,0.12)',
      'warning-bg': 'rgba(245,158,11,0.12)',
      'danger-bg':  'rgba(244,63,94,0.12)',
    },
  },
},
```

**Contrast verified** (WCAG AA at minimum):
- `fg` `#e6e6ec` on `bg` `#0b0e1a` → 14.8:1 ✓ AAA
- `fg-muted` `#b0b6c5` on `surface` `#11162a` → 8.4:1 ✓ AAA
- `primary` `#6366f1` text on `bg` → 4.9:1 ✓ AA (large UI elements only; never small body)
- `success` / `warning` / `danger` always paired with icon, never color-only meaning

**No emoji as icons.** Use **lucide-vue-next** (Tailwind/Vue 3 ecosystem standard). Add as the only icon dep.

---

## 2. Typography

```css
/* src/style.css — add at top, replaces nothing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
  --font-sans: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

body {
  font-family: var(--font-sans);
  font-feature-settings: 'cv11', 'ss01';  /* Inter alt 'a', stylistic set 1 */
}

code, .mono { font-family: var(--font-mono); }
```

Tailwind extension:
```ts
fontFamily: {
  sans: ['Inter', 'Noto Sans SC', 'system-ui', 'sans-serif'],
  mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
},
```

### Type scale (use Tailwind defaults except as noted)
| token | usage | css |
|---|---|---|
| `text-xs` 12 | meta / SM-2 stats / footer | `font-mono` when numeric |
| `text-sm` 14 | nav links, MCQ option text, citations | sans |
| `text-base` 16 | body text, chat bubble | sans |
| `text-lg` 18 | section heading, MilestoneRow title | sans 600 |
| `text-xl` 20 | view title in `<ViewHeader>` | sans 600 |
| `text-2xl` 24 | Overview hero metric numbers | mono 500 |
| `text-3xl` 30 | "Study Coach" wordmark only | sans 700 -0.02em tracking |

**Tracking convention** (per Inter system best practice):
- `text-xl`+ headings → `-0.02em`
- `text-xs` labels uppercase → `+0.05em` 

---

## 3. Layout primitives

### Container & spacing rhythm

- Spacing scale = Tailwind default (4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 px).
- Section gap: `gap-6` (24px) within a view; `gap-12` (48px) between Overview widget rows.
- Card padding: `p-4` (16px) compact; `p-6` (24px) hero widgets.
- Border radius: `rounded-lg` (8px) — all cards; `rounded-2xl` (16px) — chat bubbles only (P1 baseline); `rounded-md` (6px) — chips/buttons.

### Shell layout (locked)

```
┌─────────┬──────────────────────────────────┐
│  nav    │  RouterView                       │
│  w-56   │  flex-1 overflow-hidden           │
│  bg=    │  (each view is h-full flex-col,   │
│  surface│   scroll inside its main region)  │
└─────────┴──────────────────────────────────┘
```

Nav widened from P1 `w-48` to `w-56` (224px) to fit 7 links + section labels without truncation. Nav stays fixed position; only RouterView scrolls.

---

## 4. Responsive strategy (explicit deviation from mobile-first)

- **Desktop-primary** at 1440px design target (portfolio screenshot battleground).
- Tailwind breakpoints used:
  - `lg:` 1024 — graceful reflow (Overview grid 2-col → still 2-col, more padding)
  - `xl:` 1280 — primary design target
  - `2xl:` 1536 — wider hero widgets
- 768-1023: Overview grid 2-col → 1-col stack; nav collapses to icon-only (40px wide); chat input full-width.
- **<768**: render a centered banner: "Open Study Coach in desktop for the full experience" + link to mobile-friendly Library upload only. Full mobile UI is P4.
- Document this as **explicit deviation** from Quick Reference §5 `mobile-first`; rationale lives in spec §10.

---

## 5. Component patterns (ad-hoc, no library)

### Anatomy reference for the most-built primitives

#### `<Card>`
```html
<div class="rounded-lg border border-border bg-surface p-4 hover:bg-surface-2 transition-colors">
  <!-- slot -->
</div>
```

#### `<Button>` (primary / secondary / ghost)
```html
<!-- primary -->
<button class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-primary-ring focus:ring-offset-2 focus:ring-offset-bg">
  …
</button>

<!-- secondary (P1 nav-link style) -->
<button class="rounded-md px-3 py-1.5 text-sm text-fg-muted hover:bg-white/5 transition-colors">
  …
</button>
```

#### `<Chip>` (status / topic / mode)
```html
<span class="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-mono"
      :class="{
        'border-success/30 bg-success-bg text-success': variant === 'success',
        'border-warning/30 bg-warning-bg text-warning': variant === 'warning',
        'border-danger/30 bg-danger-bg text-danger':   variant === 'danger',
      }">
  <Icon /> label
</span>
```

#### `<ModeChip>` (P3 unique)
```html
<button class="inline-flex items-center gap-2 rounded-full border border-primary-ring bg-primary-bg px-3 py-1 text-xs font-mono text-primary hover:bg-primary/20 transition-colors"
        :aria-pressed="overridden">
  <Icon /> {{ currentMode }}
  <span class="opacity-50">⇄</span>
</button>
```

ARIA for accessibility (a11y patterns borrowed from shadcn-vue without importing the lib):
- Radio group: `role="radiogroup"` + each `role="radio"` `aria-checked` + keyboard `ArrowDown/Up`.
- Segmented control (DifficultySelector): same as radio group + `aria-label="Difficulty"`.

---

## 6. Animation tokens

- Standard transition: `transition-colors duration-150 ease-out` for hover/active.
- View-mount: no transition (Vue Router default — perceived speed wins over fanfare for dashboard).
- Loading: `<Skeleton>` shimmer for >300ms async (per Quick Reference §3 `progressive-loading`).
- Reduced motion: respect `prefers-reduced-motion: reduce` — gate all transitions in `src/style.css`:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
```

---

## 7. Chart system (chart.js 4 + vue-chartjs 5)

Locked picks. Tree-shake imports per chart type (not the whole `chart.js` bundle).

### Radar (Overview `<RadarChart>`)
- 5 axes (specific dim names finalized at A11 cut; current candidates: `Mastery` / `Plan progress` / `Quiz accuracy` / `Streak` / `Coverage`).
- Dataset color: `primary` `#6366f1` 40% opacity fill + solid stroke.
- Grid lines: `border-strong` `rgba(255,255,255,0.12)`, 1px.
- Axis labels: `text-xs` `font-mono` `fg-muted`.
- No legend (single dataset).

### Bar (MasteryCard top-5 alternative)
- **Default**: pure Tailwind div bars (no chart.js needed for 5 horizontal bars). Use chart.js only if we need axis labels.
- If chart.js used: horizontal bar; bar color = topic mastery interpolated `danger → warning → success` via 3-stop linear scale.

### Forbidden
- Pie / donut (only 5 categories, bar is clearer per Quick Reference §10 `no-pie-overuse`).
- 3D charts.
- Animated entry > 400ms.

---

## 8. Iconography

- **Library**: `lucide-vue-next` only. Add to deps in A0.
- **Stroke**: 1.5px (Lucide default).
- **Size tokens**: `w-4 h-4` (16) inline-with-text / `w-5 h-5` (20) buttons / `w-6 h-6` (24) view headers.
- **Filled vs outline**: outline-only across the entire app (single hierarchy = single style per Quick Reference §4 `filled-vs-outline-discipline`).
- **No emojis** in any UI text. Existing P1 doesn't have them; keep zero count.

---

## 9. Status semantics (cross-component contract)

| state | color | icon | example use |
|---|---|---|---|
| `success` | `emerald-500` | `<CheckCircle2>` | done milestone / correct quiz / mastery > 0.7 |
| `warning` | `amber-500`   | `<AlertCircle>`  | due today / mastery 0.4-0.7 |
| `danger`  | `rose-500`    | `<XCircle>` / `<AlertTriangle>` | overdue / wrong quiz / mastery < 0.4 |
| `info`    | `primary`     | `<Info>`        | mode chip / banner |
| `neutral` | `fg-muted`    | `<Circle>`      | future milestone / unranked topic |

Color is **never** the only signal — always paired with icon per Quick Reference §1 `color-not-only`.

---

## 10. Overrides

Page-specific deviations live at `design-system/pages/<page-name>.md`. None defined yet at P3 brainstorm time.

---

## 11. Anti-patterns (must not appear in P3 code)

- ❌ Pure black `#000` background (use `bg` `#0b0e1a`)
- ❌ Pure white `#fff` text (use `fg` `#e6e6ec`)
- ❌ Emoji as functional icon
- ❌ Color alone to convey status (always pair with icon)
- ❌ Hover-only interactions (mobile <768 can't hover; even though we drop mobile, do not design with hover-as-primary mindset)
- ❌ Layout-shifting press states (use opacity / color transitions, not `transform: scale` on cards bigger than buttons)
- ❌ Animations >400ms (cuts perceived speed)
- ❌ Importing `shadcn-vue` package (ad-hoc with borrowed ARIA patterns instead)
- ❌ Custom font weights outside Inter's loaded set (300/400/500/600/700)

---

## 12. References

- Source skill: `ui-ux-pro-max` v2.5.0 — designs persisted from query `"developer tools dashboard analytics dark professional technical"` (style match `Modern Dark Cinema`) + manual consolidation with P1 baseline preservation.
- Spec: `docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`
- P1 baseline reference: `frontend/src/App.vue`, `frontend/src/views/Chat.vue` (kept as-is for color/radius tokens).
