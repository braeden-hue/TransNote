---
name: project-color-system
description: online_webpage UI color system — Samsung Life Insurance CI blue palette, applied 2026-05
metadata:
  type: project
---

The web demo (`online_webpage/`) was redesigned in May 2026 to match Samsung Life Insurance (삼성생명) CI colors. All files now use a white-background, Samsung blue palette.

**Why:** The project is presented to Samsung Life Insurance as the sponsoring organization, so the UI must reflect their brand identity and convey financial-sector trustworthiness.

**How to apply:** Any future UI work on `online_webpage/` must stay within this color system. Do not reintroduce dark backgrounds, orange, or purple accent colors.

## Design Tokens (CSS variables defined in style.css and landing.css)

| Token | Value | Role |
|---|---|---|
| `--primary` | `#0076CE` | Samsung Blue — buttons, borders, active states, headings |
| `--primary-dk` | `#0060AA` | Hover / pressed state for primary |
| `--sky` | `#5BB8F5` | Light blue — secondary accents, sky blue highlights |
| `--sky-dk` | `#3A9EE0` | Darker sky blue for 3rd beat |
| `--green` | `#2FAA6E` | Success / "done" states only (loading step complete) |
| `--purple` | `#5B7FD6` | Tertiary accent (feature card gradients only) |
| `--bg` | `#FFFFFF` | Page background |
| `--bg-panel` | `#F5F7FA` | Panel / section alt background |
| `--bg-card` | `#FFFFFF` | Card background |
| `--bg-hover` | `#EBF4FC` | Hover state tint |
| `--border` | `#E8EFF6` | Default border |
| `--border-hi` | `#C5D8EC` | Emphasized border |
| `--text` | `#1A1A2E` | Primary text (dark navy) |
| `--text-dim` | `#4A6080` | Secondary text |
| `--text-mute` | `#8BA5BE` | Muted / hint text |

## Beat Colors (BEAT_COLORS in samples.js, used across all JS renderers)

| Beat | Color | Tone |
|---|---|---|
| 1박 | `#0076CE` | Samsung Primary Blue |
| 2박 | `#5BB8F5` | Sky Blue |
| 3박 | `#A8D5F5` | Light Sky Blue |
| 4박 | `#D0EBFA` | Very Light Blue |

## Piano Key Highlight Colors (piano.js)

- **Expected note (next to press):** white key `#d0ebfa`, black key `#003a6e`, glow `#0076CE`
- **Pressed note:** white key `#b8ddf8`, black key `#004d8a`
- **Correct flash:** white key `#b8ddf8`, black key `#004d8a`, glow `#0076CE`
- **Wrong note blink:** CSS animation class `key-wrong-blink` — red `#ff3333` (unchanged, semantic error color)
- **Physical keys (idle):** white `#f4efe6`, black `#1c1c1c` — kept as realistic piano appearance

## Files Modified

- `online_webpage/style.css` — full CSS variable replacement, light theme
- `online_webpage/landing.css` — full CSS variable replacement, light theme
- `online_webpage/index.html` — beat-cell `--c` inline values updated
- `online_webpage/landing.html` — demo cells, beat legend, em colors, BEAT_COLORS script block
- `online_webpage/js/samples.js` — `BEAT_COLORS` export updated
- `online_webpage/js/piano.js` — expected/pressed/correct highlight colors updated
- `online_webpage/js/notation.js` — zone backgrounds, measure dot, note fill, staff label colors
- `online_webpage/js/app.js` — REF_ARROWS, NOTE_COLORS, auto-play flash, staff label colors
