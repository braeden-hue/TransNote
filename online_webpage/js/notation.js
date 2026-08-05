import { BEAT_COLORS, pitchToZone, formatNoteName, zoneLabels, effectiveClef, hasMixedClef } from './samples.js';

const UNIT_W      = 80;
const CELL_H      = 46;
const ZONE_H      = CELL_H + 10;  // 56
const MARGIN_L    = 68;
const MARGIN_Y    = 8;
const INDICATOR_H = 18;  // 화살표 표시 영역 (상단)
const NS          = 'http://www.w3.org/2000/svg';

// REF_ZONE: 마디 시작 기준점 점(●)을 표시할 존 인덱스
// treble: 가온다(4옥) = 삼등분 중 최하단(2) / bass: 가온다 기준점 = 삼등분 중 중간(1)
const REF_ZONE = { treble: 2, bass: 1 };

// 마디 중간 클렙 전환 배경 틴트 — 같은 note 리스트 안에 서로 다른 active clef가 실제로
// 섞여 있을 때만(hasMixedClef) 그린다. ~12% opacity. docs/music-notation-rule-designer.md 참고.
const CLEF_TINT = { treble: 'rgba(108,99,255,0.12)', bass: 'rgba(255,179,71,0.12)' };

function el(tag, attrs = {}) {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  return e;
}

export function renderNotation(container, notes, {
  highlightIdx = -1,
  expectedIdx  = -1,
  onNoteClick,
  clef = 'treble',
} = {}) {
  if (!notes?.length) {
    container.innerHTML = '<p style="color:#555;padding:20px">음표 데이터가 없습니다</p>';
    return null;
  }

  const totalDur = notes.reduce((s, n) => s + n.duration, 0);
  const svgW     = Math.max(totalDur * UNIT_W + MARGIN_L + 40, (container.clientWidth || 600));
  const CONTENT_Y = INDICATOR_H + MARGIN_Y;  // zones 시작 y
  const svgH      = INDICATOR_H + ZONE_H * 3 + MARGIN_Y * 2;

  const svg = el('svg', {
    width: svgW, height: svgH,
    viewBox: `0 0 ${svgW} ${svgH}`,
    style: 'display:block;',
  });

  // ── Zone backgrounds + labels ─────────────────────────────────────────────
  const ZONE_LABELS = zoneLabels(clef);
  for (let z = 0; z < 3; z++) {
    const zy = CONTENT_Y + z * ZONE_H;
    svg.appendChild(el('rect', {
      x: MARGIN_L, y: zy,
      width: svgW - MARGIN_L - 8, height: ZONE_H,
      fill: z % 2 === 0 ? '#F0F6FC' : '#F8FBFF',
    }));
    if (z > 0) {
      svg.appendChild(el('line', {
        x1: MARGIN_L, y1: zy, x2: svgW - 8, y2: zy,
        stroke: '#C5D8EC', 'stroke-width': '1.5', 'stroke-dasharray': '6,4',
      }));
    }
    const lbl = el('text', {
      x: MARGIN_L - 6, y: zy + ZONE_H / 2 + 4,
      'text-anchor': 'end', fill: '#8BA5BE',
      'font-size': '10', 'font-family': 'system-ui',
    });
    lbl.textContent = ZONE_LABELS[z];
    svg.appendChild(lbl);
  }

  // ── Collect measure-start x positions (beat === 1) ─────────────────────────
  const measureXs = [];
  {
    let sx = MARGIN_L + 4;
    notes.forEach(note => {
      if (note.beat === 1) measureXs.push(sx);
      sx += note.duration * UNIT_W;
    });
  }

  // ── Draw notes ────────────────────────────────────────────────────────────
  // 마디 중간 클렙 전환 배경 틴트는 이 note 리스트 안에 실제로 서로 다른 active clef가
  // 섞여 있을 때만 켠다 — 단일 클렙 리스트의 기존 렌더링은 절대 바뀌지 않아야 함.
  const mixedClef = hasMixedClef(notes, clef);
  let x = MARGIN_L + 4;
  const noteXMap = [];

  notes.forEach((note, i) => {
    const w       = note.duration * UNIT_W - 4;
    const effClef = effectiveClef(note, clef);
    const zone    = note.isRest ? 1 : pitchToZone(note.pitch, effClef);
    const y       = CONTENT_Y + zone * ZONE_H + 5;
    const h       = CELL_H;
    const color   = BEAT_COLORS[note.beat] || '#888';
    const isHL    = i === highlightIdx;
    const isExp   = i === expectedIdx;
    const isChord = note.chordNotes?.length > 0;

    noteXMap.push(x);

    // 배경 틴트 — 텍스트/테두리/하이라이트보다 먼저(아래 레이어) 그린다. 클릭 판정에
    // 관여하지 않도록 pointer-events: none (실제 클릭 타겟은 항상 이 뒤에 그려짐).
    if (mixedClef) {
      svg.appendChild(el('rect', {
        x, y, width: w, height: h, rx: 4,
        fill: CLEF_TINT[effClef] ?? CLEF_TINT.treble,
        'pointer-events': 'none',
      }));
    }

    if (note.isRest) {
      // ── 쉼표: 존 중앙(가운데 존)에 점선 + '쉼표' 라벨만 표시, 박스/음이름 없음 ──
      if (onNoteClick) {
        const hitBox = el('rect', {
          x, y, width: w, height: h, rx: 5,
          fill: 'transparent', stroke: 'none',
        });
        hitBox.style.cursor = 'pointer';
        hitBox.addEventListener('click', () => onNoteClick(i, note));
        svg.appendChild(hitBox);
      }

      svg.appendChild(el('line', {
        x1: x, y1: y + h, x2: x + w, y2: y + h,
        stroke: isHL ? '#fff' : isExp ? '#0076CE' : color,
        'stroke-width': isHL || isExp ? '3' : '2.5',
        'stroke-dasharray': '5,4',
        'stroke-linecap': 'round',
      }));

      const rfs = el('text', {
        x: x + w / 2, y: y + h / 2 + 4,
        'text-anchor': 'middle',
        fill: isHL ? '#fff' : isExp ? '#0076CE' : color,
        'font-size': w < 30 ? 9 : 11, 'font-weight': '700', 'font-family': 'system-ui',
        'pointer-events': 'none',
      });
      rfs.textContent = '쉼표';
      svg.appendChild(rfs);
    } else if (isChord) {
      // ── 화음: 박스 + 음표 이름 세로 스택 ─────────────────────────────────
      // mixedClef일 때는 기본 상태 fill을 반투명으로 낮춰 아래 클렙 틴트가 비쳐 보이게 한다
      // (mixedClef가 아니면 기존과 동일하게 불투명 흰색 — 단일 클렙 렌더링은 그대로 유지).
      const fill        = isHL ? color + '44' : isExp ? '#0076CE18' : (mixedClef ? 'rgba(255,255,255,0.55)' : '#FFFFFF');
      const strokeColor = isExp ? '#0076CE' : color;
      const strokeW     = isHL ? '3' : isExp ? '2.5' : '2';

      const rect = el('rect', {
        x, y, width: w, height: h, rx: 5,
        fill, stroke: strokeColor, 'stroke-width': strokeW,
      });
      if (isExp) {
        const anim = document.createElementNS(NS, 'animate');
        anim.setAttribute('attributeName', 'stroke-opacity');
        anim.setAttribute('values', '1;0.25;1');
        anim.setAttribute('dur', '0.85s');
        anim.setAttribute('repeatCount', 'indefinite');
        rect.appendChild(anim);
      }
      if (onNoteClick) {
        rect.style.cursor = 'pointer';
        rect.addEventListener('click', () => onNoteClick(i, note));
      }
      svg.appendChild(rect);

      if (isHL) {
        const ring = el('rect', {
          x: x-3, y: y-3, width: w+6, height: h+6, rx: 8,
          fill: 'none', stroke: color, 'stroke-width': '1.5',
        });
        const a = document.createElementNS(NS, 'animate');
        a.setAttribute('attributeName', 'opacity');
        a.setAttribute('values', '0.6;0;0.6');
        a.setAttribute('dur', '0.8s');
        a.setAttribute('repeatCount', 'indefinite');
        ring.appendChild(a);
        svg.appendChild(ring);
      }

      // 모든 화음 음표 이름 세로 스택
      const allNotes  = [note.pitch, ...(note.chordNotes || [])];
      const lineH     = allNotes.length <= 2 ? 14 : 12;
      const fs        = allNotes.length <= 2 ? 11 : 9;
      const totalTxtH = allNotes.length * lineH;
      allNotes.forEach((p, pi) => {
        const t = el('text', {
          x: x + w / 2,
          y: y + h / 2 - totalTxtH / 2 + lineH * (pi + 0.8),
          'text-anchor': 'middle',
          fill: isHL ? '#fff' : isExp ? '#0076CE' : color,
          'font-size': fs, 'font-weight': '700', 'font-family': 'system-ui',
          'pointer-events': 'none',
        });
        t.textContent = formatNoteName(p);
        svg.appendChild(t);
      });
    } else {
      // ── 단음: 사각형 박스 없이 텍스트 + 박자색 밑줄로만 셀 경계 표시 ─────────
      if (onNoteClick) {
        // 투명 hit-box for tap detection
        const hitBox = el('rect', {
          x, y, width: w, height: h, rx: 5,
          fill: 'transparent', stroke: 'none',
        });
        hitBox.style.cursor = 'pointer';
        hitBox.addEventListener('click', () => onNoteClick(i, note));
        svg.appendChild(hitBox);
      }

      svg.appendChild(el('line', {
        x1: x, y1: y + h, x2: x + w, y2: y + h,
        stroke: isHL ? '#fff' : isExp ? '#0076CE' : color,
        'stroke-width': isHL || isExp ? '3' : '2.5',
        'stroke-linecap': 'round',
      }));

      const fs  = w < 26 ? 10 : w < 42 ? 12 : 14;
      const txt = el('text', {
        x: x + w / 2, y: y + h / 2 + 5,
        'text-anchor': 'middle',
        fill: isHL ? '#fff' : isExp ? '#0076CE' : color,
        'font-size': fs, 'font-weight': '700', 'font-family': 'system-ui',
        'pointer-events': 'none',
      });
      txt.textContent = formatNoteName(note.pitch);
      svg.appendChild(txt);
    }

    // ── expected 화살표 (단음/화음 공통) ──────────────────────────────────
    if (isExp) {
      const arrowCx  = x + w / 2;
      const arrowTip = y - 2;
      const arrowTop = Math.max(2, arrowTip - 13);
      svg.appendChild(el('rect', {
        x: arrowCx - 6, y: arrowTop - 1, width: 12, height: arrowTip - arrowTop + 2,
        fill: 'rgba(255,255,255,0.85)', rx: '2',
      }));
      const tri = el('polygon', {
        points: `${arrowCx - 5},${arrowTop} ${arrowCx + 5},${arrowTop} ${arrowCx},${arrowTip}`,
        fill: '#0076CE',
      });
      const anim2 = document.createElementNS(NS, 'animateTransform');
      anim2.setAttribute('attributeName', 'transform');
      anim2.setAttribute('type', 'translate');
      anim2.setAttribute('values', '0 0;0 3;0 0');
      anim2.setAttribute('dur', '0.65s');
      anim2.setAttribute('repeatCount', 'indefinite');
      tri.appendChild(anim2);
      svg.appendChild(tri);
    }

    x += note.duration * UNIT_W;
  });

  // ── 마디 시작 기준점 표시 — clef에 따라 기준 존이 다름 ──────────────────────
  const refCY = CONTENT_Y + REF_ZONE[clef] * ZONE_H + ZONE_H / 2;

  measureXs.forEach(mx => {
    const rc = el('circle', { cx: mx, cy: refCY, r: '5', fill: '#0076CE' });
    const ra = document.createElementNS(NS, 'animate');
    ra.setAttribute('attributeName', 'fill-opacity');
    ra.setAttribute('values', '1;0.5;1');
    ra.setAttribute('dur', '2s');
    ra.setAttribute('repeatCount', 'indefinite');
    rc.appendChild(ra);
    svg.appendChild(rc);
  });

  container.innerHTML = '';
  container.appendChild(svg);

  return {
    scrollToNote(idx) {
      if (idx < 0 || idx >= noteXMap.length) return;
      const cx = noteXMap[idx] + notes[idx].duration * UNIT_W / 2;
      container.scrollLeft = Math.max(0, cx - container.clientWidth / 2);
    },
    // 마디 시작에 맞춰 스크롤 — 마디가 잘리지 않도록
    scrollToMeasureOf(idx) {
      if (idx < 0 || idx >= noteXMap.length) return;
      // 이 음표가 속한 마디의 첫 음 인덱스 탐색
      let mStart = idx;
      while (mStart > 0 && notes[mStart].beat !== 1) mStart--;
      // 마디 시작 x를 왼쪽 끝에 맞춤 (존 레이블이 가리지 않도록 MARGIN_L 고려)
      container.scrollLeft = Math.max(0, noteXMap[mStart] - MARGIN_L - 4);
    },
  };
}

// 오선 번호(0부터) → 표시 레이블 + 색상
function staffLabel(si, clef) {
  if (si === 0) return { label: '🎵 높은음자리 (Treble)', color: '#0076CE' };
  if (si === 1 && clef === 'bass') return { label: '🎻 낮은음자리 (Bass)', color: '#5BB8F5' };
  const clefName = clef === 'bass' ? '낮은음자리' : clef === 'alto' ? '알토' : '높은음자리';
  return { label: `Staff ${si + 1} (${clefName})`, color: '#7BB8A0' };
}

// ── 다중 오선 렌더러 (2개 이상 N개 지원) ─────────────────────────────────────
export function renderGrandStaff(container, staves, options = {}) {
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'display:flex; flex-direction:column; gap:10px;';

  staves.forEach((stave, si) => {
    const clef = stave.clef ?? (si === 0 ? 'treble' : 'bass');
    const meta = staffLabel(si, clef);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex; flex-direction:column; gap:3px;';

    const label = document.createElement('div');
    label.textContent = meta.label;
    label.style.cssText = `
      font-size:11px; font-weight:700; color:${meta.color};
      padding-left:${MARGIN_L}px; font-family:system-ui; letter-spacing:.04em;
    `;
    row.appendChild(label);

    const navWrap = document.createElement('div');
    navWrap.className = 'notation-nav-wrap';
    const btnP = document.createElement('button');
    btnP.className = 'notation-nav-btn'; btnP.innerHTML = '&#8249;'; btnP.disabled = true;
    const btnN = document.createElement('button');
    btnN.className = 'notation-nav-btn'; btnN.innerHTML = '&#8250;';
    const inner = document.createElement('div');
    inner.className = 'notation-container scrollable';
    navWrap.appendChild(btnP); navWrap.appendChild(inner); navWrap.appendChild(btnN);
    row.appendChild(navWrap);
    wrapper.appendChild(row);

    function updateGsNav() {
      btnP.disabled = inner.scrollLeft <= 1;
      btnN.disabled = inner.scrollLeft + inner.clientWidth >= inner.scrollWidth - 1;
    }
    btnP.addEventListener('click', () => { inner.scrollLeft -= inner.clientWidth * 0.8; setTimeout(updateGsNav, 350); });
    btnN.addEventListener('click', () => { inner.scrollLeft += inner.clientWidth * 0.8; setTimeout(updateGsNav, 350); });
    inner.addEventListener('scroll', updateGsNav);

    renderNotation(inner, stave.notes, { ...options, clef });
    setTimeout(updateGsNav, 50);
  });

  container.appendChild(wrapper);
}

// ── 튜토리얼용 미니 렌더러 ────────────────────────────────────────────────────
export function renderMiniNotation(container, notes, { unitW = 64, cellH = 38, onNoteClick } = {}) {
  const NOTE_TOP = 16;  // space above note cells for measure markers
  const svgW = notes.reduce((s, n) => s + n.duration * unitW, 0) + 8;
  const svgH = NOTE_TOP + cellH + 8;
  const svg  = el('svg', { width: svgW, height: svgH, viewBox: `0 0 ${svgW} ${svgH}` });

  // Measure start markers
  let mx = 4;
  notes.forEach(note => {
    if (note.beat === 1) {
      const cy = NOTE_TOP + cellH / 2;
      svg.appendChild(el('line', {
        x1: mx, y1: NOTE_TOP + 2,
        x2: mx, y2: NOTE_TOP + cellH - 2,
        stroke: '#0076CE', 'stroke-width': '1.5', 'stroke-opacity': '0.5',
      }));
      svg.appendChild(el('circle', {
        cx: mx, cy, r: '4',
        fill: '#0076CE', 'fill-opacity': '0.9',
      }));
    }
    mx += note.duration * unitW;
  });

  // Note cells — 화음만 박스, 단음은 텍스트만
  let x = 4;
  notes.forEach((note, i) => {
    const w        = note.duration * unitW - 4;
    const color    = BEAT_COLORS[note.beat] || '#888';
    const isChord  = note.chordNotes?.length > 0;

    if (isChord) {
      const rect = el('rect', {
        x, y: NOTE_TOP, width: w, height: cellH, rx: 4,
        fill: '#FFFFFF', stroke: color, 'stroke-width': '2',
      });
      if (onNoteClick) {
        rect.style.cursor = 'pointer';
        rect.addEventListener('click', () => onNoteClick(i, note));
      }
      svg.appendChild(rect);

      const allNotes = [note.pitch, ...(note.chordNotes || [])];
      const lineH    = 13;
      const fs       = allNotes.length <= 2 ? 11 : 9;
      const totalH   = allNotes.length * lineH;
      allNotes.forEach((p, pi) => {
        const t = el('text', {
          x: x + w / 2,
          y: NOTE_TOP + cellH / 2 - totalH / 2 + lineH * (pi + 0.8),
          'text-anchor': 'middle', fill: color,
          'font-size': fs, 'font-weight': '700', 'font-family': 'system-ui',
          'pointer-events': 'none',
        });
        t.textContent = formatNoteName(p);
        svg.appendChild(t);
      });
    } else {
      if (onNoteClick) {
        const hitBox = el('rect', {
          x, y: NOTE_TOP, width: w, height: cellH, rx: 4,
          fill: 'transparent', stroke: 'none',
        });
        hitBox.style.cursor = 'pointer';
        hitBox.addEventListener('click', () => onNoteClick(i, note));
        svg.appendChild(hitBox);
      }

      const txt = el('text', {
        x: x + w / 2, y: NOTE_TOP + cellH / 2 + 5,
        'text-anchor': 'middle', fill: color,
        'font-size': '12', 'font-weight': '700', 'font-family': 'system-ui',
        'pointer-events': 'none',
      });
      txt.textContent = formatNoteName(note.pitch);
      svg.appendChild(txt);
    }

    x += note.duration * unitW;
  });

  container.innerHTML = '';
  container.appendChild(svg);
}

// ── 전통 오선보 미니 렌더러 (튜토리얼 "오선보 vs 커스텀 악보" 비교 페이지 전용) ──
// lib/screens/tutorial_screen.dart의 _StaffPainter를 SVG로 옮긴 단순화 버전 —
// 정확한 음악 조판이 목적이 아니라 "전통 악보는 이렇게 생겼다"는 비교 감각만 준다.
const _STAFF_STEP = { C: -2, D: -1, E: 0, F: 1, G: 2, A: 3, B: 4 };

export function renderMiniStaff(container, notes) {
  const W = 400, H = 138, GAP = 14, TOP = 32;
  const svg = el('svg', { width: '100%', height: H, viewBox: `0 0 ${W} ${H}` });

  for (let i = 0; i < 5; i++) {
    const y = TOP + i * GAP;
    svg.appendChild(el('line', { x1: 14, y1: y, x2: W - 10, y2: y, stroke: '#6E6259', 'stroke-width': '1' }));
  }

  const bottomY = TOP + 4 * GAP; // E4 = 오선 맨 아래 줄
  const stepToY = step => bottomY - step * (GAP / 2);
  const totalDur = notes.reduce((s, n) => s + n.duration, 0) || 1;
  const usableW = W - 14 - 10 - 24;
  let x = 14 + 16;

  notes.forEach(note => {
    const w  = (note.duration / totalDur) * usableW;
    const cx = x + w / 2;
    const oct = parseInt(note.pitch.slice(-1));
    const namePart = note.pitch.slice(0, -1);
    const sharp = namePart.endsWith('#');
    const letter = sharp ? namePart[0] : namePart;
    const step = _STAFF_STEP[letter] + 7 * (oct - 4);
    const cy = stepToY(step);

    if (step < 0) {
      for (let s = -2; s >= step; s -= 2) {
        const ly = stepToY(s);
        svg.appendChild(el('line', { x1: cx - 8, y1: ly, x2: cx + 8, y2: ly, stroke: '#6E6259', 'stroke-width': '1' }));
      }
    } else if (step > 8) {
      for (let s = 10; s <= step; s += 2) {
        const ly = stepToY(s);
        svg.appendChild(el('line', { x1: cx - 8, y1: ly, x2: cx + 8, y2: ly, stroke: '#6E6259', 'stroke-width': '1' }));
      }
    }

    if (sharp) {
      const t = el('text', { x: cx - 13, y: cy + 4, 'font-size': '12', fill: '#222', 'font-family': 'system-ui' });
      t.textContent = '♯';
      svg.appendChild(t);
    }

    const hollow = note.duration >= 2;
    svg.appendChild(el('ellipse', {
      cx, cy, rx: 5.5, ry: 4, transform: `rotate(-15 ${cx} ${cy})`,
      fill: hollow ? 'none' : '#222', stroke: '#222', 'stroke-width': hollow ? '1.4' : '0',
    }));
    svg.appendChild(el('line', { x1: cx + 5, y1: cy, x2: cx + 5, y2: cy - 22, stroke: '#222', 'stroke-width': '1.2' }));

    x += w;
  });

  container.innerHTML = '';
  container.appendChild(svg);
}
