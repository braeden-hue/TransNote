import { BEAT_COLORS, pitchToZone, formatNoteName } from './samples.js';

const UNIT_W      = 80;
const CELL_H      = 46;
const ZONE_H      = CELL_H + 10;  // 56
const MARGIN_L    = 68;
const MARGIN_Y    = 8;
const INDICATOR_H = 18;  // 화살표 표시 영역 (상단)
const NS          = 'http://www.w3.org/2000/svg';

const ZONE_LABELS = ['높음 (5옥+)', '중간 (4옥)', '낮음 (3옥↓)'];

// 가온다(C4) 기준 존 — 중간(4옥)
const REF_ZONE = 1;

function el(tag, attrs = {}) {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  return e;
}

export function renderNotation(container, notes, {
  highlightIdx = -1,
  expectedIdx  = -1,
  onNoteClick,
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
  let x = MARGIN_L + 4;
  const noteXMap = [];

  notes.forEach((note, i) => {
    const w      = note.duration * UNIT_W - 4;
    const zone   = pitchToZone(note.pitch);
    const y      = CONTENT_Y + zone * ZONE_H + 5;
    const h      = CELL_H;
    const color  = BEAT_COLORS[note.beat] || '#888';
    const isHL   = i === highlightIdx;
    const isExp  = i === expectedIdx;

    noteXMap.push(x);

    const fill = isHL  ? color + '44'
               : isExp ? '#0076CE18'
               : '#FFFFFF';

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

    // ── 눌러야 할 음표 위에 내려가는 화살표 ─────────────────────────────────
    if (isExp) {
      const arrowCx  = x + w / 2;
      const arrowTip = y - 2;                       // 노트 셀 바로 위
      const arrowTop = Math.max(2, arrowTip - 13);  // 화살표 몸통 상단
      // 흰 배경(다른 존 배경이 겹쳐도 화살표가 선명하게)
      svg.appendChild(el('rect', {
        x: arrowCx - 6, y: arrowTop - 1, width: 12, height: arrowTip - arrowTop + 2,
        fill: 'rgba(255,255,255,0.85)', rx: '2',
      }));
      // 삼각형 화살표
      const tri = el('polygon', {
        points: `${arrowCx - 5},${arrowTop} ${arrowCx + 5},${arrowTop} ${arrowCx},${arrowTip}`,
        fill: '#0076CE',
      });
      // 위아래 바운스 애니메이션
      const anim2 = document.createElementNS(NS, 'animateTransform');
      anim2.setAttribute('attributeName', 'transform');
      anim2.setAttribute('type', 'translate');
      anim2.setAttribute('values', '0 0;0 3;0 0');
      anim2.setAttribute('dur', '0.65s');
      anim2.setAttribute('repeatCount', 'indefinite');
      tri.appendChild(anim2);
      svg.appendChild(tri);
    }

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

    const fs = w < 26 ? 9 : w < 42 ? 11 : 13;
    const txt = el('text', {
      x: x + w / 2, y: y + h / 2 + 5,
      'text-anchor': 'middle',
      fill: isHL ? '#fff' : strokeColor,
      'font-size': fs, 'font-weight': '700', 'font-family': 'system-ui',
      'pointer-events': 'none',
    });
    txt.textContent = formatNoteName(note.pitch);
    svg.appendChild(txt);

    x += note.duration * UNIT_W;
  });

  // ── 가온다(C4) 기준 표시 — 마디 시작마다 중간(4옥) 존 중앙에 파란 점 ──────────
  const refCY = CONTENT_Y + REF_ZONE * ZONE_H + ZONE_H / 2;

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

// ── 그랜드 스태프 (높은음자리 + 낮은음자리 병렬 표시) ─────────────────────────────
export function renderGrandStaff(container, staves, options = {}) {
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'display:flex; flex-direction:column; gap:10px;';

  const staffMeta = [
    { label: '🎵 높은음자리 (Treble)', color: '#0076CE' },
    { label: '🎵 낮은음자리 (Bass)',   color: '#5BB8F5' },
  ];

  staves.forEach((stave, si) => {
    const meta = staffMeta[si] ?? { label: `Staff ${si + 1}`, color: '#aaa' };

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

    renderNotation(inner, stave.notes, options);
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

  // Note cells
  let x = 4;
  notes.forEach((note, i) => {
    const w     = note.duration * unitW - 4;
    const color = BEAT_COLORS[note.beat] || '#888';

    const rect = el('rect', {
      x, y: NOTE_TOP, width: w, height: cellH, rx: 4,
      fill: '#FFFFFF', stroke: color, 'stroke-width': '2',
    });
    if (onNoteClick) {
      rect.style.cursor = 'pointer';
      rect.addEventListener('click', () => onNoteClick(i, note));
    }
    svg.appendChild(rect);

    const txt = el('text', {
      x: x + w / 2, y: NOTE_TOP + cellH / 2 + 5,
      'text-anchor': 'middle', fill: color,
      'font-size': '12', 'font-weight': '700', 'font-family': 'system-ui',
      'pointer-events': 'none',
    });
    txt.textContent = formatNoteName(note.pitch);
    svg.appendChild(txt);

    x += note.duration * unitW;
  });

  container.innerHTML = '';
  container.appendChild(svg);
}
