import { BEAT_COLORS, pitchToZone, formatNoteName, effectiveClef, hasMixedClef } from './samples.js';

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
  zoneColors,     // 옵션: [z0색, z1색, z2색] — 지정 시 기본 회청색 줄무늬 대신 이 3색으로 존 배경을 칠함 (튜토리얼 규칙2 전용)
  hideNoteNames = false, // 옵션: 음표 위 C/G 등 글자 라벨을 숨김 (튜토리얼 규칙2 전용 — 색/위치만으로 읽게 유도)
  hideNotes = false, // 옵션: 음표 표시(밑줄·박스·글자) 자체를 아예 안 그리고 존 배경만 남김 (튜토리얼 규칙1 전용)
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

  // ── Zone backgrounds ──────────────────────────────────────────────────────
  // (예전엔 존 왼쪽에 "높음(6옥+)" 등 회색 글자 라벨을 같이 그렸으나 전 화면에서 삭제 —
  // 색과 위치만으로 읽는다는 원칙에 맞춰 글자 설명 없이 색 구분만 남긴다.)
  for (let z = 0; z < 3; z++) {
    const zy = CONTENT_Y + z * ZONE_H;
    svg.appendChild(el('rect', {
      x: MARGIN_L, y: zy,
      width: svgW - MARGIN_L - 8, height: ZONE_H,
      fill: zoneColors ? zoneColors[z] : (z % 2 === 0 ? '#F0F6FC' : '#F8FBFF'),
      'fill-opacity': zoneColors ? '0.55' : '1',
    }));
    if (z > 0) {
      svg.appendChild(el('line', {
        x1: MARGIN_L, y1: zy, x2: svgW - 8, y2: zy,
        stroke: '#C5D8EC', 'stroke-width': '1.5', 'stroke-dasharray': '6,4',
      }));
    }
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

    // 규칙1처럼 존 배경만 보여주고 싶을 땐 음표 자체(밑줄/박스/글자/화살표)를 아예 안
    // 그림 — x 진행(너비 계산)은 그대로 유지해야 다른 옵션과 조합해도 스크롤 폭이 맞음.
    if (hideNotes) { x += note.duration * UNIT_W; return; }

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
      if (!hideNoteNames) {
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
      }
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

      if (!hideNoteNames) {
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
    }

    // ── expected 화살표 (단음/화음 공통) ──────────────────────────────────
    if (isExp) {
      const arrowCx  = x + w / 2;
      const arrowTip = y - 2;
      const arrowTop = Math.max(2, arrowTip - 18);
      svg.appendChild(el('rect', {
        x: arrowCx - 8, y: arrowTop - 1, width: 16, height: arrowTip - arrowTop + 2,
        fill: 'rgba(255,255,255,0.85)', rx: '2',
      }));
      const tri = el('polygon', {
        points: `${arrowCx - 7},${arrowTop} ${arrowCx + 7},${arrowTop} ${arrowCx},${arrowTip}`,
        fill: '#E5383B',
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

// 오선 번호(0부터) → 표시 레이블 + 색상. si가 아니라 실제 clef를 기준으로 판단해야
// 순서가 바뀌어도(예: 규칙1의 왼쪽=베이스 배치) 라벨이 안 어긋난다.
function staffLabel(si, clef) {
  if (clef === 'treble') return { label: '🎵 높은음자리 (Treble)', color: '#0076CE' };
  if (clef === 'bass')   return { label: '🎻 낮은음자리 (Bass)', color: '#5BB8F5' };
  const clefName = clef === 'alto' ? '알토' : '높은음자리';
  return { label: `Staff ${si + 1} (${clefName})`, color: '#7BB8A0' };
}

// ── 다중 오선 렌더러 (2개 이상 N개 지원) ─────────────────────────────────────
// options.zoneColorsByClef: { treble: [c0,c1,c2], bass: [c0,c1,c2] } — 클렙별로 다른
// zoneColors를 쓰고 싶을 때(튜토리얼 규칙1 전용). 지정 안 하면 기존 동작 그대로.
// options.layout: 'column'(기본, 위/아래) | 'row'(좌/우 — 가로로 눕힌 화면에서 세로
// 공간을 절반만 쓰게 해서 스크롤 없이 한 화면에 담기 위함, 튜토리얼 규칙1 전용).
export function renderGrandStaff(container, staves, options = {}) {
  container.innerHTML = '';
  const isRow = options.layout === 'row';
  const wrapper = document.createElement('div');
  wrapper.style.cssText = isRow
    ? 'display:flex; flex-direction:row; gap:12px; width:100%;'
    : 'display:flex; flex-direction:column; gap:10px;';

  staves.forEach((stave, si) => {
    const clef = stave.clef ?? (si === 0 ? 'treble' : 'bass');
    const meta = staffLabel(si, clef);

    const row = document.createElement('div');
    row.style.cssText = isRow
      ? 'display:flex; flex-direction:column; gap:3px; flex:1; min-width:0;'
      : 'display:flex; flex-direction:column; gap:3px;';

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

    renderNotation(inner, stave.notes, {
      ...options, clef,
      zoneColors: options.zoneColorsByClef?.[clef],
      // 양손 연주 모드처럼 보표마다 "다음 기대 음" 인덱스가 서로 다를 때 사용
      // (없으면 공용 expectedIdx로 폴백 — 기존 호출부는 그대로 동작).
      expectedIdx: options.expectedIdxByClef?.[clef] ?? options.expectedIdx,
    });
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

// ── 정식 오선보(디지털 악보) 렌더러 — 촬영 미리보기에서 "원본 사진과 비교"용 ──────
// 예전엔 VexFlow를 외부 CDN으로 불러오다 부스 네트워크 신뢰성 문제로 자체 SVG 렌더러로
// 갈아탔었는데(손으로 그린 오선/음표라 MuseScore 대비 조판 품질이 크게 떨어졌음), 이번엔
// VexFlow 파일 자체를 로컬 vendor/에 내려받아 CDN 의존 없이 같은 엔진(전문 조판 폰트 내장,
// 자동 스템 방향·빔·부점·임시표 처리)을 쓴다 — index.html에서 일반 <script>로 먼저 실행돼
// window.VexFlow에 채워둔다.
const _DUR_TABLE = [
  [4, 'w'], [3, 'hd'], [2, 'h'], [1.5, 'qd'], [1, 'q'],
  [0.75, '8d'], [0.5, '8'], [0.375, '16d'], [0.25, '16'], [0.125, '32'],
];
// 우리 표기(quarter=1박) duration -> VexFlow duration 문자열. 셋잇단음표처럼 표에 없는
// 박은 일단 4분음표로 근사한다(정확한 tuplet 그룹핑은 추후 개선 여지로 남겨둠).
function _vexDuration(d) {
  const hit = _DUR_TABLE.find(([v]) => Math.abs(v - d) < 0.001);
  return hit ? hit[1] : 'q';
}
function _vexKey(pitch) {
  return `${pitch.slice(0, -1).toLowerCase()}/${pitch.slice(-1)}`;
}
// beat===1 경계로 마디 분리 — app.js의 splitMeasures()와 같은 규칙.
function _splitMeasures(notes) {
  const measures = []; let cur = [];
  notes.forEach(n => {
    if (n.beat === 1 && cur.length) { measures.push(cur); cur = []; }
    cur.push(n);
  });
  if (cur.length) measures.push(cur);
  return measures;
}

// container에 정식 오선보를 그린다. staves: [{clef, notes}] (1개=단일보표, 2개=대보표).
// timeSignature: [분자, 분모] — 있으면 첫 마디 앞에 박자표를 붙인다.
// 곡 전체(여러 마디)를 한 줄로 이어 그리므로, 컨테이너보다 넓어지면 SVG 자체를 그만큼
// 넓게 만든다 — 부모(.exp-preview-notation 등)가 overflow:auto라 가로 스크롤로 볼 수 있다.
export function renderDigitalStaff(container, staves, timeSignature) {
  const VF = window.VexFlow;
  container.innerHTML = '';
  if (!VF) {
    // vendor/vexflow.js 로드 실패 등 — 조용히 빈 화면보다는 이유를 알 수 있게.
    container.innerHTML = '<p style="color:#a00;padding:20px">악보 렌더링 엔진을 불러오지 못했습니다</p>';
    return;
  }
  const { Renderer, Stave, StaveNote, Voice, Formatter, Beam, Accidental, StaveConnector, Dot, Tuplet } = VF;

  const normalized = staves.map(s => ({
    clef: s.clef ?? 'treble',
    notes: s.notes?.length ? s.notes : [{ isRest: true, duration: 4, beat: 1 }],
  }));
  const isGrand = normalized.length >= 2;
  const measuresByStave = normalized.map(s => _splitMeasures(s.notes));
  const measureCount = Math.max(1, ...measuresByStave.map(m => m.length));

  // 한 마디(보표 하나 분량)를 StaveNote 배열로 변환. clef는 이 보표 전체의 clef —
  // note 자체엔 clef 필드가 없다(treble/bass 배열 소속 자체가 clef를 암시하는 백엔드
  // 관례, token_to_notes.py 참고). 이걸 안 넘기면 StaveNote가 기본값(treble)으로 음높이를
  // 계산해버려서 베이스 음표가 높은음자리표 기준 위치에 그려지는 버그가 있었다 — 실제
  // 오선(stave)엔 bass clef가 정확히 그려지는데 그 위의 개별 음표 위치만 엉뚱한 채였음.
  // 셋잇단음표(note.tuplet===true, token_to_notes.py가 실제 길이를 2/3로 이미 보정해둔
  // 상태)는 "인쇄상"의 8분음표로 만들고(VexFlow Tuplet이 실제 틱 길이를 알아서 3:2로
  // 줄여줌), 항상 3개씩 연속으로 온다는 백엔드 관례를 그대로 이용해 3개 단위로 묶어
  // tuplets 배열에 담아 반환 — 호출부가 Tuplet(...)으로 감싸 괄호+"3" 표기를 그린다.
  function buildMeasureNotes(measureNotes, clef) {
    const restKey = clef === 'bass' ? 'd/3' : 'b/4'; // 각 clef의 오선 가운데 줄
    if (!measureNotes?.length) return { notes: [new StaveNote({ keys: [restKey], duration: 'wr', clef })], beats: 4, tuplets: [] };
    let beats = 0;
    const notes = measureNotes.map(n => {
      beats += n.duration;
      const dur = n.tuplet ? '8' : _vexDuration(n.duration);
      if (n.isRest) return new StaveNote({ keys: [restKey], duration: dur + 'r', clef });
      const pitches = [n.pitch, ...(n.chordNotes || [])];
      const sn = new StaveNote({ keys: pitches.map(_vexKey), duration: dur, clef, autoStem: true });
      pitches.forEach((p, i) => { if (p.includes('#')) sn.addModifier(new Accidental('#'), i); });
      if (dur.includes('d')) Dot.buildAndAttach([sn], { all: true });
      return sn;
    });
    const tuplets = [];
    for (let i = 0; i < notes.length; ) {
      if (measureNotes[i]?.tuplet) {
        tuplets.push(notes.slice(i, i + 3)); // mscz_to_tokens.py 관례상 3연음은 항상 3개
        i += 3;
      } else {
        i += 1;
      }
    }
    return { notes, beats: beats || 4, tuplets };
  }

  // 마디별 폭 — 그 마디에서 음이 가장 많은 보표 기준(대보표는 두 보표의 바라인이
  // 같은 x에 오도록 맞춰야 함). 첫 마디는 클렙+박자표 자리만큼 더 넓게.
  const MEASURE_PAD = 46, PX_PER_NOTE = 42, FIRST_MEASURE_EXTRA = 60;
  const measureWidths = Array.from({ length: measureCount }, (_, m) => {
    const maxNotes = Math.max(1, ...measuresByStave.map(ms => ms[m]?.length ?? 1));
    return MEASURE_PAD + maxNotes * PX_PER_NOTE + (m === 0 ? FIRST_MEASURE_EXTRA : 0);
  });
  const totalW = measureWidths.reduce((a, b) => a + b, 0) + 14;
  const STAVE_H = 100;
  const totalH = (isGrand ? STAVE_H * 2 : STAVE_H) + 20;

  const renderer = new Renderer(container, Renderer.Backends.SVG);
  renderer.resize(Math.max(totalW, container.clientWidth || 320), totalH);
  const ctx = renderer.getContext();

  let x = 6;
  for (let m = 0; m < measureCount; m++) {
    const w = measureWidths[m];
    const rowStaves = [];
    normalized.forEach((s, si) => {
      const stave = new Stave(x, 10 + si * STAVE_H, w);
      if (m === 0) {
        stave.addClef(s.clef);
        if (timeSignature) stave.addTimeSignature(`${timeSignature[0]}/${timeSignature[1]}`);
      }
      stave.setContext(ctx).draw();
      rowStaves.push(stave);

      const { notes, beats, tuplets } = buildMeasureNotes(measuresByStave[si][m], s.clef);
      const voice = new Voice({ numBeats: beats, beatValue: 4 }).setStrict(false);
      voice.addTickables(notes);
      new Formatter().joinVoices([voice]).format([voice], w - (m === 0 ? 70 : 30));
      voice.draw(ctx, stave);
      Beam.generateBeams(notes).forEach(b => b.setContext(ctx).draw());
      tuplets.forEach(group => new Tuplet(group).setContext(ctx).draw());
    });
    // 대보표면 맨 앞 마디 앞에만 브레이스(중괄호)+연결 바라인을 그려 대보표임을 표시.
    if (isGrand && m === 0) {
      new StaveConnector(rowStaves[0], rowStaves[1]).setType(StaveConnector.type.BRACE).setContext(ctx).draw();
      new StaveConnector(rowStaves[0], rowStaves[1]).setType(StaveConnector.type.SINGLE_LEFT).setContext(ctx).draw();
    }
    x += w;
  }
}
