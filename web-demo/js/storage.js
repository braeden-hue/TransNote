const STORAGE_KEY = 'omr_notations';

export function loadAll() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

export function saveNotation(notation) {
  const all = loadAll();
  const existing = all.findIndex(n => n.id === notation.id);
  if (existing >= 0) {
    all[existing] = notation;
  } else {
    all.unshift(notation);
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  return notation;
}

export function deleteNotation(id) {
  const all = loadAll().filter(n => n.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
}

export function getById(id) {
  return loadAll().find(n => n.id === id) || null;
}

export function generateId() {
  return 'n_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
}
