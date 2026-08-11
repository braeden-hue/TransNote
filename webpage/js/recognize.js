// recognize.js — /api/recognize(작업 제출) + /api/recognize/status(진행 조회)를 감싼
// 브라우저 쪽 폴링 헬퍼. 예전엔 Vercel 함수 하나가 서버 안에서 최대 54초 동안 RunPod
// 상태를 블로킹 폴링했는데, 콜드스타트가 그보다 더 오래 걸리는 일이 흔해서 실패로
// 이어졌다. 이제 서버는 job 제출만 하고 즉시 응답하고, 완료 여부는 여기서 브라우저가
// 직접 짧은 간격으로 반복 확인한다 — 콜드스타트가 몇 분이 걸려도 각 요청 자체는
// 순식간에 끝나므로 서버 쪽 실행시간 제한과 전혀 무관해진다.

const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000; // 5분 — 이 이상이면 정말 문제가 있는 것으로 보고 포기

// file: 업로드/촬영한 이미지 File. onProgress(선택): { status, delayTimeMs } 콜백 —
// IN_QUEUE/IN_PROGRESS 상태일 때마다 호출(대기 시간 등을 화면에 보여주고 싶을 때 사용).
export async function recognizeImage(file, { model = 'custom', onProgress } = {}) {
  const form = new FormData();
  form.append('file', file);

  const submitRes = await fetch(`/api/recognize?model=${model}`, { method: 'POST', body: form });
  const submitBody = await submitRes.json().catch(() => ({}));
  if (!submitRes.ok || submitBody.error) {
    throw new Error(submitBody.error || `서버 오류 (HTTP ${submitRes.status})`);
  }
  const jobId = submitBody.jobId;
  // 로컬 오프라인 부스용 server/server.py는 RunPod job 큐가 없어 결과를 바로 동기
  // 응답으로 돌려준다(콜드스타트 자체가 없으니 폴링이 필요 없음) — jobId가 없으면
  // submitBody 자체를 최종 결과로 취급한다. Vercel+RunPod 배포에서만 jobId가 온다.
  if (!jobId) return submitBody;

  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const statusRes = await fetch(`/api/recognize/status?id=${encodeURIComponent(jobId)}`);
    const data = await statusRes.json().catch(() => ({}));
    if (!statusRes.ok || data.status === 'FAILED') {
      throw new Error(data.error || `서버 오류 (HTTP ${statusRes.status})`);
    }
    if (data.status === 'COMPLETED') return data.result;
    onProgress?.(data); // { status: 'IN_QUEUE' | 'IN_PROGRESS', delayTimeMs }
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new Error('인식이 너무 오래 걸려요(5분 초과) — 잠시 후 다시 시도해주세요');
}
