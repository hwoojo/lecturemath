"use strict";
const api = window.BOARDNOTES_API_URL.replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
let health = null;
let activeJob = false;
const stages = {queued:"처리 대기",loading_model:"모델 준비",binarizing:"판서 추출",cc_analysis:"같은 판서 추적",cc_grouping:"판서 통합",deletion_segmentation:"삭제 시점 분석",keyframes:"페이지 구성",pdf:"PDF 생성",complete:"판서 노트 완성",failed:"처리 실패"};
function showError(message) { $("error").textContent = message; $("error").hidden = false; }
async function responseJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `요청 실패 (${response.status})`);
  return body;
}
async function checkServer() {
  try {
    health = await responseJson(await fetch(`${api}/health`, {signal: AbortSignal.timeout(10000)}));
    $("limits").textContent = `최대 ${health.max_upload_mb} MB · ${Math.floor(health.max_duration_seconds/60)}분 · CPU 환경에서는 처리 시간이 길어질 수 있습니다.`;
    $("submit").disabled = activeJob || !health.model_file_present;
    if (!health.model_file_present) showError("서버에 V34 모델 파일이 준비되지 않았습니다. 서버 관리자가 체크포인트를 설정한 뒤 페이지를 새로고침해 주세요.");
    return health.model_file_present;
  } catch (error) {
    showError(`추론 서버에 연결할 수 없습니다. API 주소와 서버 실행 상태를 확인해 주세요.\n${error.message}`);
    $("limits").textContent = `API: ${api}`;
    return false;
  }
}
$("video").addEventListener("change", () => {
  const file = $("video").files[0];
  $("filename").textContent = file ? `${file.name} · ${(file.size/1024/1024).toFixed(1)} MB` : "선택한 파일이 없습니다.";
});
async function poll(jobId) {
  $("download").hidden = true;
  for (;;) {
    const job = await responseJson(await fetch(`${api}/jobs/${jobId}`, {signal: AbortSignal.timeout(15000)}));
    $("status").hidden = false;
    $("stage").textContent = stages[job.stage] || job.stage;
    $("progress").value = job.progress;
    $("percent").textContent = `${job.progress}%`;
    $("detail").textContent = `작업 ID: ${jobId} · 페이지를 새로고침해도 이 브라우저에서 진행 상황을 다시 확인합니다.`;
    if (job.status === "failed") { localStorage.removeItem("boardnotesJob"); throw new Error(job.error || "영상 처리에 실패했습니다."); }
    if (job.status === "complete") {
      $("download").href = `${api}/jobs/${jobId}/pdf`;
      $("download").hidden = false;
      $("detail").textContent = "PDF가 준비되었습니다. 서버 보관 기간 내에 다운로드해 주세요.";
      localStorage.removeItem("boardnotesJob");
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("video").files[0];
  if (!file || activeJob) return;
  $("error").hidden = true;
  $("download").hidden = true;
  activeJob = true; $("submit").disabled = true;
  try {
    if (health && file.size > health.max_upload_mb*1024*1024) throw new Error("업로드 크기 제한을 초과했습니다.");
    $("status").hidden = false; $("stage").textContent = "영상 업로드 중"; $("percent").textContent = ""; $("progress").removeAttribute("value");
    $("detail").textContent = "업로드가 끝나면 판서 분석을 시작합니다.";
    const query = new URLSearchParams({fps: $("fps").value, timestamps: $("timestamps").checked});
    const job = await responseJson(await fetch(`${api}/jobs?${query}`, {method:"POST", headers:{"Content-Type":"application/octet-stream"}, body:file}));
    localStorage.setItem("boardnotesJob", JSON.stringify({api, id:job.id}));
    await poll(job.id);
  } catch (error) { showError(error.message); }
  finally { activeJob = false; $("submit").disabled = !health?.model_file_present; }
});
async function init() {
  const ready = await checkServer();
  let saved;
  try { saved = JSON.parse(localStorage.getItem("boardnotesJob")); } catch { localStorage.removeItem("boardnotesJob"); }
  if (ready && saved?.api === api && /^[0-9a-f]{32}$/.test(saved.id)) {
    activeJob = true; $("submit").disabled = true;
    try { await poll(saved.id); } catch(error) { showError(error.message); }
    finally { activeJob = false; $("submit").disabled = false; }
  }
}
init();
