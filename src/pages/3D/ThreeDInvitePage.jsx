import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./ThreeDInvitePage.module.css";

import { useInvitation } from "../../contexts/InvitationContext.jsx";

// ✅ 이미 만들어둔 컴포넌트들
import SingleImageUploader from "../../components/common/SingleImageUploader.jsx";
import ImageUploader from "../../components/common/ImageUploader.jsx";
import { useNavigate } from "react-router-dom";

export function ThreeDInvitePage() {
  const {
    data,
    setThreeDMainImage,
    setThreeDReferenceImages,
    startThreeDJob,
    stopThreeDPolling,
  } = useInvitation();

  const navigate = useNavigate();
  const movedRef = useRef(false);

  // ====== Context에서 초기값 가져오기 ======
  const initialMain = useMemo(() => data?.threeD?.mainImages?.[0] || null, [data?.threeD?.mainImages]);
  const initialRefs = useMemo(() => data?.threeD?.referenceImages || [], [data?.threeD?.referenceImages]);

  // ====== 로컬 state ======
  const [mainImage, setMainImage] = useState(initialMain); // File | null
  const [refFiles, setRefFiles] = useState(initialRefs);   // File[]

  // 토스트 (기존 로직 유지 가능)
  const [toast, setToast] = useState({ open: false, message: "", variant: "info" });
  const toastTimerRef = useRef(null);
  const lastToastKeyRef = useRef("");

  const threeD = data?.threeD || {};
  const isGenerating = ["SUBMITTED", "PENDING", "RUNNING"].includes(threeD.status);

  const canSubmit = !!mainImage && !isGenerating;

  const showToast = (message, variant = "info") => {
    if (!message) return;
    const key = `${variant}:${message}`;
    if (lastToastKeyRef.current === key) return;
    lastToastKeyRef.current = key;

    setToast({ open: true, message, variant });

    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => {
      setToast((prev) => ({ ...prev, open: false }));
    }, 1800);
  };

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    };
  }, []);

  // ====== 메인 이미지 변경 => context 저장 ======
  const handleMainChange = (fileOrNull) => {
    setMainImage(fileOrNull);
    setThreeDMainImage(fileOrNull); // ✅ context.threeD.mainImages = [file] or []
  };

  // ====== 레퍼런스 변경 => context 저장 ======
  const handleRefChange = (nextFiles) => {
    const limited = (nextFiles || []).filter(Boolean).slice(0, 2);
    setRefFiles(limited);
    setThreeDReferenceImages(limited);
  };

  // ✅ threeD 상태 토스트 (원래 쓰던 거랑 동일하게 유지)
  useEffect(() => {
    const s = threeD.status;
    if (!s || s === "IDLE") return;

    if (s === "SUBMITTED") showToast("요청을 전송했어요. 생성 준비 중…", "info");
    if (s === "PENDING") showToast("대기열에 들어갔어요. 잠시만요…", "info");
    if (s === "RUNNING") showToast("3D 생성 중입니다! (최대 10분)", "info");
    if (s === "DONE") showToast("완성됐어요! 결과를 확인해보세요 ✅", "success");
    if (s === "FAILED") showToast(threeD.error || "생성에 실패했어요 😢", "error");
    if (s === "CANCELED") showToast("요청을 취소했어요.", "warn");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threeD.status]);

    useEffect(() => {
      if (data?.threeD?.status === "DONE") {
        // model3dUrl은 다음 페이지에서 사용
        navigate(STEPS.threedResult.path); // 너가 만든 결과 페이지 경로로
      }
    }, [data?.threeD?.status, navigate]);

  useEffect(() => {
    if (!threeD.error) return;
    showToast(threeD.error, "error");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threeD.error]);

  // ====== 제출 ======
  const handleSubmit = async () => {
    if (!mainImage || isGenerating) return;

    try {
      const references = (refFiles || []).filter(Boolean).slice(0, 2);

      // (디버깅용) 실제 넘어가는지 확인
      console.log("[3D SUBMIT] main:", mainImage);
      console.log("[3D SUBMIT] refs:", references);

      await startThreeDJob({
        submitUrl: "/api/invitations/3d",
        statusUrl: "/api/invitations/3d/status",
        intervalMs: 5000,
        mainOverride: mainImage,
        referencesOverride: references,
      });
    } catch (e) {
      showToast(e?.message || "요청 중 오류가 발생했어요.", "error");
    }
  };

  const handleCancel = () => {
    stopThreeDPolling?.("CANCELED");
  };

  return (
    <div className={styles.page}>
      {/* 우측 상단 토스트 */}
      <div
        className={`${styles.toast} ${toast.open ? styles.toastOpen : ""} ${
          styles[`toast_${toast.variant}`] || ""
        }`}
        role="status"
        aria-live="polite"
      >
        {toast.message}
      </div>

      <div className={styles.bgOrbs} aria-hidden="true">
        <span className={styles.orb1} />
        <span className={styles.orb2} />
        <span className={styles.orb3} />
      </div>

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <div className={styles.emoji} aria-hidden="true">👰🏻 🧡 🤵🏻</div>
            <h1 className={styles.title}>3D 청첩장 만들기</h1>
            <p className={styles.subTitle}>우리만의 사진으로 특별한 3D 청첩장을 만들 수 있어요!</p>
          </header>

          {/* ✅ 메인 이미지(필수) */}
          <section className={styles.section}>
            <div className={styles.sectionTop}>
              <div className={styles.sectionTitle}>
                1. 메인 사진 <span className={styles.req}>*</span>
              </div>
              {!!mainImage && <div className={styles.pillOk}>업로드 완료</div>}
            </div>

            <SingleImageUploader
              label="메인 사진 업로드"
              helperText="신랑/신부가 함께 나온 대표 사진 1장을 올려주세요"
              buttonText="메인 사진 선택"
              accept="image/*"
              value={mainImage}
              onChange={handleMainChange}
            />
          </section>

          {/* ✅ 레퍼런스 이미지(선택) */}
          <section className={styles.section}>
            <div className={styles.sectionTop}>
              <div className={styles.sectionTitle}>2. 레퍼런스 사진 (선택, 최대 2장)</div>
              <div className={styles.hintText}>
                원하는 분위기/포즈 참고용 (없어도 진행 가능)
              </div>
            </div>

            <ImageUploader maxCount={2} value={refFiles} onChange={handleRefChange} accept="image/*" />
          </section>

          {/* 상태 표시 */}
          {threeD.status && threeD.status !== "IDLE" && (
            <div className={styles.statusRow}>
              <div className={styles.statusPill}>
                {threeD.status === "SUBMITTED" && "요청 전송"}
                {threeD.status === "PENDING" && "대기 중"}
                {threeD.status === "RUNNING" && "생성 중"}
                {threeD.status === "DONE" && "완료"}
                {threeD.status === "FAILED" && "실패"}
                {threeD.status === "CANCELED" && "취소됨"}
              </div>
              {!!threeD.jobId && <div className={styles.jobId}>jobId: {threeD.jobId}</div>}
            </div>
          )}

          {/* 버튼 */}
          <div className={styles.footer}>
            {isGenerating ? (
              <button type="button" className={styles.dangerBtn} onClick={handleCancel}>
                생성 취소
              </button>
            ) : (
              <button
                type="button"
                className={styles.primaryBtn}
                onClick={handleSubmit}
                disabled={!canSubmit}
              >
                다음
              </button>
            )}
          </div>

          {!mainImage && (
            <div className={styles.notice}>
              * 메인 사진 1장은 반드시 업로드해야 다음 단계로 진행할 수 있어요
            </div>
          )}

          {threeD.status === "DONE" && (
            <div className={styles.doneBox}>
              <div className={styles.doneTitle}>✅ 3D 생성이 완료됐어요</div>
              <div className={styles.doneDesc}>threeD.result에 결과가 저장되었습니다.</div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
