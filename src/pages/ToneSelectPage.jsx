import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./ToneSelectPage.module.css";
import ProgressBar from "../components/common/ProgressBar";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";
import { useInvitation } from "../contexts/InvitationContext.jsx";

export function ToneSelectPage() {
  const navigate = useNavigate();
  const { data, updateField } = useInvitation();

  // ✅ 옵션 id는 "상수" 느낌으로 전부 대문자
  const toneOptions = useMemo(
    () => [
      { id: "FORMAL", title: "격식있는", desc: "전통적이고 예의바른 표현", example: "삼가 청하옵건대…" },
      { id: "WARM", title: "따뜻한", desc: "정감있고 포근한 느낌", example: "따뜻한 마음으로 초대합니다" },
      { id: "MODERN", title: "현대적인", desc: "세련되고 트렌디한 표현", example: "저희의 새로운 시작에 함께해 주세요" },
      { id: "CLASSIC", title: "클래식", desc: "고전적이고 우아한 분위기", example: "영원한 사랑을 약속하는 자리에" },
      { id: "CASUAL", title: "캐주얼", desc: "편안하고 친근한 느낌", example: "우리 결혼해요! 축하해주러 와요" },
      { id: "ROMANTIC", title: "로맨틱", desc: "감성적이고 낭만적인 표현", example: "사랑이 꽃피는 그날, 함께해주세요" },
    ],
    []
  );

  // ✅ context에 tone이 있으면 그걸로 시작, 없으면 기본값 WARM
  const [selectedTone, setSelectedTone] = useState(() => data.tone ?? "WARM");

  // ✅ 페이지 재진입/새로고침 후에도 context가 바뀌면 UI 선택값도 동기화
  useEffect(() => {
    if (data.tone && data.tone !== selectedTone) {
      setSelectedTone(data.tone);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.tone]);

  const btnRefs = useRef([]);

  // (선택) 디버깅
  useEffect(() => {
    console.log("[InvitationContext data]", data);
  }, [data]);

  const persistTone = () => {
    // ✅ Context에 상수 형태로 저장
    updateField("tone", selectedTone);
  };

  const handleNext = () => {
    persistTone();
    console.log("선택된 톤(저장):", selectedTone);
    navigate(STEPS.design.path);
  };

  const handleBack = () => {
    persistTone();
    console.log("선택된 톤(저장):", selectedTone);
    navigate(STEPS.image.path);
  };

  const getCols = () => {
    const w = window.innerWidth;
    if (w >= 1024) return 3;
    if (w >= 768) return 2;
    return 1;
  };

  const onGridKeyDown = (e) => {
    const idx = toneOptions.findIndex((t) => t.id === selectedTone);
    const cols = getCols();
    let next = idx >= 0 ? idx : 0;

    if (e.key === "ArrowRight") next = Math.min(toneOptions.length - 1, next + 1);
    if (e.key === "ArrowLeft") next = Math.max(0, next - 1);
    if (e.key === "ArrowDown") next = Math.min(toneOptions.length - 1, next + cols);
    if (e.key === "ArrowUp") next = Math.max(0, next - cols);

    if (["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) {
      e.preventDefault();
      const nextId = toneOptions[next]?.id;
      if (nextId) {
        setSelectedTone(nextId);
        btnRefs.current[next]?.focus?.();
      }
    }

    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (idx >= 0) setSelectedTone(toneOptions[idx].id);
    }
  };

  return (
    <div className={styles.page}>
      <ProgressBar title="청첩장 만들기" currentStep={3} totalSteps={7} />

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <p className={styles.stepLabel}>STEP 3</p>
            <h1 className={`${styles.pageTitle} step-title`}>문구 톤 선택</h1>
            <p className={styles.subTitle}>청첩장에 사용할 문구의 톤을 선택해주세요</p>
          </header>

          <div
            className={styles.grid}
            role="radiogroup"
            aria-label="문구 톤 선택"
            tabIndex={0}
            onKeyDown={onGridKeyDown}
          >
            {toneOptions.map((opt, i) => {
              const selected = opt.id === selectedTone;

              return (
                <button
                  key={opt.id}
                  ref={(el) => (btnRefs.current[i] = el)}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => setSelectedTone(opt.id)}
                  className={`${styles.toneCard} ${selected ? styles.selected : ""}`}
                >
                  <div className={styles.cardTop}>
                    <div>
                      <h3 className={styles.toneTitle}>{opt.title}</h3>
                      <p className={styles.toneDesc}>{opt.desc}</p>
                    </div>

                    <span
                      className={`${styles.check} ${selected ? styles.checkOn : ""}`}
                      aria-hidden="true"
                    >
                      ✓
                    </span>
                  </div>

                  <div className={styles.exampleBox}>
                    <span className={styles.exampleText}>{opt.example}</span>
                  </div>
                </button>
              );
            })}
          </div>

          <footer className={styles.footer}>
            <button type="button" className={styles.nextBtn} onClick={handleNext}>
              다음 단계
            </button>
          </footer>
        </section>
      </main>
    </div>
  );
}
