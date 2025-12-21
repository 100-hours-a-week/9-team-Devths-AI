import React, { useMemo, useRef, useState } from "react";
import styles from "./ResultPage.module.css";
import ProgressBar from "../components/common/ProgressBar";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";
import { useInvitation } from "../contexts/InvitationContext.jsx";
import { useEffect } from "react";

export function ResultPage() {

   const { data } = useInvitation();

   useEffect(() => {
      console.log("[InvitationContext data - 최종]", data);
   }, []);

  const navigate = useNavigate();

  const images = useMemo(() => {
    const designImages = data?.design?.result2dImageUrls;
    console.log('제발: ', designImages);
    if (Array.isArray(designImages) && designImages.length) {
      return designImages.map((url, idx) => ({
        id: `result-${idx + 1}`,
        title: `청첩장 ${idx + 1}`,
        desc: idx === 0
          ? "STEP2에서 업로드한 웨딩 사진 + 배경 디자인 + 선택한 테두리"
          : "백엔드에서 전달된 최종 이미지",
        src: url,
      }));
    }

    const apiImages = data?.threeD?.result?.images;
    if (apiImages && Array.isArray(apiImages) && apiImages.length) {
      return apiImages.map((url, idx) => ({
        id: `result-${idx + 1}`,
        title: `청첩장 ${idx + 1}`,
        desc: "백엔드에서 전달된 최종 이미지",
        src: url,
      }));
    }

    return [
      {
        id: "result-1",
        title: "페이지 1",
        desc: "STEP2에서 업로드한 웨딩 사진 + 배경 디자인 + 선택한 테두리",
        src: "/images/1.png",
      },
      {
        id: "result-2",
        title: "페이지 2",
        desc: "문구/정보가 포함된 청첩장",
        src: "/images/2.png",
      },
      {
        id: "result-3",
        title: "페이지 3",
        desc: "추가 옵션으로 꾸며본 레이아웃 예시",
        src: "/images/3.png",
      },
    ];
  }, [data?.design?.result2dImageUrls, data?.threeD?.result?.images]);

  const [index, setIndex] = useState(0);
  const startXRef = useRef(0);
  const draggingRef = useRef(false);

  const total = images.length;
  const current = images[index];

  const clampIndex = (next) => Math.max(0, Math.min(total - 1, next));

  const goPrev = () => setIndex((i) => clampIndex(i - 1));
  const goNext = () => setIndex((i) => clampIndex(i + 1));

  // --- swipe ---
  const onTouchStart = (e) => {
    draggingRef.current = true;
    startXRef.current = e.touches[0].clientX;
  };

  const onTouchEnd = (e) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    const endX = e.changedTouches[0].clientX;
    const dx = endX - startXRef.current;

    // threshold: 스와이프 감지 거리
    const TH = 50;
    if (dx > TH) goPrev();
    if (dx < -TH) goNext();
  };

  const downloadAll = () => {
    // 브라우저 정책상 연속 다운로드가 막힐 수 있음 (테스트용으로는 OK)
    images.forEach((img, i) => {
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = img.src;
        a.download = `${img.id}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }, i * 250);
    });
  };

  const goOptions = () => {
    navigate(STEPS.options.path);
  };

  const redoDesign = () => {
    navigate(STEPS.design.path);
  };

  return (
    <div className={styles.page}>
      {/* 네 프로젝트 흐름 유지: 7스텝 중 "결과"가 마지막이라고 가정 */}
      <ProgressBar title="청첩장 만들기" currentStep={7} totalSteps={7} />

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <p className={styles.stepLabel}>STEP 7</p>
            <h1 className={`${styles.pageTitle} step-title`}>최종 결과</h1>
            <p className={styles.hint}>
              <span aria-hidden="true">📱</span> 좌우로 스와이프하여 {total}장의 청첩장을 확인하세요
            </p>
          </header>

          <div className={styles.viewerCard}>
            <div className={styles.viewerTop}>
              <div className={styles.viewerBadge}>청첩장 {index + 1}</div>
              <div className={styles.counter} aria-label={`현재 ${index + 1} / ${total}`}>
                {index + 1} / {total}
              </div>
            </div>

            <div
              className={styles.viewer}
              onTouchStart={onTouchStart}
              onTouchEnd={onTouchEnd}
            >
              <button
                type="button"
                className={`${styles.arrow} ${styles.left}`}
                onClick={goPrev}
                disabled={index === 0}
                aria-label="이전 이미지"
              >
                ‹
              </button>

              <div
                className={styles.track}
                style={{ transform: `translateX(-${index * 100}%)` }}
              >
                {images.map((img) => (
                  <article key={img.id} className={styles.slide}>
                    <div className={styles.poster}>
                      <img className={styles.img} src={img.src} alt={img.title} />
                    </div>

                    <div className={styles.meta}>
                      <h2 className={styles.title}>{img.title}</h2>
                      <p className={styles.desc}>{img.desc}</p>
                    </div>
                  </article>
                ))}
              </div>

              <button
                type="button"
                className={`${styles.arrow} ${styles.right}`}
                onClick={goNext}
                disabled={index === total - 1}
                aria-label="다음 이미지"
              >
                ›
              </button>
            </div>

            <div className={styles.dots} role="tablist" aria-label="청첩장 페이지">
              {images.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  className={`${styles.dot} ${i === index ? styles.dotOn : ""}`}
                  onClick={() => setIndex(i)}
                  aria-label={`${i + 1}번 이미지로 이동`}
                  aria-current={i === index ? "true" : "false"}
                />
              ))}
            </div>
          </div>

          <div className={styles.ctaRow}>
            <button type="button" className={`${styles.ctaBtn} ${styles.ctaBtnPrimary}`} onClick={downloadAll}>
              청첩장 {total}장 모두 다운로드
            </button>
            <button type="button" className={`${styles.ctaBtn} ${styles.ctaBtnOutline}`} onClick={redoDesign}>
              디자인 다시하기
            </button>
          </div>
        </section>
        <button type="button" className={styles.tryOptionsBtn} onClick={goOptions}>
          ✨ 3D 모바일 청첩장 기능 확인해보기
        </button>
      </main>
    </div>
  );
}
