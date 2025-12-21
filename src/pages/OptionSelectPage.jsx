// src/pages/OptionSelectPage.jsx
import React, { useMemo } from "react";
import styles from "./OptionSelectPage.module.css";
import ProgressBar from "../components/common/ProgressBar";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";

export function OptionSelectPage() {
  const navigate = useNavigate();

  const handleNext = () => {
      navigate(STEPS.threed.path);
   };

  const options = useMemo(
    () => [
      {
        id: "3d",
        theme: "blue",
        title: "3D 청첩장 만들기",
        desc: "신랑신부의 3D 캐릭터로 특별한 청첩장 제작",
        bullets: ["3D 캐릭터 생성", "같이 나온 사진 3장 필요", "몸의 60~70% 사진"],
        cta: "시작하기",
        onClick: () => {
          // TODO: 나중에 3D 기능 페이지로 라우팅
          handleNext();
        },
      },
      {
        id: "poster",
        theme: "purple",
        title: "영화 포스터 만들기",
        desc: "좋아하는 영화/드라마 포스터에 신랑신부 얼굴 합성",
        bullets: ["포스터 이미지 업로드", "같이 나온 사진 최대 3장", "커스텀 문구 입력"],
        cta: "시작하기",
        onClick: () => {
          // TODO: 나중에 포스터 기능 페이지로 라우팅
          alert("영화 포스터 기능은 준비 중이에요!");
        },
      },
    ],
    []
  );

  const handleBack = () => navigate(STEPS.result.path);

  return (
    <div className={styles.page}>
      <ProgressBar title="청첩장 만들기" currentStep={8} totalSteps={8} />

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <p className={styles.stepLabel}>STEP 8</p>
            <h1 className={`${styles.title} step-title`}>특별한 기능 선택하기</h1>
            <p className={styles.subTitle}>
              청첩장을 더욱 특별하게 만들 기능을 선택하세요
              <br />
              원하는 만큼 여러 기능을 사용할 수 있습니다!
            </p>
          </header>

          <div className={styles.grid}>
            {options.map((opt) => (
              <article key={opt.id} className={`${styles.optCard} ${styles[opt.theme]}`}>
                <div className={styles.optTop}>
                  <div className={styles.optIcon} aria-hidden="true">
                    {opt.id === "3d" ? "🎨" : "🎬"}
                  </div>
                  <h3 className={styles.optTitle}>{opt.title}</h3>
                </div>

                <p className={styles.optDesc}>{opt.desc}</p>

                <ul className={styles.bullets}>
                  {opt.bullets.map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>

                <button type="button" className={styles.optBtn} onClick={opt.onClick}>
                  {opt.cta}
                </button>
              </article>
            ))}
          </div>

          <section className={styles.tipBox} aria-label="Tip">
            <div className={styles.tipTitle}>💡 Tip</div>
            <ul className={styles.tipList}>
              <li>각 기능은 독립적으로 작동하며, 원하는 순서대로 사용할 수 있습니다</li>
              <li>한 기능을 완료한 후 이 화면으로 돌아와 다른 기능을 선택할 수 있습니다</li>
              <li>모든 기능을 사용할 필요는 없고, 원하는 기능만 선택하세요</li>
              <li>완료된 기능은 언제든지 다시 만들 수 있습니다</li>
            </ul>
          </section>

          <footer className={styles.footer}>
            <button type="button" className={styles.backBtn} onClick={handleBack}>
              ← 결과로 돌아가기
            </button>
          </footer>
        </section>
      </main>
    </div>
  );
}
