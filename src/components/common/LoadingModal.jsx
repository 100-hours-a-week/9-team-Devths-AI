import React, { useEffect, useMemo } from "react";
import { createPortal } from "react-dom";
import styles from "./LoadingModal.module.css";

function getPortalRoot() {
  let root = document.getElementById("modal-root");
  if (!root) {
    root = document.createElement("div");
    root.id = "modal-root";
    document.body.appendChild(root);
  }
  return root;
}

export default function LoadingModal({
  open,
  title = "청첩장을 만들고 있어요",
  message = "이미지를 분석하고 디자인을 생성하는 중입니다. 잠시만 기다려주세요.",
  hint = "완료까지 수 초~수십 초 걸릴 수 있어요",
}) {
  const portalRoot = useMemo(() => (typeof document !== "undefined" ? getPortalRoot() : null), []);

  // ✅ 모달 열릴 때 스크롤 잠금
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open || !portalRoot) return null;

  return createPortal(
    <div className={styles.backdrop} role="alert" aria-live="assertive" aria-busy="true">
      <div className={styles.glow} aria-hidden="true" />
      <div className={styles.modal} role="dialog" aria-modal="true" aria-label={title}>
        <div className={styles.header}>
          <div className={styles.badge} aria-hidden="true">
            AI Design
          </div>
          <h2 className={styles.title}>{title}</h2>
          <p className={styles.message}>{message}</p>
        </div>

        <div className={styles.center}>
          <div className={styles.ring} aria-hidden="true">
            <span className={styles.r1} />
            <span className={styles.r2} />
            <span className={styles.r3} />
          </div>

          <div className={styles.progressLine} aria-hidden="true">
            <span className={styles.progressBar} />
          </div>

          <p className={styles.hint}>{hint}</p>
        </div>

        <div className={styles.footer}>
          <div className={styles.pulseDot} aria-hidden="true" />
          <span className={styles.footerText}>작업 중에는 페이지를 닫지 말아주세요</span>
        </div>
      </div>
    </div>,
    portalRoot
  );
}
