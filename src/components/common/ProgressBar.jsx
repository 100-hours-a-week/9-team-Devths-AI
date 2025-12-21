import React from "react";
import styles from "./ProgressBar.module.css";

export default function ProgressBar({ title, currentStep, totalSteps }) {
  const percent = Math.round((currentStep / totalSteps) * 100);

  return (
    <header className={styles.header}>
      <div className={styles.topRow}>
        <h2 className={styles.title}>{title}</h2>
        <div className={styles.stepText}>
          {currentStep}/{totalSteps}
        </div>
      </div>

      <div className={styles.track} aria-label={`progress ${percent}%`}>
        <div className={styles.fill} style={{ width: `${percent}%` }} />
      </div>
    </header>
  );
}