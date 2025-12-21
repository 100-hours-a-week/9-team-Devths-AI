import React from "react";
import styles from "./GuideTextSection.module.css";

export default function GuideBox({
  title,
  items = [],
  icon = "📷",
  tone = "warm", // warm | neutral
}) {
  return (
    <section className={`${styles.box} ${tone === "warm" ? styles.warm : styles.neutral}`}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          {icon}
        </span>
        <h3 className={styles.title}>{title}</h3>
      </div>

      <ul className={styles.list}>
        {items.map((text, idx) => (
          <li key={`${idx}-${text}`} className={styles.item}>
            <span className={styles.bullet} aria-hidden="true" />
            <span className={styles.text}>{text}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
