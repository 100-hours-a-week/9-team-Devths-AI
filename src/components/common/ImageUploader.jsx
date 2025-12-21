import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import styles from "./ImageUploader.module.css";

export default function ImageUploader({
  maxCount = 10,
  value,            // ✅ File[]
  onChange,         // ✅ (nextFiles: File[]) => void
  accept = "image/*",
}) {
  const inputId = useId();
  const inputRef = useRef(null);
  const listRef = useRef(null);

  // uncontrolled 지원
  const [innerFiles, setInnerFiles] = useState([]); // File[]
  const files = value ?? innerFiles;

  const setFiles = (nextFiles) => {
    if (onChange) onChange(nextFiles);
    else setInnerFiles(nextFiles);
  };

  const remaining = useMemo(
    () => Math.max(0, maxCount - files.length),
    [maxCount, files.length]
  );

  const openPicker = () => {
    if (!remaining) return;
    inputRef.current?.click();
  };

  const handlePick = (e) => {
    const picked = Array.from(e.target.files || []);
    if (!picked.length) return;

    const allowed = picked.slice(0, remaining);
    setFiles([...files, ...allowed]);

    // 같은 파일 다시 선택 가능하게
    e.target.value = "";
  };

  const removeAt = (idx) => {
    const next = files.filter((_, i) => i !== idx);
    setFiles(next);
  };

  // preview는 내부에서만 만들고, 변경/언마운트 시 revoke
  const [previews, setPreviews] = useState([]); // { id, url }[]
  useEffect(() => {
    const nextPreviews = files.map((file) => {
      const id = `${file.name}-${file.size}-${file.lastModified}`;
      const url = URL.createObjectURL(file);
      return { id, url };
    });

    setPreviews(nextPreviews);

    return () => {
      nextPreviews.forEach((p) => URL.revokeObjectURL(p.url));
    };
  }, [files]);

  // 새로 추가되면 오른쪽 끝으로 스크롤
  useEffect(() => {
    if (!listRef.current) return;
    listRef.current.scrollTo({
      left: listRef.current.scrollWidth,
      behavior: "smooth",
    });
  }, [files.length]);

  return (
    <div className={styles.wrap}>
      {/* 왼쪽: 이미지 추가 카드(고정) */}
      <button
        type="button"
        className={styles.addCard}
        onClick={openPicker}
        disabled={!remaining}
      >
        <div className={styles.plus}>+</div>
        <div className={styles.addText}>이미지 추가</div>
        <div className={styles.count}>
          {files.length}/{maxCount}
        </div>
        {!remaining && <div className={styles.hint}>최대 {maxCount}장까지</div>}
      </button>

      {/* 오른쪽: 썸네일 리스트(가로 스크롤) */}
      <div className={styles.list} ref={listRef}>
        {previews.map((p, idx) => (
          <div key={p.id} className={styles.thumbItem}>
            <div className={styles.thumb}>
              <img src={p.url} alt={`upload-${idx + 1}`} />
            </div>

            <div className={styles.thumbFooter}>
              <span className={styles.thumbIndex}>{idx + 1}</span>
              <button
                type="button"
                className={styles.removeBtn}
                onClick={() => removeAt(idx)}
                aria-label="remove"
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>

      <input
        id={inputId}
        ref={inputRef}
        className={styles.hiddenInput}
        type="file"
        accept={accept}
        multiple
        onChange={handlePick}
      />
    </div>
  );
}
