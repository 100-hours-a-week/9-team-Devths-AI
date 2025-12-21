// src/pages/ThreeDInvitationResultPage.jsx
import React, { useMemo, useState } from "react";
import styles from "./SamplePage.module.css";
import ModelViewer from "../../components/ThreeDModel";
import { useInvitation } from "../../contexts/InvitationContext";


function formatDateTime(date, time) {
  if (!date && !time) return "";
  const [y, m, d] = (date || "").split("-").map(Number);
  const [hh, mm] = (time || "").split(":").map(Number);

  if (!y || !m || !d) return [date, time].filter(Boolean).join(" ");

  const dt = new Date(y, (m || 1) - 1, d || 1, hh || 0, mm || 0);
  const weekday = ["일", "월", "화", "수", "목", "금", "토"][dt.getDay()];
  const ap = (hh ?? 0) < 12 ? "오전" : "오후";
  const hour12 = ((hh ?? 0) % 12) || 12;
  const minute = String(mm ?? 0).padStart(2, "0");
  return `${m}월 ${d}일 (${weekday}) ${ap} ${hour12}:${minute}`;
}

function calcDDay(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = dateStr.split("-").map(Number);
  if (!y || !m || !d) return null;

  const target = new Date(y, m - 1, d, 0, 0, 0);
  const now = new Date();
  const diff = target.getTime() - now.getTime();
  const days = Math.ceil(diff / (1000 * 60 * 60 * 24));
  return days;
}

export function SamplePage() {
  const { data } = useInvitation();
  const [copied, setCopied] = useState(false);

  const groomName = data?.groom?.name || "라이언";
  const brideName = data?.bride?.name || "춘식이";

  const hallName = data?.wedding?.hallName || "더 바실리움";
  const address = data?.wedding?.address || "경기 성남시 분당구 양현로 322 1층";
  const dateTimeText = useMemo(
    () => formatDateTime(data?.wedding?.date, data?.wedding?.time) || "2026년 1월 1일 (목) 오후 1:00",
    [data?.wedding?.date, data?.wedding?.time]
  );

  const dday = useMemo(() => calcDDay(data?.wedding?.date), [data?.wedding?.date]);

  // ✅ 실제 glb url (없으면 로컬 테스트)
  const model3dUrl = data?.threeD?.result2dImageUrls?.[0]
    || data?.threeD?.assets?.model3dUrl
    || "/images/model1.glb";

  const inviteTitle = data?.extraMessage?.trim()
    ? "초대합니다"
    : "Save the Date";

  const extraMessage =
    data?.extraMessage ||
    "소중한 분들을 모시고 저희의 새로운 시작을 함께하고자 합니다.\n따뜻한 마음으로 축복해 주세요.";

  // (임시) 연락처/계좌 정보 - 나중에 data에 붙이면 여기만 교체하면 됨
  const contacts = {
    groom: { label: `${groomName}에게 연락`, tel: "010-0000-0000" },
    bride: { label: `${brideName}에게 연락`, tel: "010-0000-0000" },
  };

  const accounts = [
    { who: groomName, bank: "OO은행", number: "123-456-789012", name: groomName },
    { who: brideName, bank: "OO은행", number: "987-654-321098", name: brideName },
  ];

  // (임시) 갤러리: 나중에 assets 붙이면 교체
  const gallery = [
    "/images/sample1.jpg",
    "/images/sample2.jpg",
    "/images/sample3.jpg",
  ];

  const handleCopy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      alert("복사에 실패했어요. 직접 선택해서 복사해주세요.");
    }
  };

  const mapQuery = encodeURIComponent(`${hallName} ${address}`);
  const naverMapUrl = `https://map.naver.com/v5/search/${mapQuery}`;

  return (
    <div className={styles.page}>
      <div className={styles.bg} aria-hidden="true" />

      <main className={styles.container}>
        {/* 상단 타이틀 */}
        <header className={styles.header}>
          <div className={styles.chip}>3D Mobile Invitation</div>

          <h1 className={styles.title}>
            {groomName} <span className={styles.amp}>&</span> {brideName}
          </h1>

          <p className={styles.subTitle}>{dateTimeText}</p>

          {dday !== null && (
            <div className={styles.ddayPill}>
              {dday > 0 && <>D-{dday}</>}
              {dday === 0 && <>D-DAY ✨</>}
              {dday < 0 && <>D+{Math.abs(dday)}</>}
            </div>
          )}
        </header>

        {/* 3D 모델 영역 */}
        <section className={styles.modelSection}>
          <div className={styles.modelHeader}>
            <div className={styles.modelLabel}>3D 초대장</div>
            <div className={styles.modelHint}>드래그해서 돌려보기</div>
          </div>

          <ModelViewer modelUrl={model3dUrl} />

          {data?.threeD?.message && (
            <div className={styles.serverMessage}>{data.threeD.message}</div>
          )}
        </section>

        {/* 예식 정보 */}
        <section className={styles.card}>
          <div className={styles.sectionTitle}>예식 정보</div>

          <div className={styles.row}>
            <span className={styles.key}>일시</span>
            <span className={styles.value}>{dateTimeText}</span>
          </div>

          <div className={styles.row}>
            <span className={styles.key}>장소</span>
            <span className={styles.value}>{hallName}</span>
          </div>

          <div className={styles.row}>
            <span className={styles.key}>주소</span>
            <span className={styles.value}>{address}</span>
          </div>

          <div className={styles.actions}>
            <button type="button" className={styles.secondaryBtn} onClick={() => handleCopy(address)}>
              {copied ? "복사 완료!" : "주소 복사"}
            </button>

            <a className={styles.primaryBtn} href={naverMapUrl} target="_blank" rel="noreferrer">
              지도 열기
            </a>
          </div>
        </section>

        {/* 초대 문구 */}
        <section className={styles.card}>
          <div className={styles.sectionTitle}>{inviteTitle}</div>
          <p className={styles.message}>{extraMessage}</p>
        </section>

        {/* 연락하기 */}
        <section className={styles.card}>
          <div className={styles.sectionTitle}>연락하기</div>
          <div className={styles.contactGrid}>
            <a className={styles.contactBtn} href={`tel:${contacts.groom.tel}`}>
              <div className={styles.contactTop}>📞</div>
              <div className={styles.contactLabel}>{contacts.groom.label}</div>
              <div className={styles.contactSub}>{contacts.groom.tel}</div>
            </a>

            <a className={styles.contactBtn} href={`tel:${contacts.bride.tel}`}>
              <div className={styles.contactTop}>💌</div>
              <div className={styles.contactLabel}>{contacts.bride.label}</div>
              <div className={styles.contactSub}>{contacts.bride.tel}</div>
            </a>
          </div>
        </section>

        {/* 계좌 */}
        <section className={styles.card}>
          <div className={styles.sectionTitle}>마음 전하실 곳</div>

          <div className={styles.accountList}>
            {accounts.map((a, idx) => (
              <div key={idx} className={styles.accountItem}>
                <div className={styles.accountWho}>{a.who}</div>
                <div className={styles.accountInfo}>
                  <span className={styles.accountBank}>{a.bank}</span>
                  <span className={styles.accountNumber}>{a.number}</span>
                </div>
                <button
                  type="button"
                  className={styles.accountCopy}
                  onClick={() => handleCopy(`${a.bank} ${a.number} (${a.name})`)}
                >
                  복사
                </button>
              </div>
            ))}
          </div>

          <div className={styles.hintText}>
            * 계좌 정보는 임시 값이며, 나중에 실제 데이터로 연결하면 돼요
          </div>
        </section>

        {/* 갤러리 */}
        <section className={styles.card}>
          <div className={styles.sectionTitle}>갤러리</div>

          <div className={styles.gallery}>
            {gallery.map((src, idx) => (
              <div key={idx} className={styles.galleryItem}>
                {/* 이미지 파일 없으면 깨질 수 있으니, 실제 연결 전까진 placeholder로 보이게 */}
                <img
                  src={src}
                  alt={`gallery-${idx + 1}`}
                  className={styles.galleryImg}
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                    e.currentTarget.parentElement.classList.add(styles.galleryFallback);
                  }}
                />
              </div>
            ))}
          </div>

          <div className={styles.hintText}>
            * 지금은 임시 경로예요. 나중에 S3/서버 URL로 교체하면 돼요
          </div>
        </section>

        {/* 하단 고정 버튼 */}
        <footer className={styles.footer}>
          <button type="button" className={styles.footerBtn}>
            참석 여부 전달하기 (임시)
          </button>
        </footer>
      </main>
    </div>
  );
}
