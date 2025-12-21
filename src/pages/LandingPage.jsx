import React, { useState } from "react";
import styles from "./LandingPage.module.css";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";
import { useFetch } from "../hooks/useFetch.jsx";

export function LandingPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  const { apiFetch, simulate } = useFetch();


  const routeTo3d = () => {
      navigate(STEPS.threed.path);
  }

  const handleStart = async () => {
    if (loading) return;
    setLoading(true);
    setErrMsg("");

    try {
      //[API 연동] 백엔드 연동 후 수정
      const res = await apiFetch("/api/invitations/init", {
        method: "GET",
        credentials: "include",
      });

      console.log('init 요청 응답: ', res);

      // if (!res.ok) {
      //   let msg = `초기화 요청 실패 (${res.status})`;
      //   try {
      //     const j = await res.json();
      //     msg = j?.message || msg;
      //   } catch {}
      //   throw new Error(msg);
      // }

      // 서버가 쿠키를 Set-Cookie로 내려주면, 위 credentials: "include" 덕분에 브라우저에 저장됨
      // 필요하면 응답 바디를 읽어도 됨
      // const result = await res.json().catch(() => null);
      // console.log("[init result]", result);

      //수정 필요!!!!!!!!!!!
      navigate(STEPS.threed.path);
    } catch (e) {
      setErrMsg(e?.message || "요청 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handle3d = async () => {
    if (loading) return;
    setLoading(true);
    setErrMsg("");

    try {
      //[API 연동] 백엔드 연동 후 수정
      const res = await apiFetch("/api/invitations/init", {
        method: "GET",
        credentials: "omit",
      });

      console.log('init 요청 응답: ', res);

      // if (!res.ok) {
      //   let msg = `초기화 요청 실패 (${res.status})`;
      //   try {
      //     const j = await res.json();
      //     msg = j?.message || msg;
      //   } catch {}
      //   throw new Error(msg);
      // }

      // 서버가 쿠키를 Set-Cookie로 내려주면, 위 credentials: "include" 덕분에 브라우저에 저장됨
      // 필요하면 응답 바디를 읽어도 됨
      // const result = await res.json().catch(() => null);
      // console.log("[init result]", result);

      navigate(STEPS.threed.path);
    } catch (e) {
      setErrMsg(e?.message || "요청 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.bg} aria-hidden="true">
        <span className={styles.orb1} />
        <span className={styles.orb2} />
        <span className={styles.orb3} />
      </div>

      <main className={styles.main}>
        <section className={styles.heroCard}>
          <div className={styles.badge}>AI Wedding Invitation</div>

          <h1 className={styles.title}>
            따뜻한 감성의 <span className={styles.highlight}>AI 청첩장</span>을
            <br />
            만들어보세요
          </h1>

          <p className={styles.subTitle}>
            신랑·신부 정보 입력 → 사진 업로드 → 디자인 선택까지
            <br />
            완성된 청첩장 이미지를 바로 다운로드할 수 있어요.
          </p>

          <div className={styles.ctaRow}>
            <button
              type="button"
              className={styles.primaryBtn}
              onClick={handleStart}
              disabled={loading}
            >
              {loading ? (
                <span className={styles.btnInner}>
                  <span className={styles.spinner} aria-hidden="true" />
                  준비 중...
                </span>
              ) : (
                "시작하기"
              )}
            </button>

            <button
              type="button"
              className={styles.ghostBtn}
              onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" })}
            >
              기능 미리보기
            </button>
          </div>

          {errMsg && <div className={styles.errorBox}>{errMsg}</div>}

          <div className={styles.metaRow}>
            <div className={styles.metaItem}>
              <span className={styles.metaDot} />
              임시 토큰 발급 후 진행
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaDot} />
              결과물 다운로드 지원
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaDot} />
              웨딩 감성 + 글래스 UI
            </div>
          </div>
        </section>

        <section className={styles.features}>
          <article className={styles.featureCard}>
            <div className={styles.featureIcon} aria-hidden="true">🧾</div>
            <div className={styles.featureTitle}>정보 입력</div>
            <div className={styles.featureDesc}>예식장, 날짜/시간, 추가 안내까지 한 번에</div>
          </article>

          <article className={styles.featureCard}>
            <div className={styles.featureIcon} aria-hidden="true">🖼️</div>
            <div className={styles.featureTitle}>사진 업로드</div>
            <div className={styles.featureDesc}>메인/스타일 이미지로 원하는 무드 전달</div>
          </article>

          <article className={styles.featureCard}>
            <div className={styles.featureIcon} aria-hidden="true">✨</div>
            <div className={styles.featureTitle}>디자인 선택</div>
            <div className={styles.featureDesc}>톤/프레임 조합으로 완성도 높은 결과</div>
          </article>
        </section>
      </main>
    </div>
  );
}
