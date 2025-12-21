import React, { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, Html, useGLTF, Bounds } from "@react-three/drei";
import { useNavigate } from "react-router-dom";
import styles from "./ThreeDInvitationResultPage.module.css";
import { useInvitation } from "../../contexts/InvitationContext.jsx"; // 경로 맞게 조정

function Loader() {
  return (
    <Html center>
      <div className={styles.loader}>
        <div className={styles.spinner} />
        <div className={styles.loaderText}>3D 모델 불러오는 중…</div>
        <div className={styles.loaderSub}>네트워크/용량에 따라 시간이 걸릴 수 있어요</div>
      </div>
    </Html>
  );
}

function GLBModel({ url }) {
  // ✅ 여기서 url로 GET 요청이 나감
  const gltf = useGLTF(url);
  return <primitive object={gltf.scene} />;
}

export function ThreeDInvitationResultPage() {
  const navigate = useNavigate();
  const { data } = useInvitation();

  const threeD = data?.threeD || {};
  const model3dUrl = threeD?.assets?.model3dUrl || "";
  const message = threeD?.message || "";

  // (선택) preload: 진입 직후 미리 로딩 시작
  useMemo(() => {
    if (model3dUrl) useGLTF.preload(model3dUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model3dUrl]);

  const canRender = !!model3dUrl && threeD.status === "DONE";

  const handleDownload = () => {
    if (!model3dUrl) return;
    // 단순 다운로드(브라우저가 직접 받도록)
    const a = document.createElement("a");
    a.href = model3dUrl;
    a.download = "invitation_3d.glb"; // 서버가 Content-Disposition 주면 이 이름은 무시될 수 있음
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleCopyLink = async () => {
    if (!model3dUrl) return;
    try {
      await navigator.clipboard.writeText(model3dUrl);
      alert("링크를 복사했어요!");
    } catch {
      alert("복사에 실패했어요. 주소를 직접 복사해주세요.");
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.bgOrbs} aria-hidden="true">
        <span className={styles.orb1} />
        <span className={styles.orb2} />
        <span className={styles.orb3} />
      </div>

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <div className={styles.badge}>3D 결과</div>
            <h1 className={styles.title}>3D 청첩장이 완성되었어요</h1>
            {message ? <p className={styles.subTitle}>{message}</p> : <p className={styles.subTitle}>아래에서 3D 모델을 확인해보세요</p>}
          </header>

          {!canRender ? (
            <div className={styles.empty}>
              <div className={styles.emptyTitle}>아직 불러올 수 있는 3D 모델이 없어요</div>
              <div className={styles.emptyDesc}>
                - status: <b>{threeD.status}</b>
                <br />
                - model3dUrl: <b>{model3dUrl ? "있음" : "없음"}</b>
              </div>

              <div className={styles.actions}>
                <button className={styles.secondaryBtn} onClick={() => navigate(-1)}>
                  이전으로
                </button>
                <button className={styles.primaryBtn} onClick={() => navigate("/option")}>
                  옵션 선택으로
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className={styles.viewerWrap}>
                <Canvas
                  className={styles.canvas}
                  camera={{ position: [0, 1.2, 3.2], fov: 45 }}
                  dpr={[1, 2]}
                >
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[3, 5, 2]} intensity={1.0} />

                  <Suspense fallback={<Loader />}>
                    {/* ✅ Bounds: 모델 크기/위치 자동 맞춤 */}
                    <Bounds fit clip observe margin={1.2}>
                      <GLBModel url={model3dUrl} />
                    </Bounds>

                    <Environment preset="city" />
                    <OrbitControls
                      makeDefault
                      enableDamping
                      dampingFactor={0.08}
                      rotateSpeed={0.8}
                      minDistance={1.2}
                      maxDistance={8}
                    />
                  </Suspense>
                </Canvas>

                <div className={styles.hint}>
                  마우스로 드래그: 회전 · 휠: 확대/축소 · 우클릭 드래그: 이동
                </div>
              </div>

              <div className={styles.meta}>
                <div className={styles.metaRow}>
                  <span className={styles.metaLabel}>invitationId</span>
                  <span className={styles.metaValue}>{threeD.invitationId || "-"}</span>
                </div>
                <div className={styles.metaRow}>
                  <span className={styles.metaLabel}>model3dUrl</span>
                  <span className={styles.metaValueEllipsis} title={model3dUrl}>
                    {model3dUrl}
                  </span>
                </div>
              </div>

              <div className={styles.actions}>
                <button className={styles.secondaryBtn} onClick={() => navigate("/option")}>
                  다른 옵션 만들기
                </button>
                <button className={styles.secondaryBtn} onClick={handleCopyLink}>
                  링크 복사
                </button>
                <button className={styles.primaryBtn} onClick={handleDownload}>
                  GLB 다운로드
                </button>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
