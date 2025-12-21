import styles from "./ImageUploadPage.module.css";
import ProgressBar from "../components/common/ProgressBar";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";
import GuideBox from "../components/common/GuideTextSection.jsx";
import SingleImageUploader from "../components/common/SingleImageUploader.jsx";
import { useEffect } from "react";
import { useInvitation } from "../contexts/InvitationContext.jsx";

export function ImageUploadPage() {

   const { data, setUserImages } = useInvitation();

   useEffect(() => {
      //전역 context 확인용
      console.log("[InvitationContext data]", data);
   }, []);

   const navigate = useNavigate();

   const handleNext = () => {
      console.log(STEPS.tone.path,'로 이동');
      navigate(STEPS.tone.path);
   };

   const handleBack = () => {
      console.log(STEPS.basic.path,'로 이동');
      navigate(STEPS.basic.path);
   }

   const items = [
      "웨딩 촬영 사진, 일상 사진, 연애 사진 등 다양한 사진을 올려주세요",
      "고화질 이미지를 권장합니다 (최소 1000x1000 픽셀)",
      "가로/세로 비율이 다양해도 괜찮습니다",
      "AI가 자동으로 최적의 레이아웃을 생성합니다",
      "업로드 순서대로 우선순위가 부여됩니다",
   ];

   return (
      <div className={styles.page}>
         <ProgressBar title="청첩장 만들기" currentStep={2} totalSteps={7} />
   
         <main className={styles.main}>
            <section className={styles.card}>
               <header className={styles.header}>
                  <p className={styles.stepLabel}>STEP 2</p>
                  <h1 className={`${styles.pageTitle} step-title`}>청첩장 메인 사진 업로드</h1>
               </header>

               <div className={styles.uploaderSection}>
                  <SingleImageUploader
                    value={data.assets.userImages?.[0] ?? null}
                    onChange={(fileOrNull) => {
                      setUserImages(fileOrNull ? [fileOrNull] : []);
                    }}
                  />
               </div>

               <div className={styles.guideSection}>
                  <GuideBox title="사진 업로드 가이드" items={items} icon="📸" />
               </div>
            </section>
            
            <button type="button" className={styles.nextButton} onClick={handleNext}>
              다음 단계로 이동
            </button>
         </main>
      </div>
   );
}
