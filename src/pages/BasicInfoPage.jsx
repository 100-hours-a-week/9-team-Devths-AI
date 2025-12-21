import styles from "./BasicInfoPage.module.css";
import { BasicInfoForm } from "../components/BasicInfoForm";
import ProgressBar from "../components/common/ProgressBar";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";

export function BasicInfoPage() {

   const navigate = useNavigate();

   const handleNext = () => {
      console.log(STEPS.image.path,'로 이동');
      navigate(STEPS.image.path);
   };

   return (
      <div className={styles.page}>
      <ProgressBar title="청첩장 만들기" currentStep={1} totalSteps={7} />

      <main className={styles.main}>
        <section className={styles.card}>
          <header className={styles.header}>
            <p className={styles.stepLabel}>STEP 1</p>
            <h1 className={`${styles.pageTitle} step-title`}>기본 정보 입력</h1>
          </header>
          <BasicInfoForm />
        </section>
      </main>
    </div>
   );
}
