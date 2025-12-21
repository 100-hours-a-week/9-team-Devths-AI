import React from "react";
import styles from "./BasicInfoForm.module.css";
import { useInvitation } from "../contexts/InvitationContext";
import { useState } from "react";
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { STEPS } from "../router.jsx";

export function BasicInfoForm() {

   const { data, updateField } = useInvitation();

   const navigate = useNavigate();

   const handleNext = () => {
      console.log(STEPS.image.path,'로 이동');
      navigate(STEPS.image.path);
   };

   const [form, setForm] = useState(() => ({
      groomName: data.groom.name || "",
      brideName: data.bride.name || "",
      groomFatherName: data.groom.fatherName || "",
      groomMotherName: data.groom.motherName || "",
      brideFatherName: data.bride.fatherName || "",
      brideMotherName: data.bride.motherName || "",
      venueName: data.wedding.hallName || "",
      venueAddress: data.wedding.address || "",
      weddingDate: data.wedding.date || "",
      weddingTime: data.wedding.time || "",
      parkingInfo: data.extraMessage || "",       // 주차/추가문구를 extraMessage로 쓰고 싶다면 여기 매핑
      additionalInfo: data.additionalRequest || "",
   }));

   const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

   const isValid = useMemo(() => {
      const required = [
         "groomName",
         "brideName",
         "groomFatherName",
         "groomMotherName",
         "brideFatherName",
         "brideMotherName",
         "venueName",
         "venueAddress",
         "weddingDate",
         "weddingTime",
      ];
      return required.every((k) => String(form[k] ?? "").trim() !== "");
   }, [form]);

   const handleSubmit = (e) => {
      e.preventDefault();

      updateField("groom.name", form.groomName);
      updateField("groom.fatherName", form.groomFatherName);
      updateField("groom.motherName", form.groomMotherName);

      updateField("bride.name", form.brideName);
      updateField("bride.fatherName", form.brideFatherName);
      updateField("bride.motherName", form.brideMotherName);

      updateField("wedding.hallName", form.venueName);
      updateField("wedding.address", form.venueAddress);
      updateField("wedding.date", form.weddingDate);
      updateField("wedding.time", form.weddingTime);

      updateField("extraMessage", form.parkingInfo);
      updateField("additionalRequest", form.additionalInfo);

      handleNext();
   };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <div className={styles.grid2}>
        <div>
          <label className={styles.labelLg}>
            신랑 이름 <span className={styles.req}>*</span>
          </label>
          <input
            type="text"
            name="groomName"
            placeholder="홍길동"
            className={styles.input}
            value={form.groomName}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label className={styles.labelLg}>
            신부 이름 <span className={styles.req}>*</span>
          </label>
          <input
            type="text"
            name="brideName"
            placeholder="김영희"
            className={styles.input}
            value={form.brideName}
            onChange={handleChange}
            required
          />
        </div>
      </div>

      {/* 신랑 부모님 */}
      <div>
        <h3 className={styles.sectionTitle}>
          신랑 부모님 성함 <span className={styles.req}>*</span>
        </h3>

        <div className={styles.grid2}>
          <div>
            <label className={styles.labelBase}>
              아버지 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="groomFatherName"
              placeholder="홍아무개"
              className={styles.input}
              value={form.groomFatherName}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <label className={styles.labelBase}>
              어머니 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="groomMotherName"
              placeholder="김아무개"
              className={styles.input}
              value={form.groomMotherName}
              onChange={handleChange}
              required
            />
          </div>
        </div>
      </div>

      {/* 신부 부모님 */}
      <div>
        <h3 className={styles.sectionTitle}>
          신부 부모님 성함 <span className={styles.req}>*</span>
        </h3>

        <div className={styles.grid2}>
          <div>
            <label className={styles.labelBase}>
              아버지 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="brideFatherName"
              placeholder="김아무개"
              className={styles.input}
              value={form.brideFatherName}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <label className={styles.labelBase}>
              어머니 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="brideMotherName"
              placeholder="이아무개"
              className={styles.input}
              value={form.brideMotherName}
              onChange={handleChange}
              required
            />
          </div>
        </div>
      </div>

      {/* 예식장 정보 */}
      <div>
        <h3 className={styles.sectionTitle}>
          예식장 정보 <span className={styles.req}>*</span>
        </h3>

        <div className={styles.stack6}>
          <div>
            <label className={styles.labelBase}>
              예식장 이름 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="venueName"
              placeholder="○○웨딩홀"
              className={styles.input}
              value={form.venueName}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <label className={styles.labelBase}>
              예식장 주소 <span className={styles.req}>*</span>
            </label>
            <input
              type="text"
              name="venueAddress"
              placeholder="서울특별시 강남구..."
              className={styles.input}
              value={form.venueAddress}
              onChange={handleChange}
              required
            />
          </div>
        </div>
      </div>

      {/* 예식 일시 */}
      <div>
        <h3 className={styles.sectionTitle}>
          예식 일시 <span className={styles.req}>*</span>
        </h3>

        <div className={styles.grid2}>
          <div>
            <label className={styles.labelBase}>
              예식 날짜 <span className={styles.req}>*</span>
            </label>
            <input
              type="date"
              name="weddingDate"
              className={styles.input}
              value={form.weddingDate}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <label className={styles.labelBase}>
              예식 시간 <span className={styles.req}>*</span>
            </label>
            <input
              type="time"
              name="weddingTime"
              className={styles.input}
              value={form.weddingTime}
              onChange={handleChange}
              required
            />
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className={styles.divider}>
        <h3 className={styles.dividerTitle}>
          추가 정보 <span className={styles.dividerHint}>(선택)</span>
        </h3>
        <p className={styles.dividerDesc}>선택사항이며 입력하지 않아도 됩니다</p>
      </div>

      {/* 주차/추가문구 */}
      <div>
        <label className={styles.labelBase}>주차 정보</label>
        <textarea
          name="parkingInfo"
          placeholder="예) 건물 지하 1~3층 주차 가능 (3시간 무료)"
          rows={3}
          className={styles.textarea}
          value={form.parkingInfo}
          onChange={handleChange}
        />
      </div>

      {/* 기타 요청사항 */}
      <div>
        <label className={styles.labelBase}>추가 정보</label>
        <textarea
          name="additionalInfo"
          placeholder="꼭 포함하고 싶은 문구나 기타 요청 사항을 자유롭게 작성해주세요"
          rows={4}
          className={styles.textarea}
          value={form.additionalInfo}
          onChange={handleChange}
        />
      </div>

      <div className={styles.submitRow}>
        <button type="submit" disabled={!isValid} className={styles.primaryBtn}>
          다음
        </button>
      </div>
    </form>
  );
}
