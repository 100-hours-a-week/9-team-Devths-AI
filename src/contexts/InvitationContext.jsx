// src/contexts/InvitationContext.jsx
import React, {
  createContext,
  useContext,
  useMemo,
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";
import { useFetch } from "../hooks/useFetch";

const InvitationContext = createContext(null);

export const initialInvitationData = {
  groom: { name: "", fatherName: "", motherName: "" },
  bride: { name: "", fatherName: "", motherName: "" },
  wedding: { hallName: "", address: "", date: "", time: "" },
  extraMessage: "",
  additionalRequest: "",
  tone: null,
  frame: null,

  assets: {
    styleImages: [],
    userImages: [],
  },

  design: {
    result2dImageUrls: [],
  },

  threeD: {
    status: "IDLE",
    invitationId: null,
    assets: null, // 여기에 최종 glb 파일 presigned URL 저장
    result2dImageUrls: [],
    error: null,
    startedAt: null,

    mainImages: [],       // 1장 (필수)
    referenceImages: [],  // 최대 2장 (선택)
  },
};

   function is3DDone(serverStatus) {
      return String(serverStatus || "").toUpperCase() === "COMPLETED";
   }

   function isFailed(serverStatus) {
      const v = String(serverStatus || "").toUpperCase();
      return v.includes("FAILED") || v.includes("ERROR");
   }

function setByPath(obj, path, value) {
  const keys = path.split(".");
  const next = Array.isArray(obj) ? [...obj] : { ...obj };

  let cur = next;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    const prevVal = cur[k];
    cur[k] = Array.isArray(prevVal) ? [...prevVal] : { ...(prevVal ?? {}) };
    cur = cur[k];
  }
  cur[keys[keys.length - 1]] = value;
  return next;
}

function normalizeJobStatus(raw) {
  const v = String(raw || "").toUpperCase();
  if (["PENDING", "QUEUED", "WAITING"].includes(v)) return "PENDING";
  if (["RUNNING", "PROCESSING", "IN_PROGRESS"].includes(v)) return "RUNNING";
  if (["DONE", "COMPLETED", "SUCCESS"].includes(v)) return "DONE";
  if (["FAILED", "ERROR"].includes(v)) return "FAILED";
  return v || "PENDING";
}

export function InvitationProvider({ children }) {

  const [data, setData] = useState(initialInvitationData);

  const { loading, apiFetch, simulate } = useFetch();

  // ✅ 최신 data를 항상 여기 보관 (stale closure 방지)
  const dataRef = useRef(data);
  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  const pollTimerRef = useRef(null);
  const abortRef = useRef(null);

  //이전 폴링의 타이머 정리 (clearTimeout) + 이전 fetch 요청이 진행되는 경우 abort
  const stopThreeDPolling = useCallback((reason = "CANCELED") => {
    
   if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    setData((prev) => {
      const curStatus = prev?.threeD?.status;
      if (curStatus === "DONE" || curStatus === "FAILED") return prev;

      return setByPath(prev, "threeD", {
        ...prev.threeD,
        status: reason,
        error: reason === "CANCELED" ? "사용자에 의해 취소되었습니다." : prev.threeD?.error,
      });
    });
  }, []);

  // ✅ 3D 메인/레퍼런스 setter 추가
  const setThreeDMainImage = useCallback((fileOrNull) => {
    setData((prev) =>
      setByPath(prev, "threeD.mainImages", fileOrNull ? [fileOrNull] : [])
    );
  }, []);

  const setThreeDReferenceImages = useCallback((files) => {
    const safe = Array.isArray(files) ? files.filter(Boolean).slice(0, 2) : [];
    setData((prev) => setByPath(prev, "threeD.referenceImages", safe));
  }, []);

  const setDesignResultImages = useCallback((urls = []) => {
    const safe = Array.isArray(urls) ? urls.filter(Boolean) : [];
    setData((prev) => setByPath(prev, "design.result2dImageUrls", safe));
  }, []);

  const resetDesignStage = useCallback(() => {
    setData((prev) => ({
      ...prev,
      assets: {
        ...prev.assets,
        styleImages: [],
      },
      design: {
        ...initialInvitationData.design,
      },
      threeD: {
        ...initialInvitationData.threeD,
        mainImages: prev.threeD?.mainImages ?? [],
        referenceImages: prev.threeD?.referenceImages ?? [],
      },
    }));
  }, []);

  // ✅ 핵심: startThreeDJob이 "최신 dataRef" 또는 override를 사용하게
  //[3D 모델 생성 API 호출 + 폴링]
  const startThreeDJob = useCallback(
    async ({
      submitUrl = "/api/invitations/3d", //처음 생성 요청 API
      statusUrl = "/api/invitations/3d/status", //상태 확인용 콜백 API
      intervalMs = 5000, //폴링 주기
      buildStatusUrl,
      credentials = "include", //쿠키 사용을 위한 옵션
      mainOverride,        //메인 이미지 1장 -> 단일 파일
      referencesOverride,  //레퍼런스 이미지 최대 2장 -> 파일 배열
    } = {}) => {

      stopThreeDPolling("CANCELED");

      const controller = new AbortController();
      abortRef.current = controller;

      const current = dataRef.current;

      const mainImage = mainOverride ?? current?.threeD?.mainImages?.[0] ?? null;
      const optionalImages = referencesOverride ?? current?.threeD?.referenceImages ?? [];

      //업로드 파일 개수가 맞지 않는 경우 에러 발생
      if (!mainImage) throw new Error("메인 사진 1장은 필수입니다.");
      if (optionalImages.length > 2) throw new Error("레퍼런스 사진은 최대 2장까지 가능합니다.");

      //3D 모델 요청 input 값 업데이트
      setData((prev) => setByPath(prev, "threeD", {
          ...prev.threeD,
          status: "SUBMITTED",
          invitationId: null,
          result: null,
          error: null,
          startedAt: Date.now(),
      }));

      try {
        
         //[3D 모델 생성 API 호출]
         const formData = new FormData();
         formData.append("mainImage", mainImage);
         optionalImages.forEach((f) => formData.append("optionalImages", f));

         //submitUrl: /api/invitations/3d (메소드 호출 시 인자로 넣어줌)
        const submitRes = await apiFetch(submitUrl, {
           method: "POST",
           body: formData,
           credentials: "include",
        });


         if (Array.isArray(submitRes?.result2dImageUrls) && submitRes.result2dImageUrls.length) {
            setData((prev) =>
              setByPath(prev, "threeD", {
                ...prev.threeD,
                result2dImageUrls: submitRes.result2dImageUrls,
                assets: {
                  ...(prev.threeD?.assets ?? {}),
                  model3dUrl: submitRes.result2dImageUrls[0],
                },
              })
            );
         }

         const status = submitRes?.status;
        if(status) console.log('요청 상황: ', status);

        setData((prev) => setByPath(prev, "threeD", {
           ...prev.threeD,
           status: "PENDING",
         }));

         
         //[3D 모델 생성 완료 콜백 API 호출]
         const pollOnce = async () => {

          if (!abortRef.current) return;

          try {
            const statusData = await apiFetch(statusUrl, {
               method: "GET",
               credentials: "include",
               timeoutMs: intervalMs,
            });

            //모델 생성에 성공한 경우
            if (is3DDone(statusData?.status)) {
               console.log('3D 모델 생성 성공!!');
               console.log(statusData);
               const urls = statusData?.result2dImageUrls || [];
               setData((prev) =>
                  setByPath(prev, "threeD", {
                     ...prev.threeD,
                     status: "DONE",
                     invitationId: statusData?.invitationId ?? null,
                     assets: {
                       ...(statusData?.assets ?? prev.threeD?.assets ?? {}),
                       model3dUrl: urls[0] || statusData?.assets?.model3dUrl,
                     },
                     result2dImageUrls: urls,
                     message: statusData?.message ?? null,
                     error: null,
                  })
               );
               if (Array.isArray(urls) && urls.length) {
                 window.location.href = "/3d/sample";
               }
               stopThreeDPolling("DONE");
               return;
            }

            //모델 생성에 실패한 경우
            if (isFailed(statusData?.status)) {
               console.log('3D 모델 생성 실패ㅜㅜ');
               console.log(statusData);
               setData((prev) =>
                  setByPath(prev, "threeD", {
                     ...prev.threeD,
                     status: "FAILED",
                     invitationId: statusData?.invitationId ?? null,
                     assets: statusData?.assets ?? null,
                     message: statusData?.message ?? null,
                     error: statusData?.message || "생성에 실패했습니다.",
                  })
               );
               stopThreeDPolling("FAILED");
               return;
            }

            //모델 생성이 진행 중인 경우
            setData((prev) =>
              setByPath(prev, "threeD", {
                  ...prev.threeD,
                  status: "RUNNING",
                  invitationId: statusData?.invitationId ?? prev.threeD.invitationId,
                  assets: statusData?.assets ?? prev.threeD.assets,
                  message: statusData?.message ?? prev.threeD.message,
                  error: null,
              })
            );

            pollTimerRef.current = setTimeout(pollOnce, intervalMs);

          } catch (e) {
            if (e?.name === "AbortError") return;
            //네트워크 오류인 경우 재시도
            pollTimerRef.current = setTimeout(pollOnce, intervalMs);
          }
        };

        pollTimerRef.current = setTimeout(pollOnce, intervalMs);
      } catch (e) {
        if (e?.name === "AbortError") return;

        setData((prev) =>
          setByPath(prev, "threeD", {
            ...prev.threeD,
            status: "FAILED",
            error: e?.message || "요청 중 오류가 발생했습니다.",
          })
        );
        stopThreeDPolling("FAILED");
      }
    },
    [stopThreeDPolling]
  );

  useEffect(() => {
    return () => stopThreeDPolling("CANCELED");
  }, [stopThreeDPolling]);

  const actions = useMemo(
    () => ({
      updateField: (path, value) => setData((prev) => setByPath(prev, path, value)),
      patchData: (partial) => setData((prev) => ({ ...prev, ...partial })),

      setStyleImages: (files) => setData((prev) => setByPath(prev, "assets.styleImages", files)),
      setUserImages: (files) => setData((prev) => setByPath(prev, "assets.userImages", files)),
      setDesignResultImages,

      // ✅ 3D
      setThreeDMainImage,
      setThreeDReferenceImages,
      resetDesignStage,
      startThreeDJob,
      stopThreeDPolling,
    }),
    [setDesignResultImages, setThreeDMainImage, setThreeDReferenceImages, resetDesignStage, startThreeDJob, stopThreeDPolling]
  );

  return <InvitationContext.Provider value={{ data, ...actions }}>{children}</InvitationContext.Provider>;
}

export function useInvitation() {
  const ctx = useContext(InvitationContext);
  if (!ctx) throw new Error("useInvitation must be used within InvitationProvider");
  return ctx;
}
