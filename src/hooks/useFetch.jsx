import { useState, useEffect, useRef, useCallback } from "react";

//const URL_BASE = "https://api.dualstack.ktb-hakcathon-alb-422139206.ap-northeast-2.elb.amazonaws.com"; //서버 URL로 변경 필요
const URL_BASE = "https://api.marry-invitation.com";

export function getAccessToken() {
  return localStorage.getItem('accessToken');
}

export function setAccessToken(token) {
  localStorage.setItem('accessToken', token);
}

export function getRefreshToken() {
  return localStorage.getItem('refreshToken');
}

export function setRefreshToken(token) {
  localStorage.setItem('refreshToken', token);
}

export function setAuthUser({userId, profileImg}) {
  localStorage.setItem('userId', userId);
  localStorage.setItem('profileImg', profileImg);
}

export function getAuthUser() {
  const userId = localStorage.getItem('userId');
  const profileImg = localStorage.getItem('profileImg');
  return {
    userId: userId,
    profileImg: profileImg
  }
}

export function deleteAuthUser() {
  localStorage.removeItem('userId');
  localStorage.removeItem('profileImg');
}

export function useFetch() {

   const [ state, setState ] = useState({ data: null, error: null, loading: false });
   const [ loading, setLoading ] = useState(false);

   const controllerRef = useRef(null);
   const timeoutIdRef = useRef(null);

   const abort = useCallback(() => {
      controllerRef.current?.abort?.();
   }, []);

   const simulate = useCallback(async(ms = 5000) => {
      setState({loading: true});
      try {
         await new Promise((r) => setTimeout(r, ms));
      } finally {
         setState({loading: false});
      }
   }, []);

   const apiFetch = useCallback(async (url, options = {}) => {

      setState((s) => ({...s, loading: true, error: null }));

      options.header = options.header || {};

      if (options.withAuth) {
         const accessToken = getAccessToken();
         if (accessToken) options.header['Authorization'] = `Bearer ${accessToken}`;
      }

      if (options.body && !(options.body instanceof FormData)) {
         options.header['Content-Type'] = 'application/json';
         options.body = JSON.stringify(options.body);
      }

      try {
         const controller = new AbortController();
         controllerRef.current = controller;

         if (timeoutIdRef.current) clearTimeout(timeoutIdRef.current);
         const timeoutMs = typeof options.timeoutMs === "number" ? options.timeoutMs : 600000; // 기본 10분
         timeoutIdRef.current = setTimeout(() => controller.abort(), timeoutMs);

         const response = await  fetch(`${URL_BASE}${url}`, {
            method: options.method,
            credentials: options.credentials,
            headers: options.header,
            body: options.body,
            signal: controller.signal
         });

         let data = null;
         try {
            data = await response.json();
         }
         catch(error) {
            throw new Error(`HTTP 응답 파싱 오류: ${error?.message || error}`);
         }

         if(!response.ok) {
            const message = data?.message || `HTTP 요청 처리 오류: 상태코드 ${response.status}`;
            const error = new Error(message);
            error.status = response.status;
            error.payload = data;
            throw error;
         } 

         setState({data: data, loading: false, error: null});
         return data;

      } catch(error) {
         if (error.name === "AbortError") return;
         setState({ data: null, loading: false, error });
         throw error;
      } finally {
         if (timeoutIdRef.current) {
            clearTimeout(timeoutIdRef.current);
            timeoutIdRef.current = null;
         }
      }
   }, []);

   useEffect(() => {
      return () => {
         controllerRef.current?.abort?.();
         if (timeoutIdRef.current) clearTimeout(timeoutIdRef.current);
      };
   }, []);

   return { ...state, apiFetch, abort, simulate };

}
