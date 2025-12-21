import { SamplePage } from "./pages/3D/SamplePage";
import { ThreeDInvitationResultPage } from "./pages/3D/ThreeDInvitationResultPage";
import { ThreeDInvitePage } from "./pages/3D/ThreeDInvitePage";
import { BasicInfoPage } from "./pages/BasicInfoPage";
import { CopySelectPage } from "./pages/CopySelectPage";
import { DesignDetailPage } from "./pages/DesignDetailPage";
import { ImageUploadPage } from "./pages/ImageUploadPage";
import { LandingPage } from "./pages/LandingPage";
import { OptionSelectPage } from "./pages/OptionSelectPage";
import { ResultPage } from "./pages/ResultPage";
import { ToneSelectPage } from "./pages/ToneSelectPage";

export const STEPS = {
  landing: { path: "/", label: "홈" },
  basic: { path: "/info", label: "기본정보" },                
  tone: { path: "/tone", label: "초대 멘트 톤 선택" },              
  image: { path: "/image", label: "내 이미지" },           
  design: { path: "/design", label: "디자인" },             
  result: { path: "/result", label: "결과" },  
  options: { path: "/options", label: "특별한 기능" }, 
  threed: { path: "/3d", label: "3D 청첩장 만들기" },
  threedResult: { path: "/3d/result", label: "3D 청첩장 결과" },   
  sample: { path: "/3d/sample", label: "3D 청첩장 샘플" },       
};

export const routes = [
  { path: STEPS.landing.path, element: <LandingPage /> },
  { path: STEPS.basic.path, element: <BasicInfoPage /> },
  { path: STEPS.tone.path, element: <ToneSelectPage /> },
  { path: STEPS.image.path, element: <ImageUploadPage /> },
  { path: STEPS.design.path, element: <DesignDetailPage /> },
  { path: STEPS.result.path, element: <ResultPage /> },
  { path: STEPS.options.path, element: <OptionSelectPage /> },
  { path: STEPS.threed.path, element: <ThreeDInvitePage /> },
  { path: STEPS.threedResult.path, element: <ThreeDInvitationResultPage /> },
  { path: STEPS.sample.path, element: <SamplePage /> },
];