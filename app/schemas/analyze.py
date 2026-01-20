from pydantic import BaseModel, Field
from typing import List


class AnalyzeRequest(BaseModel):
    """분석 및 매칭도 요청 (API 3)"""
    resume_id: str = Field(..., description="이력서 ID")
    posting_id: str = Field(..., description="채용공고 ID")
    resume_text: str = Field(..., description="이력서 텍스트")
    posting_text: str = Field(..., description="채용공고 텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "resume_id": "resume_123",
                "posting_id": "posting_789",
                "resume_text": "이력서 텍스트...",
                "posting_text": "채용공고 텍스트..."
            }
        }


class ResumeAnalysis(BaseModel):
    """이력서 분석 결과"""
    strengths: List[str] = Field(..., description="강점 목록")
    weaknesses: List[str] = Field(..., description="약점 목록")
    suggestions: List[str] = Field(..., description="제안 사항")


class PostingAnalysis(BaseModel):
    """채용공고 분석 결과"""
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    required_skills: List[str] = Field(..., description="필수 스킬")
    preferred_skills: List[str] = Field(..., description="우대 스킬")
    deadline: str = Field(..., description="마감일")


class Matching(BaseModel):
    """매칭 결과"""
    score: int = Field(..., ge=0, le=100, description="매칭 점수 (0-100)")
    grade: str = Field(..., description="등급 (A-F)")
    matched_skills: List[str] = Field(..., description="일치하는 스킬")
    missing_skills: List[str] = Field(..., description="부족한 스킬")
    bonus_skills: List[str] = Field(..., description="추가 강점")
    strength_summary: str = Field(..., description="강점 요약")
    weakness_summary: str = Field(..., description="약점 요약")
    recommendations: List[str] = Field(..., description="추천 사항")


class AnalyzeResponse(BaseModel):
    """분석 및 매칭도 응답 (API 3)"""
    success: bool = Field(True, description="성공 여부")
    resume_analysis: ResumeAnalysis = Field(..., description="이력서 분석")
    posting_analysis: PostingAnalysis = Field(..., description="채용공고 분석")
    matching: Matching = Field(..., description="매칭 결과")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "resume_analysis": {
                    "strengths": ["프론트엔드 개발 경험 3년", "React, TypeScript 숙련도 높음"],
                    "weaknesses": ["클라우드 경험 부족"],
                    "suggestions": ["AWS 자격증 취득 추천"]
                },
                "posting_analysis": {
                    "company": "카카오",
                    "position": "프론트엔드 개발자",
                    "required_skills": ["React", "TypeScript", "Next.js"],
                    "preferred_skills": ["GraphQL", "AWS"],
                    "deadline": "2026-01-15"
                },
                "matching": {
                    "score": 85,
                    "grade": "A",
                    "matched_skills": ["React", "TypeScript"],
                    "missing_skills": ["GraphQL", "Next.js"],
                    "bonus_skills": ["협업 프로젝트 경험 다수"],
                    "strength_summary": "기술 스택 일치율 85%, 요구 경험 일치율 90%",
                    "weakness_summary": "추가 학습 권장: GraphQL, Next.js",
                    "recommendations": [
                        "포트폴리오에 프로젝트 성과 수치화 필요",
                        "자기소개서에 협업 경험 강조"
                    ]
                }
            }
        }
