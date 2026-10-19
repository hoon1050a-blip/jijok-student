import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
import json
import time
import difflib

st.set_page_config(page_title="생기부 역량 평가 및 보완점 진단기", page_icon="📊", layout="wide")

# ==========================================
# 0. 세션 상태 초기화 (재실행 시 상태 보존)
# ==========================================
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None
if "safe_student_text" not in st.session_state:
    st.session_state.safe_student_text = ""
if "target_title" not in st.session_state:
    st.session_state.target_title = ""
if "token_info" not in st.session_state:
    st.session_state.token_info = None
if "quant_summary_md" not in st.session_state:
    st.session_state.quant_summary_md = ""
if "applied_weights_info" not in st.session_state:
    st.session_state.applied_weights_info = ""
if "major_notice_md" not in st.session_state:
    st.session_state.major_notice_md = ""

# ==========================================
# 1. 5등급제 환산 수식 (9등급 1.97 -> 5등급 1.21 정밀 보정) 및 가중치 함수
# ==========================================
# 9등급제 1.97이 5등급제에서 1.2등급 안팎(1.21)으로 수렴하도록 실증 통계 매듭점 반영
g9_knots_final = np.array([1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00])
g5_knots_final = np.array([1.00, 1.10, 1.22, 1.55, 1.85, 2.15, 2.45, 3.00, 3.60, 4.15, 4.65, 5.00])

def convert_9_to_5(g9):
    return round(float(np.interp(g9, g9_knots_final, g5_knots_final)), 2)

def calc_weighted_mean(series_val, series_weight):
    if len(series_val) == 0:
        return np.nan
    w = series_weight if series_weight.sum() > 0 else np.ones(len(series_val))
    return float(np.average(series_val, weights=w))

ART_PE_KWS = [
    "체육", "스포츠", "운동", "태권도", "무도", "경호", "미술", "디자인", "회화", "조소", "공예",
    "만화", "애니메이션", "웹툰", "음악", "성악", "피아노", "관현악", "작곡", "실용음악",
    "연기", "연극", "영화", "무용", "실기", "뷰티", "헤어", "공연", "뮤지컬", "사진", "조형", "예술", "영상"
]

STEM_KWS = [
    "공학", "컴퓨터", "소프트웨어", "전자", "전기", "기계", "화학", "화공", "생명", "바이오", "물리", "수학",
    "통계", "수리", "신소재", "재료", "고분자", "의예", "의학", "치의", "한의", "약학", "제약", "간호",
    "보건", "수의", "건축", "토목", "도시공학", "건설", "조경", "환경", "에너지", "인공지능", "AI",
    "데이터", "IT", "정보통신", "통신", "보안", "전산", "스마트", "나노", "반도체", "배터리", "원예",
    "산림", "농업", "농생명", "식물", "동물", "축산", "수산", "해양", "식품", "영양", "조리", "의류",
    "물리치료", "임상병리", "작업치료", "방사선", "치위생", "응급구조", "안경광학", "재활", "헬스케어",
    "수학교육", "물리교육", "화학교육", "생물교육", "지구과학교육", "과학교육", "가정교육",
    "항공", "우주", "조선", "모빌리티", "자동차", "로봇", "메카트로닉스", "철도", "선박", "자율운항",
    "천문", "지질", "지구", "기상", "자연과학", "이공", "과학기술", "사물인터넷", "모바일", "디지털",
    "공과", "소방", "목재", "생활과학", "기초과학"
]

HUMANITIES_KWS = [
    "경영", "경제", "통상", "무역", "회계", "세무", "금융", "보험", "물류", "비즈니스", "마케팅", "관광", "호텔", "항공서비스",
    "행정", "정치", "외교", "언론", "미디어", "방송", "광고", "홍보", "저널리즘", "커뮤니케이션",
    "사회", "복지", "심리", "상담", "아동", "청소년", "가족", "노인", "법학", "법무", "경찰", "범죄", "교정", "공공", "안보", "군사", "국방", "정책",
    "국어", "국문", "한국어", "문예창작", "영어", "영문", "독어", "독문", "독일", "불어", "불문", "프랑스",
    "중어", "중문", "중국", "일어", "일문", "일본", "스페인", "러시아", "아랍", "베트남", "언어", "통번역",
    "사학", "역사", "국사", "철학", "윤리", "종교", "신학", "불교", "기독교", "문화", "인류", "민속", "고고학",
    "문헌정보", "도서관", "지리학", "지역학", "글로벌", "국제", "인문", "교양", "소비자", "자치",
    "교육학", "국어교육", "영어교육", "역사교육", "지리교육", "윤리교육", "일반사회교육", "사회교육", "유아교육", "초등교육", "특수교육"
]

def classify_major_track_text(major_str, adm_type_str=""):
    m_str = str(major_str).strip()
    t_str = str(adm_type_str).strip()
    if "실기" in t_str or any(k in m_str for k in ART_PE_KWS):
        return "예체능"
    if any(k in m_str for k in ["인문", "사회", "소비자", "자치"]):
        if not any(k in m_str for k in ["자연", "공학", "공과"]):
            return "인문·사회"
    if any(k in m_str for k in ["공과", "자연과학", "공학", "소방", "목재", "생활과학", "기초과학"]):
        return "자연·공학"
    if any(k in m_str for k in ["자율전공", "자유전공", "창의융합", "융합인재", "자율융합", "전학부", "기초학부"]):
        return "자율·융합"
    if any(k in m_str for k in ["사회", "행정", "경영", "경제", "무역", "심리", "복지", "법", "언론", "미디어", "정치", "외교", "인문", "어문", "문학", "초등교육", "유아교육", "국어교육", "영어교육"]):
        if not any(k in m_str for k in ["공학", "데이터", "AI", "인공지능", "시스템", "소프트웨어", "컴퓨터"]):
            return "인문·사회"
    if any(k in m_str for k in STEM_KWS):
        return "자연·공학"
    if any(k in m_str for k in HUMANITIES_KWS):
        return "인문·사회"
    return "자율·융합"

@st.cache_data
def load_admission_data():
    try:
        df = pd.read_csv("jijok_admission_data.csv")
        if "내신등급" not in df.columns and "내신성적" in df.columns:
            df["내신등급"] = pd.to_numeric(df["내신성적"], errors="coerce")
        df = df.dropna(subset=["최종단계", "내신등급"]).copy()
        df["최종단계"] = df["최종단계"].astype(str).str.strip()
        df["세부유형"] = df["세부유형"].fillna("학생부종합").astype(str).str.replace("학생부종합학생부종합", "학생부종합", regex=False).str.strip()

        if "대입학년도" not in df.columns and "년도" in df.columns:
            df["대입학년도"] = df["년도"].astype(int) + 1
        if "연도가중치" not in df.columns and "대입학년도" in df.columns:
            df["연도가중치"] = df["대입학년도"].map({2026: 5, 2025: 3, 2024: 2}).fillna(1).astype(int)

        # 9등급 -> 5등급 정밀 환산식 전수 적용
        df["내신등급_5등급환산"] = df["내신등급"].apply(convert_9_to_5)

        if "실기전형여부" not in df.columns:
            df["실기전형여부"] = df["전형유형"].astype(str).str.contains("실기", na=False)
        if "검색용_대학명" not in df.columns:
            df["검색용_대학명"] = df["대학명"].apply(lambda x: re.sub(r'-.*', '', re.sub(r'\(.*?\)', '', str(x))).strip())
        if "계열" not in df.columns:
            df["계열"] = df.apply(lambda r: classify_major_track_text(r["모집단위"], r["전형유형"]), axis=1)
        return df
    except Exception:
        return None

adm_df = load_admission_data()

# ==========================================
# 2. 개인정보 비식별화 함수
# ==========================================
def mask_personal_info(text, student_name=""):
    text = re.sub(r'\d{6}[-\s]*[1-4]\d{6}', '******-*******', text)
    text = re.sub(r'01[016789][-\s]*\d{3,4}[-\s]*\d{4}', '010-****-****', text)
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '***@***.***', text)
    if student_name and len(student_name.strip()) >= 2:
        text = text.replace(student_name.strip(), "OOO")
    return text

# ==========================================
# 3. 대학·세부전형별 자동 반영비율 및 상위권 대학 안전장치 엔진
# ==========================================
def normalize_univ_keyword(univ_input):
    s = univ_input.strip()
    s = re.sub(r'\(.*?\)', '', s)
    s = s.replace("국립", "").replace("대학교", "").replace("대학", "").strip()
    alias_map = {
        "충대": "충남", "한밭": "한밭", "공주": "공주", "충북": "충북",
        "외대": "한국외국어", "한국외대": "한국외국어", "한기대": "한국기술교육",
        "교원대": "한국교원", "교통대": "한국교통", "금오공대": "금오공과",
        "과기대": "서울과학기술", "서울과기대": "서울과학기술", "시립대": "서울시립",
        "성대": "성균관", "숙대": "숙명여자", "이대": "이화여자", "고대": "고려", "연대": "연세",
        "경상대": "경상국립", "강원대": "강원", "전북대": "전북", "전남대": "전남", "경북대": "경북", "부산대": "부산"
    }
    return alias_map.get(s, s)

def is_top_competitive_univ(univ_input, major_input=""):
    u_key = normalize_univ_keyword(univ_input)
    top_univs = [
        "서울", "연세", "고려", "서강", "성균관", "한양", "중앙", "경희", "한국외국어", "서울시립",
        "이화여자", "건국", "동국", "홍익", "숙명여자", "국민", "숭실", "세종", "단국", "아주", "인하",
        "서울과학기술", "과학기술원", "포항공과", "한국교원"
    ]
    med_kws = ["의예", "의학", "치의", "한의", "약학", "수의"]
    if any(k in major_input for k in med_kws):
        return True
    if "서울대" in univ_input or any(t in u_key for t in top_univs):
        return True
    return False

def normalize_major_abbreviation(major_input):
    m = major_input.strip()
    abbr_map = {
        "컴공": "컴퓨터공학", "컴소": "컴퓨터소프트웨어", "소웨": "소프트웨어", "인공지능": "인공지능",
        "전전": "전기전자통신공학", "전기과": "전기공학", "전자과": "전자공학", "정통": "정보통신공학",
        "기계과": "기계공학", "기공": "기계공학", "화공": "화학공학", "생공": "생명공학", "생명과": "생명과학",
        "바메공": "바이오메디컬", "신소재": "신소재공학", "산공": "산업경영공학", "건축과": "건축",
        "수교": "수학교육", "수교과": "수학교육과", "국교": "국어교육", "영교": "영어교육",
        "경영과": "경영학", "경제과": "경제학", "행학과": "행정학", "행정과": "행정학", "정외": "정치외교",
        "미컴": "미디어커뮤니케이션", "신방": "언론정보", "심리과": "심리학", "간호과": "간호학",
        "물치": "물리치료", "임병": "임상병리", "자전": "자율전공", "자율전공": "자율전공학부"
    }
    return abbr_map.get(m, m)

def get_auto_univ_weights(univ_input, major_input="", subtype_input=""):
    if not univ_input.strip():
        return 40, 40, 20, "기본 표준 비율 (4 : 4 : 2)"

    u_key = normalize_univ_keyword(univ_input)
    m_key = major_input.strip()
    s_key = subtype_input.strip()

    if "중앙" in u_key:
        if "탐구" in s_key:
            return 50, 40, 10, f"{univ_input} [CAU탐구형인재] 기준 (학업·탐구 50% : 진로·전공 40% : 공동체 10%)"
        elif "융합" in s_key:
            return 40, 40, 20, f"{univ_input} [CAU융합형인재] 기준 (학업 40% : 진로 40% : 공동체 20% 균형형)"
        return 45, 40, 15, f"{univ_input} 학생부종합 평균 기준 (4.5 : 4.0 : 1.5)"

    if "한양" in u_key:
        if "추천" in s_key:
            return 50, 30, 20, f"{univ_input} [추천형] 학업성취 중심 기준 (5 : 3 : 2)"
        elif "면접" in s_key:
            return 40, 35, 25, f"{univ_input} [면접형] 학업·진로·소통 기준 (4.0 : 3.5 : 2.5)"
        return 45, 40, 15, f"{univ_input} [서류형/일반] 학업·진로 중심 기준 (4.5 : 4.0 : 1.5)"

    if "고려" in u_key:
        if "계열적합" in s_key:
            return 40, 50, 10, f"{univ_input} [계열적합전형] 전공진로·탐구 중심 기준 (4 : 5 : 1)"
        return 50, 30, 20, f"{univ_input} [학업우수전형] 학업역량 중심 기준 (5 : 3 : 2)"

    if "성균관" in u_key:
        if "탐구" in s_key or "과학인재" in s_key:
            return 45, 45, 10, f"{univ_input} [탐구형/과학인재] 전공탐구 중심 기준 (4.5 : 4.5 : 1.0)"
        return 50, 30, 20, f"{univ_input} [융합형] 학업 중심 기준 (5 : 3 : 2)"

    if "충남" in u_key:
        if "서류" in s_key:
            return 45, 40, 15, f"{univ_input} [{s_key}] 서류 100% 학업·진로 중심 기준 (4.5 : 4.0 : 1.5)"
        return 40, 40, 20, f"{univ_input} [{s_key if s_key else '학생부종합'}] 표준 균형 기준 (4 : 4 : 2)"

    if any(k in u_key for k in ["교육", "교원"]) or "교육" in m_key or "사범" in m_key:
        return 35, 40, 25, f"{univ_input} 교원·사범계열 학종 기준 (3.5 : 4.0 : 2.5 - 공동체·교직인성 강화)"

    academic_heavy = {
        "서울": (50, 30, 20, "서울대 일반/지균 학업·탐구 중심 기준 (5 : 3 : 2)"),
        "서강": (50, 30, 20, "서강대 일반형 학업역량 중심 기준 (5 : 3 : 2)"),
        "이화여자": (50, 30, 20, "이화여대 미래인재 학업 중심 기준 (5 : 3 : 2)"),
        "과학기술원": (50, 35, 15, "과기원(KAIST/GIST/DGIST/UNIST) 학업·연구 중심 기준 (5.0 : 3.5 : 1.5)")
    }
    for k, val in academic_heavy.items():
        if k in u_key:
            return val

    career_heavy = {
        "연세": (30, 50, 20, f"{univ_input} 활동우수형 진로·학업 반영 기준 (3 : 5 : 2)"),
        "경희": (30, 50, 20, f"{univ_input} 네오르네상스전형 공식 반영비율 (3 : 5 : 2)"),
        "건국": (30, 50, 20, f"{univ_input} KU자기추천/Cogito 공식 반영비율 (3 : 5 : 2)"),
        "동국": (30, 50, 20, f"{univ_input} Do Dream전형 진로역량 중심 기준 (3 : 5 : 2)"),
        "서울시립": (35, 45, 20, f"{univ_input} 학생부종합 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "세종": (35, 45, 20, f"{univ_input} 창의인재전형 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "숙명여자": (35, 45, 20, f"{univ_input} 숙명인재 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "서울과학기술": (35, 45, 20, f"{univ_input} 학교생활우수자/창의융합 기준 (3.5 : 4.5 : 2.0)"),
        "숭실": (35, 45, 20, f"{univ_input} SSU미래인재 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "국민": (35, 45, 20, f"{univ_input} 국민프런티어 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "단국": (35, 45, 20, f"{univ_input} DKU인재 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "아주": (35, 45, 20, f"{univ_input} 에이스전형 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "인하": (35, 45, 20, f"{univ_input} 인하미래인재 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "한국기술교육": (35, 45, 20, f"{univ_input} 창의인재 진로·실천 중심 기준 (3.5 : 4.5 : 2.0)")
    }
    for k, val in career_heavy.items():
        if k in u_key:
            return val

    return 40, 40, 20, f"{univ_input} 학생부종합 표준 균형 반영 비율 (4 : 4 : 2 자동 적용)"

# ==========================================
# 4. 계열 분리 및 소표본(N < 5) 계층적 합불 보정 분석 엔진
# ==========================================
def extract_major_root_keywords(major_input):
    m = normalize_major_abbreviation(major_input)
    keyword_groups = [
        ["수학", "통계", "수리", "데이터", "금융수학"],
        ["컴퓨터", "소프트웨어", "인공지능", "AI", "정보", "보안", "전산", "스마트", "IT", "융합학부"],
        ["기계", "자동차", "항공", "우주", "로봇", "메카트로닉스", "자율운항", "조선", "모빌리티"],
        ["전기", "전자", "반도체", "정보통신", "제어", "광학", "전파"],
        ["화학", "화공", "신소재", "고분자", "재료", "에너지", "나노", "배터리"],
        ["생명", "생물", "바이오", "유전", "식품", "농업", "원예", "산림", "수의", "동식물", "약학"],
        ["물리", "천문", "지구", "환경", "기상", "해양", "지질"],
        ["건축", "토목", "도시", "건설", "조경", "실내", "인프라"],
        ["간호", "보건", "물리치료", "임상병리", "작업치료", "방사선", "응급", "치위생", "의예", "한의"],
        ["교육", "사범", "초등", "교직"],
        ["경영", "경제", "금융", "회계", "무역", "통상", "물류", "비즈니스", "글로벌비즈니스"],
        ["행정", "정치", "외교", "법학", "경찰", "사회", "복지", "심리", "언론", "미디어", "광고", "자치"],
        ["국어", "국문", "영어", "영문", "일어", "중어", "불어", "독어", "언어", "문헌", "사학", "역사", "철학", "인문"]
    ]
    matched_keywords = []
    for grp in keyword_groups:
        if any(k in m for k in grp):
            matched_keywords.extend(grp)
    if not matched_keywords:
        clean_m = m.replace("학과", "").replace("학부", "").replace("전공", "").replace("공학", "").strip()
        if len(clean_m) >= 2:
            matched_keywords.append(clean_m[:2])
    return list(set(matched_keywords))

def find_best_matching_majors(univ_df, major_input):
    if univ_df.empty:
        return pd.DataFrame(), "해당 대학/계열 학종 데이터 없음", []

    norm_m = normalize_major_abbreviation(major_input)
    core_m = norm_m.replace("학과", "").replace("학부", "").replace("전공", "").strip()
    available_majors = sorted(univ_df["모집단위"].dropna().unique().tolist())

    if norm_m in available_majors:
        return univ_df[univ_df["모집단위"] == norm_m], f"✅ 정확 일치 매칭: 【{norm_m}】", [norm_m]

    if core_m:
        direct_mask = univ_df["모집단위"].str.contains(re.escape(core_m), na=False)
        if direct_mask.any():
            matched_names = univ_df[direct_mask]["모집단위"].unique().tolist()
            return univ_df[direct_mask], f"✅ 핵심어 포함 매칭: 【{', '.join(matched_names[:3])}】", matched_names

    root_kws = extract_major_root_keywords(norm_m)
    scored_majors = []
    for m_name in available_majors:
        seq_ratio = difflib.SequenceMatcher(None, core_m, m_name).ratio()
        kw_bonus = 0.35 if any(k in m_name for k in root_kws) else 0.0
        total_score = seq_ratio + kw_bonus
        if total_score >= 0.42:
            scored_majors.append((total_score, m_name))

    scored_majors.sort(reverse=True, key=lambda x: x[0])
    if scored_majors:
        best_names = [x[1] for x in scored_majors[:4]]
        sub_df = univ_df[univ_df["모집단위"].isin(best_names)]
        return sub_df, f"🔄 유사 모집단위 자동 보정: '{major_input}' 대신 해당 대학 동일 계열 내 가장 유사한 【{', '.join(best_names[:3])}】 데이터로 매칭했습니다.", best_names

    return pd.DataFrame(), f"⚠️ 해당 대학 학종 데이터 내 '{major_input}' 직접 사례가 없어 동일 대학·동일 계열 전체 학종 데이터로 보정합니다.", available_majors[:5]

def analyze_school_admission_benchmark(df, univ_input, major_input, subtype_input="전체 (학생부종합 통합)", track_filter="자동 감지"):
    if df is None or df.empty:
        return {
            "match_level": "데이터 없음",
            "detected_track": "미분류",
            "major_notice": "",
            "summary_md": "⚠️ `jijok_admission_data.csv` 파일을 찾을 수 없어 일반 전국 학생부종합 기준으로 진단합니다.",
            "prompt_context": "본교 3개년 입결 파일이 연결되지 않았습니다.",
            "similar_majors_context": "본교 데이터 없음"
        }

    if track_filter and track_filter not in ["전체 계열", "자동 감지"]:
        target_track = track_filter
    else:
        target_track = classify_major_track_text(major_input)

    if target_track == "예체능":
        base_df = df.copy()
        silgi_note = "예체능 계열 학과이므로 실기/실적 데이터를 포함하여 분석합니다."
    else:
        base_df = df[~df["실기전형여부"]].copy()
        silgi_note = f"**[{target_track} 계열 전용 분석]** 일반 계열 왜곡을 막기 위해 실기전형을 배제하고, 다른 계열 데이터와의 혼입을 차단했습니다."

    jonghap_all_df = base_df[base_df["전형유형"].str.contains("종합", na=False)].copy()

    exact_univ_mask = jonghap_all_df["대학명"] == univ_input.strip()
    if exact_univ_mask.any():
        univ_jonghap_df = jonghap_all_df[exact_univ_mask].copy()
    else:
        u_key = normalize_univ_keyword(univ_input)
        univ_mask = jonghap_all_df["대학명"].str.contains(re.escape(u_key), na=False)
        if "검색용_대학명" in jonghap_all_df.columns:
            univ_mask = univ_mask | jonghap_all_df["검색용_대학명"].str.contains(re.escape(u_key), na=False)
        univ_jonghap_df = jonghap_all_df[univ_mask].copy()

    # 동일 대학 내 동일 계열(인문·사회 vs 자연·공학) 분리
    if target_track in ["인문·사회", "자연·공학"] and not univ_jonghap_df.empty:
        univ_same_track_df = univ_jonghap_df[univ_jonghap_df["계열"].isin([target_track, "자율·융합"])].copy()
        if univ_same_track_df.empty:
            univ_same_track_df = univ_jonghap_df.copy()
    else:
        univ_same_track_df = univ_jonghap_df.copy()

    specific_subtype_active = (
        subtype_input
        and not subtype_input.startswith("전체")
        and not subtype_input.startswith("기본")
    )
    if specific_subtype_active and not univ_same_track_df.empty:
        subtype_df = univ_same_track_df[univ_same_track_df["세부유형"] == subtype_input].copy()
        if subtype_df.empty:
            subtype_df = univ_same_track_df.copy()
            subtype_label = f"학생부종합 ({target_track} 계열)"
        else:
            subtype_label = subtype_input
    else:
        subtype_df = univ_same_track_df.copy()
        subtype_label = f"학생부종합 ({target_track} 계열)"

    matched_major_df, major_notice_str, matched_major_list = find_best_matching_majors(subtype_df, major_input)
    if matched_major_df.empty and specific_subtype_active:
        matched_major_df, major_notice_str, matched_major_list = find_best_matching_majors(univ_same_track_df, major_input)

    pass_in_matched = matched_major_df[matched_major_df["최종단계"].isin(["합격", "충원합격"])] if not matched_major_df.empty else pd.DataFrame()
    pass_in_same_track = univ_same_track_df[univ_same_track_df["최종단계"].isin(["합격", "충원합격"])] if not univ_same_track_df.empty else pd.DataFrame()

    # 소표본(N < 5) 계층적 결합: 표본 부족 시 동일 계열 전체 학종 결합
    if not matched_major_df.empty and len(pass_in_matched) > 0:
        if len(matched_major_df) < 5 and len(univ_same_track_df) > len(matched_major_df):
            target_df = univ_same_track_df.copy()
            match_level = (
                f"⚖️ 1+2단계 소표본 안전 결합 ({subtype_label}): [{major_input}] 직접 지원이 {len(matched_major_df)}건(합격 {len(pass_in_matched)}건)뿐이므로, "
                f"소표본 왜곡(합격률 100% 착시)을 방지하기 위해 **[{univ_input} {target_track} 계열 학종 전체 {len(target_df)}건 (합격 {len(pass_in_same_track)}건)]**을 통합하여 엄격 보정"
            )
        else:
            target_df = matched_major_df.copy()
            match_level = f"✅ 1단계 학과·전형 매칭 ({subtype_label}): 본교 3개년 [{univ_input} - {', '.join(matched_major_list[:2])}] 지원 총 {len(target_df)}건 (합격 {len(pass_in_matched)}건)"
    elif not univ_same_track_df.empty and len(pass_in_same_track) > 0:
        target_df = univ_same_track_df.copy()
        match_level = f"🔄 2단계 동일 대학·동일 계열({target_track}) 학종 보정: [{major_input}] 직접 합격 사례가 없어 **[{univ_input} {target_track} 계열 학종 전체 지원 {len(target_df)}건 (합격 {len(pass_in_same_track)}건)]** 기준으로 보정"
    else:
        target_df = pd.DataFrame()
        match_level = f"🌐 3단계 전국 학종 및 본교 [{target_track}] 계열 보정: 본교 3개년 데이터 내 [{univ_input} {target_track}] 학종 합격 사례가 없어, 본교 {target_track} 계열 학종 가중 합격선 및 전국 입결 기준으로 엄격 보정"

    # 동일 계열 한정 추천 대학 산출용 데이터
    sim_keywords = extract_major_root_keywords(major_input)
    valid_types_df = base_df[
        (base_df["전형유형"].isin(["학생부위주(종합)", "학생부위주(교과)"]))
        & (base_df["계열"].isin([target_track, "자율·융합"]))
    ].copy()
    if sim_keywords:
        pattern = "|".join([re.escape(k) for k in sim_keywords])
        sim_df = valid_types_df[valid_types_df["모집단위"].str.contains(pattern, na=False)].copy()
    else:
        sim_df = valid_types_df.copy()

    similar_lines = []
    if not sim_df.empty:
        sim_pass = sim_df[sim_df["최종단계"].isin(["합격", "충원합격"])]
        if not sim_pass.empty:
            grouped_rows = []
            for (u_name, m_name, t_type), grp in sim_df.groupby(["대학명", "모집단위", "전형유형"]):
                p_grp = grp[grp["최종단계"].isin(["합격", "충원합격"])]
                f_grp = grp[grp["최종단계"] == "불합격"]
                if len(p_grp) == 0:
                    continue
                w_mean_5 = calc_weighted_mean(p_grp["내신등급_5등급환산"], p_grp["연도가중치"])
                w_mean_9 = calc_weighted_mean(p_grp["내신등급"], p_grp["연도가중치"])
                pass_rate = round(len(p_grp) / len(grp) * 100, 1)
                fail_top_5 = round(float(np.percentile(f_grp["내신등급_5등급환산"], 25)), 2) if len(f_grp) > 0 else None
                grouped_rows.append({
                    "대학명": u_name, "모집단위": m_name, "전형유형": t_type,
                    "pass_cnt": len(p_grp), "total_cnt": len(grp), "pass_rate": pass_rate,
                    "w_mean_5": w_mean_5, "w_mean_9": w_mean_9,
                    "min_5": p_grp["내신등급_5등급환산"].min(), "max_5": p_grp["내신등급_5등급환산"].max(),
                    "fail_top_5": fail_top_5
                })
            sim_summary_df = pd.DataFrame(grouped_rows).sort_values(by=["전형유형", "w_mean_5"])
            jh_sims = sim_summary_df[sim_summary_df["전형유형"] == "학생부위주(종합)"].head(15)
            gk_sims = sim_summary_df[sim_summary_df["전형유형"] == "학생부위주(교과)"].head(15)
            combined_sims = pd.concat([jh_sims, gk_sims])

            for _, r in combined_sims.iterrows():
                fail_note = f", 고성적탈락주의선 {r['fail_top_5']:.2f}등급" if r["fail_top_5"] is not None else ""
                similar_lines.append(
                    f"- {r['대학명']} | {r['모집단위']} [{r['전형유형']}]: 최근가중 5등급환산 평균 {r['w_mean_5']:.2f}등급 (9등급제 {r['w_mean_9']:.2f} / 합격 {r['pass_cnt']}건·총지원 {r['total_cnt']}건, 합격률 {r['pass_rate']}%{fail_note})"
                )

    similar_majors_context = "\n".join(similar_lines) if similar_lines else f"본교 내 동일 키워드 학과 합격 사례가 적어 전국 대학 [{target_track}] 계열 기준(학종·교과 통합)으로 추천하세요."

    top_univ_flag = is_top_competitive_univ(univ_input, major_input)
    if top_univ_flag:
        safety_cap_instruction = (
            f"🚨 **[상위권 대학·인기학과 판정 상한선(Safety Cap) 강제 규칙]**: "
            f"[{univ_input}] 학생부종합전형은 전국 특목고·자사고·일반고 최상위권이 경합하고 서류·면접 정성 변수가 매우 큰 전형입니다. "
            f"따라서 학생의 내신이 선배 합격선보다 높거나 표본이 적더라도 **절대 '🟢안정권'으로 판정하지 마세요!** "
            f"최대 판정 상한선은 **'🎯적정권(유력)'** 또는 **'🟡소신 경합권'**입니다."
        )
    else:
        safety_cap_instruction = "본교 합불 통계와 보수적 권장선을 기준으로 객관적으로 판정하세요."

    if target_df.empty:
        fallback_fail_info = ""
        if not univ_same_track_df.empty:
            f_best_5 = univ_same_track_df["내신등급_5등급환산"].min()
            fallback_fail_info = f" (주의: 본교에서 {univ_input} [{target_track}] 계열 학종에 {len(univ_same_track_df)}건 지원했으나 모두 불합격했으며, 최고 성적 불합격자는 5등급제 환산 {f_best_5:.2f}등급이었습니다!)"

        return {
            "match_level": match_level,
            "detected_track": target_track,
            "major_notice": f"[{target_track} 계열] {major_notice_str}",
            "summary_md": (
                f"**{match_level}**\n\n"
                f"* {silgi_note}\n"
                f"* 본교 최근 3개년(2024~2026 대입) 데이터에 해당 대학 [{target_track}] 계열 학종 합격 사례가 없어, 전국 대학 위상과 본교 학종 합불 분포를 교차하여 **보수적(엄격한) 기준**으로 진단합니다.{fallback_fail_info}"
            ),
            "prompt_context": (
                f"본교 3개년 데이터에 {univ_input} {major_input} ({target_track} 계열, {subtype_label}) 합격 사례가 없습니다.{fallback_fail_info}\n"
                f"{safety_cap_instruction}"
            ),
            "similar_majors_context": similar_majors_context
        }

    pass_df = target_df[target_df["최종단계"].isin(["합격", "충원합격"])].copy()
    fail_df = target_df[target_df["최종단계"] == "불합격"].copy()

    total_cnt = len(target_df)
    pass_cnt = len(pass_df)
    fail_cnt = len(fail_df)
    pass_rate = round((pass_cnt / total_cnt) * 100, 1) if total_cnt > 0 else 0.0

    w_pass_5 = calc_weighted_mean(pass_df["내신등급_5등급환산"], pass_df["연도가중치"])
    w_pass_9 = calc_weighted_mean(pass_df["내신등급"], pass_df["연도가중치"])

    exact_dept_note = ""
    if not matched_major_df.empty and len(pass_in_matched) > 0 and len(target_df) > len(matched_major_df):
        exact_w5 = calc_weighted_mean(pass_in_matched["내신등급_5등급환산"], pass_in_matched["연도가중치"])
        exact_w9 = calc_weighted_mean(pass_in_matched["내신등급"], pass_in_matched["연도가중치"])
        exact_dept_note = f" *(참고: {major_input} 단독 합격 {len(pass_in_matched)}건은 5등급제 {exact_w5:.2f}등급 / 9등급제 {exact_w9:.2f}등급)*"

    pass_q65_5 = float(np.percentile(pass_df["내신등급_5등급환산"], 65))
    if fail_cnt > 0:
        w_fail_5 = calc_weighted_mean(fail_df["내신등급_5등급환산"], fail_df["연도가중치"])
        fail_top25_5 = float(np.percentile(fail_df["내신등급_5등급환산"], 25))
        fail_best_5 = float(fail_df["내신등급_5등급환산"].min())
        strict_cutoff_5 = round(min(w_pass_5, pass_q65_5, max(fail_top25_5, float(pass_df["내신등급_5등급환산"].min()))), 2)
        calibration_note_md = (
            f"⚖️ **동일 계열({target_track}) 합불 교차 보정 지표:** 최근 3개년 본교 지원 **{total_cnt}건 중 {pass_cnt}건 합격 (실질 합격률 {pass_rate}%)** | "
            f"최근 가중(5:3:2) 합격선 **{w_pass_5:.2f}등급** | **보수적 권장선: {strict_cutoff_5:.2f}등급 이내**{exact_dept_note}"
        )
        fail_prompt_note = (
            f"[매우 중요 - 동일 계열({target_track}) 불합격 데이터 기반 엄격 보정 지시]\n"
            f"- 분석 대상 풀({univ_input} {target_track} 계열 학종) 총 지원 {total_cnt}건 중 합격은 {pass_cnt}건, 불합격은 {fail_cnt}건(실질 합격률 {pass_rate}%)입니다.{exact_dept_note}\n"
            f"- 불합격자들의 최근 가중 평균은 5등급제 환산 {w_fail_5:.2f}등급이며, 불합격자 상위 25% 성적은 {fail_top25_5:.2f}등급(최고 성적 탈락자 {fail_best_5:.2f}등급)입니다.\n"
            f"- 실질적인 권장선은 {strict_cutoff_5:.2f}등급 이내이며, 설령 내신이 권장선 이내라도 {safety_cap_instruction}\n"
            f"- 단, 학생이 보는 리포트 본문에는 개별 불합격 선배 사례를 나열하지 말고 '동일 계열 실질 합격률({pass_rate}%)과 보수적 권장선({strict_cutoff_5:.2f}등급)' 개념으로만 정제하여 설명하세요."
        )
    else:
        strict_cutoff_5 = round(w_pass_5 - 0.08, 2) if pass_cnt < 3 else round(w_pass_5, 2)
        calibration_note_md = (
            f"⚖️ **최근연도 가중(5:3:2) 보정 지표 ({target_track} 계열):** 최근 3개년 본교 학종 합격 **{pass_cnt}건** | "
            f"최근 가중 합격선 **{w_pass_5:.2f}등급** (소표본 보수적 권장선: **{strict_cutoff_5:.2f}등급 이내**)"
        )
        fail_prompt_note = (
            f"- 최근 가중 학종 합격 평균은 5등급제 환산 {w_pass_5:.2f}등급이나, 합격 표본이 {pass_cnt}건으로 적으므로 보수적 권장선({strict_cutoff_5:.2f}등급)을 적용하세요.\n"
            f"- {safety_cap_instruction}"
        )

    rows_md = []
    prompt_lines = [f"[{match_level}]", fail_prompt_note]

    for stage, stage_label in [("합격", "최초합격"), ("충원합격", "충원합격(추합)")]:
        st_sub = pass_df[pass_df["최종단계"] == stage]
        if st_sub.empty:
            continue
        cnt = len(st_sub)
        w_m5 = calc_weighted_mean(st_sub["내신등급_5등급환산"], st_sub["연도가중치"])
        w_m9 = calc_weighted_mean(st_sub["내신등급"], st_sub["연도가중치"])
        min_5, max_5 = st_sub["내신등급_5등급환산"].min(), st_sub["내신등급_5등급환산"].max()
        min_9, max_9 = st_sub["내신등급"].min(), st_sub["내신등급"].max()

        rows_md.append(
            f"| **{subtype_label}** | {stage_label} | {cnt}건 | **{w_m5:.2f} 등급** *(범위: {min_5:.2f}~{max_5:.2f})* | {w_m9:.2f} 등급 *(범위: {min_9:.2f}~{max_9:.2f})* |"
        )
        prompt_lines.append(
            f"- {subtype_label} [{stage_label}, {cnt}건]: 최근가중(5:3:2) 5등급제 환산 평균 {w_m5:.2f}등급(범위 {min_5:.2f}~{max_5:.2f}) / 9등급제 가중평균 {w_m9:.2f}등급(범위 {min_9:.2f}~{max_9:.2f})"
        )

    pass_sorted = pass_df.sort_values(by=["대입학년도", "내신등급_5등급환산"], ascending=[False, True]).head(6)
    case_lines = []
    for _, r in pass_sorted.iterrows():
        sub_type = str(r["세부유형"]).strip() if pd.notna(r["세부유형"]) and str(r["세부유형"]).strip() else "학생부종합"
        case_lines.append(
            f"  * [{int(r['대입학년도'])}대입] {r['모집단위']} ({sub_type} - {r['최종단계']}): **5등급환산 {r['내신등급_5등급환산']:.2f}등급** (9등급제 {r['내신등급']:.2f})"
        )

    table_header = (
        f"**{match_level}**\n\n"
        f"> {calibration_note_md}\n\n"
        "| 전형 및 계열 구분 (학종 전용) | 합격단계 | 본교 합격 건수 | **최근가중(5:3:2) 5등급제 평균 (범위)** | 최근가중 9등급제 원본 평균 (범위) |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    summary_md = table_header + "\n".join(rows_md) + "\n\n**📌 본교 동일 계열 최근 학생부종합 주요 합격 사례 (최신 연도순, 불합격 비노출):**\n" + "\n".join(case_lines)
    prompt_context = "\n".join(prompt_lines) + "\n[본교 동일 계열 합격 사례(참고)]\n" + "\n".join(case_lines)

    return {
        "match_level": match_level,
        "detected_track": target_track,
        "major_notice": f"[{target_track} 계열] {major_notice_str}",
        "summary_md": summary_md,
        "prompt_context": prompt_context,
        "similar_majors_context": similar_majors_context
    }

# ==========================================
# 5. KaTeX 수식 및 마스터진단표 지원 PDF 컴포넌트
# ==========================================
def pdf_download_component(result_text: str, title_prefix: str):
    safe_text = json.dumps(result_text, ensure_ascii=False)
    safe_filename = json.dumps(f"{title_prefix}_생기부종합진단리포트.pdf", ensure_ascii=False)
    safe_title = json.dumps(f"{title_prefix} 학생부종합 AI 정밀 진단 리포트", ensure_ascii=False)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
        <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            body {{ font-family: 'Noto Sans KR', sans-serif; margin: 0; padding: 8px 0; }}
            .btn-container {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
            .pdf-btn {{
                background-color: #FF4B4B; color: white; border: none; padding: 10px 18px;
                font-size: 14px; font-weight: 600; border-radius: 8px; cursor: pointer;
                font-family: 'Noto Sans KR', sans-serif; transition: opacity 0.2s;
            }}
            .pdf-btn.print-mode {{ background-color: #1E3A8A; }}
            .pdf-btn:hover {{ opacity: 0.88; }}
            .status-msg {{ margin-top: 8px; font-size: 13px; font-weight: 500; color: #374151; min-height: 22px; }}
            #pdf-render-area {{
                display: none; padding: 15px 20px; color: #111827;
                line-height: 1.65; font-size: 10pt; background: #ffffff;
            }}
            #pdf-render-area .doc-header {{
                text-align: center; font-size: 17pt; font-weight: 700; color: #1E3A8A;
                border-bottom: 3px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px;
            }}
            #pdf-render-area h2 {{
                color: #1E3A8A; border-left: 7px solid #1E3A8A; padding-left: 10px;
                margin-top: 22px; margin-bottom: 10px; font-size: 13.5pt; page-break-after: avoid;
            }}
            #pdf-render-area h3 {{
                color: #1D4ED8; margin-top: 16px; margin-bottom: 8px; font-size: 11.5pt; page-break-after: avoid;
            }}
            #pdf-render-area table {{
                width: 100%; border-collapse: collapse; margin: 10px 0 16px 0;
                font-size: 9pt; page-break-inside: avoid;
            }}
            #pdf-render-area th, #pdf-render-area td {{
                border: 1px solid #CBD5E1; padding: 6px 8px; text-align: left;
            }}
            #pdf-render-area th {{ background-color: #EFF6FF; color: #1E3A8A; font-weight: 700; text-align: center; }}
            #pdf-render-area blockquote {{
                background-color: #F8FAFC; border-left: 4px solid #3B82F6;
                margin: 0 0 14px 0; padding: 10px 14px; border-radius: 5px;
                page-break-inside: avoid; break-inside: avoid;
            }}
            #pdf-render-area blockquote p {{ margin: 5px 0; }}
            #pdf-render-area hr {{ border: 0; border-top: 1px solid #E5E7EB; margin: 16px 0; }}
        </style>
    </head>
    <body>
        <div class="btn-container">
            <button class="pdf-btn" id="dl-btn" onclick="downloadDirectPDF()">📥 진단 리포트 PDF 바로 다운로드</button>
            <button class="pdf-btn print-mode" onclick="openPrintPDF()">🖨️ 고화질 PDF 저장 / 인쇄 (마스터 진단표·수식 완벽 권장)</button>
        </div>
        <div class="status-msg" id="status-text"></div>
        <div id="pdf-render-area"></div>

        <script>
            const rawMarkdown = {safe_text};
            const fileName = {safe_filename};
            const docTitle = {safe_title};
            const renderArea = document.getElementById('pdf-render-area');
            const statusText = document.getElementById('status-text');
            const dlBtn = document.getElementById('dl-btn');

            function renderMathAndMarkdown() {{
                renderArea.innerHTML = '<div class="doc-header">' + docTitle + '</div>' + marked.parse(rawMarkdown);
                renderMathInElement(renderArea, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '\\\\[', right: '\\\\]', display: true}}
                    ],
                    throwOnError: false
                }});
            }}

            function downloadDirectPDF() {{
                dlBtn.disabled = true;
                statusText.innerText = "⏳ 진단 리포트 렌더링 및 PDF 변환 중입니다 (약 3~5초 소요)...";
                renderMathAndMarkdown();
                renderArea.style.display = 'block';

                const opt = {{
                    margin:       [10, 10, 10, 10],
                    filename:     fileName,
                    image:        {{ type: 'jpeg', quality: 0.98 }},
                    html2canvas:  {{ scale: 2, useCORS: true, scrollY: 0 }},
                    jsPDF:        {{ unit: 'mm', format: 'a4', orientation: 'portrait' }},
                    pagebreak:    {{ mode: ['avoid-all', 'css', 'legacy'] }}
                }};

                html2pdf().set(opt).from(renderArea).save().then(() => {{
                    renderArea.style.display = 'none';
                    statusText.innerText = "✅ PDF 다운로드가 완료되었습니다!";
                    dlBtn.disabled = false;
                }}).catch((err) => {{
                    renderArea.style.display = 'none';
                    statusText.innerText = "⚠️ 변환 중 오류가 발생했습니다. 오른쪽 고화질 저장 버튼을 이용해 주세요.";
                    dlBtn.disabled = false;
                }});
            }}

            function openPrintPDF() {{
                renderMathAndMarkdown();
                const printWin = window.open('', '_blank', 'width=1000,height=880');
                printWin.document.write(`
                    <html>
                    <head>
                        <title>${{fileName}}</title>
                        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
                        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
                        <style>
                            body {{ font-family: 'Noto Sans KR', sans-serif; padding: 20px; line-height: 1.65; font-size: 10pt; color: #111827; }}
                            .doc-header {{ text-align: center; font-size: 17pt; font-weight: 700; color: #1E3A8A; border-bottom: 3px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px; }}
                            h2 {{ color: #1E3A8A; border-left: 7px solid #1E3A8A; padding-left: 10px; margin-top: 22px; margin-bottom: 10px; font-size: 13.5pt; page-break-after: avoid; }}
                            h3 {{ color: #1D4ED8; margin-top: 16px; margin-bottom: 8px; font-size: 11.5pt; page-break-after: avoid; }}
                            table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px 0; font-size: 9pt; page-break-inside: avoid; }}
                            th, td {{ border: 1px solid #CBD5E1; padding: 6px 8px; text-align: left; }}
                            th {{ background-color: #EFF6FF !important; color: #1E3A8A !important; font-weight: 700; text-align: center; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                            blockquote {{ background-color: #F8FAFC !important; border-left: 4px solid #3B82F6; margin: 0 0 14px 0; padding: 10px 14px; border-radius: 5px; page-break-inside: avoid; break-inside: avoid; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                            blockquote p {{ margin: 5px 0; }}
                            hr {{ border: 0; border-top: 1px solid #E5E7EB; margin: 16px 0; }}
                        </style>
                    </head>
                    <body>${{renderArea.innerHTML}}</body>
                    </html>
                `);
                printWin.document.close();
                setTimeout(() => {{
                    printWin.focus();
                    printWin.print();
                }}, 600);
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=100)


# ==========================================
# 화면 UI 구성: 지역 ➔ 계열(인문/자연) ➔ 대학(캠퍼스) ➔ 학종 세부전형 ➔ 모집단위 연동
# ==========================================
st.title("📊 생기부 역량 평가 및 보완점 진단기")
st.markdown(
    "학생의 **학교생활기록부 PDF**를 업로드하면 **① 시수 반영 내신 및 5등급제 정밀 환산**, "
    "**② 동일 대학·동일 계열(인문/자연 분리) 합불 교차 보정(최근연도 5:3:2 가중)**, "
    "**③ 30개 세부 평가 요인 마스터 진단표(●) 및 강점·감점·제언 3단 심층 분석**, "
    "**④ 차기 학기 세특·창체 로드맵 및 동일 계열 추천 대학(종합·교과 통합 6선)**을 제시합니다."
)
st.caption("하다가 어려운점이 있다면 제작자에게 문의주세요(지족고 송정훈)")

api_key = st.text_input("Gemini API 키를 입력하세요", type="password")

st.markdown("#### 1️⃣ 지역 · 계열(인문/자연) · 목표 대학 · 학생부종합 세부전형 · 모집단위 선택")
input_mode = st.radio(
    "대학 및 학과 입력 방식",
    options=[
        "📋 본교 3개년 DB 연동 필터 및 단어 검색으로 선택 (지역 ➔ 계열 ➔ 대학 ➔ 학종 세부전형 ➔ 학과)",
        "✏️ 목록에 없는 대학·학과 직접 타이핑 입력"
    ],
    index=0,
    horizontal=True
)

selected_subtype = "전체 (학생부종합 통합)"
selected_track_ui = "자동 감지"

if input_mode.startswith("📋") and adm_df is not None:
    non_silgi_df = adm_df[~adm_df["실기전형여부"]].copy()
    jh_only_df = non_silgi_df[non_silgi_df["전형유형"].str.contains("종합", na=False)].copy()

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.0, 1.1, 1.2, 1.5])
    with f_col1:
        priority_regions = ["대전", "세종", "충남", "충북", "서울", "경기", "인천", "강원", "전북", "전남", "광주", "경북", "대구", "경남", "부산", "울산", "제주"]
        avail_regions = [r for r in priority_regions if r in non_silgi_df["지역"].dropna().unique()]
        selected_region = st.selectbox("📍 1. 지역 선택", options=["전체(전국)"] + avail_regions, index=0)

    with f_col2:
        selected_track_ui = st.selectbox(
            "🏛️ 2. 계열 구분",
            options=["전체 계열", "인문·사회", "자연·공학", "자율·융합"],
            index=0
        )

    with f_col3:
        univ_search_kw = st.text_input("🔎 3. 대학명 단어 검색", placeholder="예: 한양대, 중앙대, 충남대")

    filtered_univ_df = non_silgi_df.copy()
    if selected_region != "전체(전국)":
        filtered_univ_df = filtered_univ_df[filtered_univ_df["지역"] == selected_region]
    if selected_track_ui != "전체 계열":
        filtered_univ_df = filtered_univ_df[filtered_univ_df["계열"] == selected_track_ui]
    if univ_search_kw.strip():
        clean_kw = normalize_univ_keyword(univ_search_kw)
        kw_mask = (
            filtered_univ_df["대학명"].str.contains(re.escape(clean_kw), na=False)
            | filtered_univ_df["검색용_대학명"].str.contains(re.escape(clean_kw), na=False)
            | filtered_univ_df["대학명"].str.contains(re.escape(univ_search_kw.strip()), na=False)
        )
        filtered_univ_df = filtered_univ_df[kw_mask]

    univ_options = sorted(filtered_univ_df["대학명"].dropna().unique().tolist())
    if not univ_options:
        univ_options = sorted(non_silgi_df["대학명"].dropna().unique().tolist())
        st.caption("⚠️ 해당 조건에 맞는 대학이 없어 전체 대학 목록을 표시합니다.")

    with f_col4:
        default_u_idx = univ_options.index("충남대학교(대전)") if "충남대학교(대전)" in univ_options else 0
        university = st.selectbox("🏫 4. 대학명(캠퍼스) 선택", options=univ_options, index=default_u_idx)

    s_col1, s_col2, s_col3 = st.columns([1.4, 1.5, 1.1])
    univ_jh_sub = jh_only_df[jh_only_df["대학명"] == university]
    if selected_track_ui != "전체 계열":
        univ_jh_track = univ_jh_sub[univ_jh_sub["계열"].isin([selected_track_ui, "자율·융합"])]
        if not univ_jh_track.empty:
            univ_jh_sub = univ_jh_track

    subtype_list = sorted(univ_jh_sub["세부유형"].dropna().unique().tolist())

    with s_col1:
        if subtype_list:
            subtype_options = ["전체 (해당 대학 학생부종합 통합)"] + subtype_list
            selected_subtype = st.selectbox(
                "📋 5. 학생부종합 세부전형 선택 (교과전형 제외)",
                options=subtype_options,
                index=0
            )
        else:
            selected_subtype = "기본 학생부종합 (본교 3단계 보정 적용)"
            st.selectbox(
                "📋 5. 학생부종합 세부전형 선택 (교과전형 제외)",
                options=[selected_subtype],
                disabled=True
            )

    with s_col2:
        if selected_subtype not in ["전체 (해당 대학 학생부종합 통합)", "기본 학생부종합 (본교 3단계 보정 적용)"]:
            major_pool_df = univ_jh_sub[univ_jh_sub["세부유형"] == selected_subtype]
        else:
            major_pool_df = univ_jh_sub if not univ_jh_sub.empty else non_silgi_df[non_silgi_df["대학명"] == university]

        univ_all_pool = non_silgi_df[non_silgi_df["대학명"] == university]
        if selected_track_ui != "전체 계열":
            major_pool_df = major_pool_df[major_pool_df["계열"] == selected_track_ui] if not major_pool_df.empty else major_pool_df
            univ_all_pool = univ_all_pool[univ_all_pool["계열"] == selected_track_ui]

        major_options = sorted(major_pool_df["모집단위"].dropna().unique().tolist())
        for m_item in sorted(univ_all_pool["모집단위"].dropna().unique().tolist()):
            if m_item not in major_options:
                major_options.append(m_item)

        custom_major_opt = "➕ [직접 입력] 목록에 없는 학과·약칭 직접 입력"
        selected_major_box = st.selectbox(
            "🎓 6. 모집단위(학과) 선택 (타이핑 검색 가능)",
            options=major_options + [custom_major_opt],
            index=0
        )
        if selected_major_box == custom_major_opt:
            major = st.text_input("직접 입력할 모집단위(학과명)", placeholder="예: 사회학과, 수학교육과, 컴공")
        else:
            major = selected_major_box

    with s_col3:
        student_name = st.text_input("👤 7. 학생 이름 (비식별화용)", placeholder="예: 홍길동")

else:
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.0, 1.2, 1.2, 0.9])
    with col1:
        university = st.text_input("희망 대학 (캠퍼스 포함 권장)", placeholder="예: 중앙대학교(서울), 충남대학교(대전)")
    with col2:
        selected_track_ui = st.selectbox("희망 계열", options=["자동 감지", "인문·사회", "자연·공학", "자율·융합"], index=0)
    with col3:
        selected_subtype = st.text_input("희망 학종 세부전형 (선택)", value="전체 (학생부종합 통합)", placeholder="예: CAU융합형인재")
    with col4:
        major = st.text_input("모집 단위(학과 - 약칭 자동 보정)", placeholder="예: 사회학과, 수학교육과, 컴공")
    with col5:
        student_name = st.text_input("학생 이름 (비식별화용)", placeholder="예: 홍길동")

st.markdown("#### 2️⃣ 희망 대학·세부전형 학생부종합 반영 비율 설정")
weight_mode = st.radio(
    "반영 비율 적용 방식 선택",
    options=[
        "🔍 선택한 대학·세부전형 기준 자동 매칭 (미확인 대학은 표준 4 : 4 : 2 자동 적용)",
        "🎛️ 슬라이더로 수동 비율 직접 입력"
    ],
    index=0,
    horizontal=True
)

if weight_mode.startswith("🔍"):
    w_acad, w_career, w_comm, weight_desc = get_auto_univ_weights(university, major, selected_subtype)
    st.info(f"📌 **자동 적용된 평가 비율:** 📘 학업역량 **{w_acad}%** : 📗 진로역량 **{w_career}%** : 📙 공동체역량 **{w_comm}%** — *({weight_desc})*")
else:
    st.caption("슬라이더를 움직여 학업·진로·공동체역량 비율을 직접 설정하세요 (기본값 40% : 40% : 20%).")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        w_acad = st.slider("📘 학업역량 반영 비율 (%)", min_value=10, max_value=70, value=40, step=5)
    with w_col2:
        w_career = st.slider("📗 진로역량 반영 비율 (%)", min_value=10, max_value=70, value=40, step=5)
    with w_col3:
        w_comm = 100 - (w_acad + w_career)
        if w_comm >= 5:
            st.metric("📙 공동체역량 반영 비율 (자동 계산)", f"{w_comm}%")
        else:
            st.error(f"학업+진로 합계가 95%를 초과했습니다! (현재 공동체역량: {w_comm}%) 슬라이더를 조절해주세요.")
    weight_desc = f"수동 설정 비율 ({w_acad} : {w_career} : {w_comm})"

st.markdown("#### 3️⃣ 학생 생기부 PDF 업로드")
u_col1, u_col2 = st.columns([2, 1])
with u_col1:
    uploaded_file = st.file_uploader("나이스 학생부 PDF 파일 업로드 (1~3학년 기록 학기 자동 감지)", type="pdf")
with u_col2:
    pdf_password = st.text_input("PDF 비밀번호 (생년월일 6자리 등, 없으면 비워둠)", type="password")

# ==========================================
# 6. 진단 시작 버튼 클릭 시 동작
# ==========================================
if st.button("🚀 생기부 심층 진단 및 마스터 리포트 생성하기", type="primary"):
    if not api_key:
        st.warning("API 키를 입력해주세요.")
    elif not uploaded_file:
        st.warning("생기부 PDF 파일을 업로드해주세요.")
    elif not university or not major:
        st.warning("희망 대학과 모집단위(학과)를 선택하거나 입력해주세요.")
    elif w_comm < 5:
        st.warning("공동체역량 비율이 최소 5% 이상이 되도록 학업·진로 비율을 조절해주세요.")
    else:
        with st.spinner("생기부 비식별화, 동일 계열(인문/자연) 합불 교차 보정, 엄격 루브릭 채점 및 마스터 진단표 산출 중입니다..."):
            try:
                raw_student_text = ""
                with pdfplumber.open(uploaded_file, password=pdf_password if pdf_password else None) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_student_text += page_text + "\n"

                safe_student_text = mask_personal_info(raw_student_text, student_name)
                bench_info = analyze_school_admission_benchmark(
                    adm_df, university, major, selected_subtype, selected_track_ui
                )
                detected_track = bench_info["detected_track"]

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name='gemini-3.8-flash',
                    generation_config={"max_output_tokens": 16384}
                )

                prompt = f"""
                당신은 {university} {major} ({selected_subtype}, [{detected_track} 계열])의 학생부종합전형 수석 입학사정관이자 냉철한 대입 진학 지도 전문가입니다.
                아래 제공된 [본교 최근 3개년 희망 대학·동일 계열({detected_track}) 학생부종합 합불 교차 보정 데이터(최근연도 5:3:2 가중 및 정밀 5등급제 환산 포함)], [본교 내 동일 계열({detected_track}) 타 대학 합격자 참고 데이터(종합+교과 통합)], 그리고 비식별화된 [학생 생활기록부 추출 내용]을 정밀 분석하여 **점수 인플레이션(칭찬 편향)이 완전히 배제된 냉정하고 객관적인 입학사정관 종합 진단 리포트**를 작성하세요.

                [핵심 지시사항 1: 현재 기록된 학기 자동 감지 및 '동일 계열({detected_track}) 합불 보정' 기반의 엄격한 성적 판정]
                - 업로드된 생기부 교과학습발달상황 표를 확인하여, 성적(석차등급)이 실제로 산출되어 있는 학기까지만 자동으로 감지하세요.
                - 학년별 비율 없이, 현재까지 석차등급 숫자가 존재하는 모든 과목의 **이수단위(시수)를 가중치로 반영한 누적 평균 등급**($\\frac{{\\sum(\\text{{단위수}} \\times \\text{{석차등급}})}}{{\\sum \\text{{단위수}}}}$)을 전체 교과, 주요 교과(국·영·수·사·과), 희망 전공({detected_track} - {major}) 핵심 교과로 나누어 정밀하게 제시하세요.
                  * 만약 [{detected_track}]이 '인문·사회' 계열이면 **국어·영어·수학·사회** 교과를 중심으로 분석하되 수학(통계·논리 기초)과 영어 원점수 하락 여부를 엄격히 점검하세요.
                  * 만약 [{detected_track}]이 '자연·공학' 계열이면 **수학·과학·영어·국어** 교과를 중심으로 분석하되 수학(미적분·기하·대수)과 과학 선택과목 위계 및 원점수를 엄격히 점검하세요.
                - **생존자 편향 및 소표본 착시 방지 (매우 중요):**
                  1) 특정 학과의 합격자가 1~2건뿐일 때 그 1건만 보고 합격률 100%라거나 '안정권'이라고 판정하는 것을 절대 금지합니다! 제공된 **동일 대학 [{detected_track}] 계열 전체의 실질 합격률(%)과 불합격자 상위권 성적, 보수적 권장선**을 기준으로 판정하세요.
                  2) 특히 서울·수도권 주요 상위권 대학(서·연·고·서·성·한·중·경·외·시·이·건·동·홍·숙 등) 및 의약학 계열의 학생부종합전형은 일반고에서 서류·면접 변수가 매우 크므로, 내신이 아무리 높아도 **절대 '🟢안정권'으로 판정하지 마세요! (최대 상한선: '🎯적정권' 또는 '🟡소신 경합권')**

                [핵심 지시사항 2: 점수 인플레이션 방지용 '엄격 감점 루브릭(Strict Rubric)']
                - 적용된 반영 기준: {weight_desc} (학업역량 {w_acad}% : 진로역량 {w_career}% : 공동체역량 {w_comm}%)
                - 9개 핵심 지표를 10점 만점으로 채점할 때 **9.3~9.8점 남발(점수 퍼주기)을 절대 금지**합니다! 아래 강제 감점 기준을 엄격히 적용하세요:
                  * **강제 감점 규칙 A (지필 성적 약점)**: 국·영·수·사·과 주요 과목 중 2등급 이하가 있거나 원점수가 85점 미만(특히 70점대~80점대 초반)인 과목이 1개라도 존재하면, [1. 학업성취도] 및 [5. 전공 관련 교과 성취도] 점수는 **최대 8.2 ~ 8.8점 이하**로 엄격히 제한하세요.
                  * **강제 감점 규칙 B (지필 성적과 세특 용어의 괴리)**: 지필고사 원점수가 70~80점대(2~3등급)인 과목에서 세특에만 대학 수준의 난해한 전문 용어(예: 페아노 공리계, 코시 평균값 정리, 편미분방정식 등)가 나열되어 있다면, 이를 무조건 칭찬하지 말고 [2. 학업태도] 및 [3. 교과 심화 탐구력]에서 **"지필 성적 대비 세특의 실질 체화도 의구심"**으로 지적하며 **8.0 ~ 8.7점 이하**로 감점하세요.
                  * **강제 감점 규칙 C (진로 일관성 및 봉사·리더십 한계)**: 학년별 진로 희망의 흔들림(예: 법조인/변호사 문구 혼재 등), 미이수 심화 과목 존재, 또는 봉사활동이 단순 환경정화·문단속 위주일 경우 [6. 진로 탐색], [8. 나눔과 배려], [9. 성실성·리더십]을 **7.8 ~ 8.6점 수준**으로 냉정하게 채점하세요.
                  * 즉, 전국 최상위 0.1% 완벽한 생기부가 아닌 이상 **100점 만점 환산 총점이 82점 ~ 90점 사이에서 변별력 있게 형성**되도록 엄격히 채점하세요.

                [핵심 지시사항 3: '핵심 역량 세부 요인 마스터 진단표 (30개 문항 체크리스트)' 작성]
                - 학업역량(10문항), 진로역량(10문항), 공동체역량(10문항) 총 **30개 세부 평가 요인**에 대해 `[매우우수 | 우수 | 보통 | 미흡 | 매우미흡]` 5단계 중 해당하는 칸 하나에만 **`●`** 기호를 표시하세요. (약점이 있는 항목에는 과감히 `우수`, `보통`, `미흡`에 `●`를 표시하여 변별력을 확보하세요.)

                [핵심 지시사항 4: 9개 세부 영역별 3단 심층 분석 (🔵강점 / 🔴감점 사유 / 🟢지도 제언)]
                - 9개 세부 영역 각각에 대해 다음 3단 구조로 **최소 8~10문장 이상** 매우 상세하게 분석하세요:
                  1) **🔵 강점 및 긍정적 기록 심층 분석 (최소 4~5문장 이상)**
                  2) **🔴 감점 사유 및 입학사정관 의구심 포인트 (최소 3~4문장 이상)**
                  3) **🟢 맞춤 지도 제언 및 핀셋 보완 전략 (2~3문장)**

                [핵심 지시사항 5: 차기 학기 세특·창체 로드맵 및 동일 계열({detected_track}) 추천 대학 포트폴리오(학종+교과 통합 6선)]
                - 5번 섹션에서는 다음 학기 교과 세특 심화 탐구 주제 3가지와 창체 나선형 연결 로드맵을 제시하세요.
                - 6번 섹션에서는 반드시 **[{detected_track}] 계열 내에서만** 학생부종합전형(학종)과 학생부교과전형(교과)을 모두 고려하여 **[🚀 상향·소신 추천 (2곳)]**, **[🎯 적정·맞춤 추천 (2곳)]**, **[🛡️ 안정·실속 추천 (2곳)]** 총 6곳과 유사 학과 확장 가이드를 제시하세요.

                - 수학·과학 수식이나 기호가 등장할 때는 깨지지 않도록 표준 LaTeX 문법인 $수식$ 형태로 작성하세요. 인사말 없이 아래 [출력 양식]을 정확히 지켜 출력하세요.

                ---
                [출력 양식]
                ## 📈 1. 생기부 자동 감지 내신 산출 및 [{university} {major}] ({detected_track} 계열) 학종 기준 성적 분석
                > - **감지된 생기부 반영 범위**: (예: 1학년 1학기 ~ 2학년 1학기까지 총 OO단위 이수 기준)
                > - **적용 전형 · 계열 및 데이터 보정 기준**: [{selected_subtype}] | {bench_info['major_notice']}
                > - **학생 시수 반영 누적 내신 (5등급제)**:
                >   * **전체 전교과 평균**: **O.OO 등급** (학기별 추이 설명)
                >   * **국·영·수·사·과 주요 교과 평균**: **O.OO 등급**
                >   * **희망 전공({major} - {detected_track}) 핵심 교과 평균**: **O.OO 등급** (주요 과목 원점수·평균·이수자수 및 취약 과목 원점수 명시)
                > - **[{university} {major} ({selected_subtype})] 합불 보정 기준 엄격 판정**: **[🎯적정권(유력) / 🟡소신 경합권 / 🔴상향 도전권(역전필요) / 🟢안정권(거점국립·지역대학 한정) 중 택1]**
                >   * (본교 3개년 동일 계열({detected_track}) 최근가중(5:3:2) 학종 합격선, 동일 계열 실질 합격률, 보수적 권장선 및 상위권 대학 정성 평가 변수를 결합하여 3~4문장으로 냉정하고 상세하게 서술)

                ## 🏆 2. 대학·전형 반영비율({w_acad}:{w_career}:{w_comm}) 엄격 채점 점수표 및 입학사정관 최종 통합 의견
                > 📌 **적용 기준**: {weight_desc} (학업 {w_acad}% : 진로 {w_career}% : 공동체 {w_comm}% | 엄격 감점 루브릭 적용)

                | 평가 영역 | 세부 평가 항목 (9대 핵심 지표) | 원점수 (10점 만점) | 영역 소계 (30점 만점) | 대학 반영비율 환산 점수 |
                | :--- | :--- | :---: | :---: | :---: |
                | **📘 학업역량 ({w_acad}%)** | 1. 학업성취도 및 이수 성실성 | O.O / 10 | **OO.O / 30** | **OO.O / {w_acad}점** |
                | **📘 학업역량 ({w_acad}%)** | 2. 학업태도 및 자기주도성 | O.O / 10 | ↑ | ↑ |
                | **📘 학업역량 ({w_acad}%)** | 3. 교과 심화 탐구력 | O.O / 10 | ↑ | ↑ |
                | **📗 진로역량 ({w_career}%)** | 4. 전공 관련 교과 이수 노력 | O.O / 10 | **OO.O / 30** | **OO.O / {w_career}점** |
                | **📗 진로역량 ({w_career}%)** | 5. 전공 관련 교과 성취도 | O.O / 10 | ↑ | ↑ |
                | **📗 진로역량 ({w_career}%)** | 6. 진로 탐색 및 창체·세특 연계 | O.O / 10 | ↑ | ↑ |
                | **📙 공동체역량 ({w_comm}%)** | 7. 협업과 소통 능력 | O.O / 10 | **OO.O / 30** | **OO.O / {w_comm}점** |
                | **📙 공동체역량 ({w_comm}%)** | 8. 나눔과 배려 실천 | O.O / 10 | ↑ | ↑ |
                | **📙 공동체역량 ({w_comm}%)** | 9. 성실성·규칙 준수 및 리더십 | O.O / 10 | ↑ | ↑ |
                | **🌟 최종 종합 총점** | **9개 항목 종합 ({university} {major} 맞춤 환산)** | - | **OO.O / 90** | **🎯 OO.O / 100점** |

                ### 🔍 입학사정관 영역별 최종 통합 의견 (Executive Summary)
                > - **[학업성취도]**: (2~3문장 핵심 요약 — 강점 과목과 원점수/등급 하락 과목을 동시에 냉정히 짚을 것)
                > - **[학업태도]**: (2~3문장 핵심 요약)
                > - **[탐구력]**: (2~3문장 핵심 요약)
                > - **[전공이수노력]**: (2~3문장 핵심 요약)
                > - **[전공성취도]**: (2~3문장 핵심 요약)
                > - **[진로탐색경험]**: (2~3문장 핵심 요약)
                > - **[협업과 소통]**: (2~3문장 핵심 요약)
                > - **[나눔과 배려]**: (2~3문장 핵심 요약)
                > - **[성실성과 규칙·리더십]**: (2~3문장 핵심 요약)

                ## 📊 3. 핵심 역량 세부 요인 마스터 진단표 (30개 평가 요인 5단계 체크리스트)
                *(각 문항별로 학생의 실제 생기부 기록에 부합하는 등급 칸 **단 하나에만 `●` 기호**를 표시하고 나머지 칸은 비워두세요.)*

                | 평가 영역 | 세부 평가 요인 (30대 핵심 체크리스트) | 매우우수 | 우수 | 보통 | 미흡 | 매우미흡 |
                | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
                | **학업성취도** | 1. 전체적인 교과 성적은 지원 대학 합격자 평균 및 권장선 대비 우수한가? | | | | | |
                | **학업성취도** | 2. 학기별·학년별 성적이 큰 기복 없이 고르게 유지 또는 우상향하고 있는가? | | | | | |
                | **학업성취도** | 3. 대학 수학에 필요한 기본 교과(국·영·수·사·과) 성적 균형이 잡혀 있는가? | | | | | |
                | **학업성취도** | 4. 과목별 등급 외에 원점수(평균·표준편차·이수자수 고려) 경쟁력이 탁월한가? | | | | | |
                | **학업태도** | 5. 새로운 지식을 획득하기 위해 자기주도적으로 학습하려는 태도가 뚜렷한가? | | | | | |
                | **학업태도** | 6. 교과 수업에 흐트러짐 없이 몰입하고 과제를 끝까지 완수하는 끈기가 있는가? | | | | | |
                | **학업태도** | 7. 💡 지적 끈기와 성실한 수업 태도가 여러 과목 교사들의 세특에서 일관되게 교차 검증되는가? | | | | | |
                | **탐구력** | 8. 교과 내 실험·토론·보고서 작성 등 탐구 활동에 주도적으로 참여하고 있는가? | | | | | |
                | **탐구력** | 9. 💡 수업 내용의 단순 수용에 그치지 않고 독서·실험·후속 보고서로 '연쇄적인 심화 탐구'를 확장했는가? | | | | | |
                | **탐구력** | 10. 💡 교과 지식을 타 교과와 연결하는 '융합적 사고력' 및 실생활·최신 기술에 적용하는 통찰력이 있는가? | | | | | |
                | **전공이수노력** | 11. 희망 전공(계열)에 필수적인 권장 선택과목들을 빠짐없이 선택하여 이수했는가? | | | | | |
                | **전공이수노력** | 12. 등급 확보가 불리한 소인수·심화 과목이나 공동교육과정에 과감히 도전한 흔적(학업적 용기)이 있는가? | | | | | |
                | **전공이수노력** | 13. 선택과목(일반/진로/융합)을 교과목 학습 단계(위계)에 따라 체계적으로 이수했는가? | | | | | |
                | **전공성취도** | 14. 지원 전공 관련 핵심 교과의 석차등급/성취도, 원점수, 평균 대비 성취 수준이 탁월한가? | | | | | |
                | **전공성취도** | 15. 전공 관련 일반선택과목 대비 진로·융합선택과목의 성취 수준과 탐구 밀도가 일관되게 우수한가? | | | | | |
                | **전공성취도** | 16. 💡 전공 관련 과목의 지필평가 성적(정량)과 세특에 기록된 탐구의 깊이(정성)가 괴리 없이 일치하는가? | | | | | |
                | **진로탐색경험** | 17. 교과 세특 활동에서 지원 전공(계열)에 대한 뚜렷한 문제의식과 탐색 경험이 드러나는가? | | | | | |
                | **진로탐색경험** | 18. 창의적 체험활동(자율·동아리·진로)과 전공 관련 독서 활동이 유기적으로 연결되어 있는가? | | | | | |
                | **진로탐색경험** | 19. 💡 1학년의 폭넓은 관심사가 학년이 올라갈수록 특정 심화 주제로 구체화·전문화되는 성장이 보이는가? | | | | | |
                | **진로탐색경험** | 20. 💡 단순 조사·발표를 넘어 직접 대안을 제시하거나 실험·설계로 구현해 낸 '능동적 문제 해결 경험'이 있는가? | | | | | |
                | **협업과 소통** | 21. 모둠 실험·융합 프로젝트에서 자신의 구체적인 역할과 실질적 기여도가 명확히 드러나는가? | | | | | |
                | **협업과 소통** | 22. 타인의 의견을 경청·수용하여 자신의 생각을 발전시키거나 동료의 학습을 도운 쌍방향 협력 사례가 있는가? | | | | | |
                | **협업과 소통** | 23. 💡 팀 활동 중 의견 충돌이나 난관이 발생했을 때 논리적·창의적 방식으로 중재하여 목표를 달성했는가? | | | | | |
                | **나눔과 배려** | 24. 학급 및 학교생활 속에서 타인을 위해 양보하거나 배려를 실천하고 생활화한 구체적 기록이 있는가? | | | | | |
                | **나눔과 배려** | 25. 💡 봉사활동이 단순 환경미화·문단속 등 일회성 의무 봉사에 그치지 않고 진로·공동체와 연계해 지속되었는가? | | | | | |
                | **성실성과 규칙** | 26. 출결 상황(미인정 지각·결석 유무) 및 교칙 준수 등 기본 생활 습관이 모범적으로 유지되었는가? | | | | | |
                | **성실성과 규칙** | 27. 교내 1인 1역, 학급 당번 등 자신이 맡은 공동체 내 역할에 끝까지 책임감을 다한 기록이 있는가? | | | | | |
                | **리더십** | 28. 공동체의 목표 달성을 위해 계획을 수립하고 실행을 주도적으로 이끈 경험이 있는가? | | | | | |
                | **리더십** | 29. 💡 공식 직책 유무와 관계없이 소외된 구성원을 다독이거나 공동체에 긍정적 영향력을 행사했는가? | | | | | |
                | **리더십** | 30. 💡 [문제 인식 ➔ 주도적 제안 ➔ 협업 및 실행 ➔ 공동체의 실질적 변화]로 이어지는 완결된 서사가 있는가? | | | | | |

                ## 🔍 4. 9개 세부 영역별 점수 판단 근거 및 3단 심층 분석 (강점 / 감점 사유 / 지도 제언)
                *(주의: 1번부터 9번 항목까지 절대 생략하거나 짧게 요약하지 마세요. 각 항목마다 과목명·원점수·세특 문맥을 인용하여 최소 8~10문장 이상 상세히 작성하세요.)*
                ### 📌 [항목 번호 및 이름] — **O.O / 10점**
                > - **🔵 강점 및 긍정적 기록 심층 분석**: (최소 4~5문장 이상. 학기별 과목 성취도(원점수/평균/표준편차/이수자수/등급)와 세특·창체에 기록된 구체적 탐구 주제·원리·수식·활동 과정을 상세히 인용하고 입학사정관이 높이 평가할 포인트를 서술)
                > - **🔴 감점 사유 및 입학사정관 의구심 포인트**: (최소 3~4문장 이상. 성적 추이상 약점, 지필 성적과 세특 난이도의 괴리 여부, 미이수 심화 과목, 단순 정리식 서술이나 아쉬운 기록 방식을 날카롭게 지적하여 왜 만점에서 감점되었는지 명확히 설명)
                > - **🟢 맞춤 지도 제언 및 핀셋 보완 전략**: (2~3문장. 남은 학기 세특이나 창체, 또는 면접에서 이 항목의 감점을 만회하기 위해 수행해야 할 구체적 활동 및 기록 방향 제시)

                ## 💡 5. 다음 학기 생기부 최종 보완 처방전 (세특·자율·동아리·진로 Action Plan)
                > 🎯 **최우선 보완 타깃 항목**: (9개 항목 중 가장 시급히 보완해야 할 2~3개 항목과 핵심 이유)
                >
                > **1) [전공 관련 선택과목 및 교과 세특 심화 로드맵]**
                > - (남아있는 학기에 어떤 과목 세특에서 기존 탐구 키워드를 어떻게 확장할지 구체적인 심화 탐구 주제 3가지를 **[과목명 - 추천 탐구 주제 - 핵심 학술 원리 및 세특 작성 포인트]** 형태로 상세히 제시)
                >
                > **2) [창의적 체험활동(자율·동아리·진로·봉사) 나선형 연결 전략]**
                > - (기존 창체 활동의 한계를 극복하고 전공 역량과 공동체 역량을 동시에 입증할 수 있는 구체적인 프로젝트·실험·독서·산출물 아이디어 제시)
                >
                > **3) [지필 성적 관리 및 세특 진정성(괴리 해소) 확보 전략]**
                > - (성적 추이 보완 목표치 및 지필 성적 수준에 부합하는 설득력 있는 세특 구성 전략)

                ## 🧭 6. 현재 학생 상황 맞춤 [{detected_track}] 계열 추천 대학 및 유사 학과 포트폴리오 (학종·교과 통합 6선)
                > - **희망 대학([{university} {major} - {selected_subtype}]) 적합도 및 수시 6장 전형 전략 총평**: (현재 학생의 5등급제 내신과 생기부 역량을 종합할 때 희망 대학 지원이 적절한지, 그리고 학생부종합전형과 학생부교과전형 중 어떤 비중으로 수시 6장 포트폴리오를 짜는 것이 유리한지 2~3문장으로 진단)

                | 지원 전략 구분 | 추천 대학명 (캠퍼스 명시) | 추천 모집단위 ({detected_track} 동일·유사 학과) | 추천 전형유형 (학생부종합 / 교과) | 예상 합격권 (5등급제 환산 기준) | 학생 생기부·내신 기준 전형별 추천 사유 및 유불리 분석 |
                | :---: | :--- | :--- | :---: | :---: | :--- |
                | **🚀 상향·소신** | (대학 1) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (학생의 어떤 세특 강점 또는 내신 구조로 도전 가능한지 구체적 이유) |
                | **🚀 상향·소신** | (대학 2) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (구체적 추천 이유 및 보완 조건) |
                | **🎯 적정·맞춤** | (대학 3) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (현재 성적·세특 수준에서 가장 합격 확률이 높은 이유) |
                | **🎯 적정·맞춤** | (대학 4) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (본교 입결 데이터 및 학생 역량 매칭 분석) |
                | **🛡️ 안정·실속** | (대학 5) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (내신·최저 변수 대비 안전하게 확보 가능한 근거) |
                | **🛡️ 안정·실속** | (대학 6) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (본교 다수 합격 사례 기반 추천 근거) |

                > - **📌 [{detected_track}] 유사 학과 확장 가이드**: (학생이 희망하는 [{major}] 외에, 현재 생기부에 적힌 탐구 주제들을 그대로 활용하여 지원할 수 있는 동일 계열 유사 학과·융합 학과 3~4개를 추천하고 그 이유를 상세히 설명)

                ---
                [본교 최근 3개년(2024~2026 대입) 희망 대학·동일 계열({detected_track})·세부전형 '학생부종합' 합불 교차 보정 데이터]
                {bench_info['prompt_context']}

                [본교 최근 3개년 동일 계열({detected_track}) 타 대학 합격자 참고 데이터 (학생부종합 + 학생부교과 통합, 실기 제외)]
                {bench_info['similar_majors_context']}

                [학생 생활기록부 추출 내용]
                {safe_student_text}
                """

                max_retries = 3
                retry_delay = 12
                response = None

                for attempt in range(max_retries):
                    try:
                        response = model.generate_content(prompt)
                        break
                    except Exception as api_err:
                        if attempt < max_retries - 1:
                            st.toast(f"일시적 한도 지연이 발생하여 {retry_delay}초 후 자동 재시도합니다... ({attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise api_err

                usage = getattr(response, "usage_metadata", None)
                if usage:
                    p_tok = getattr(usage, "prompt_token_count", 0)
                    o_tok = getattr(usage, "candidates_token_count", 0)
                    t_tok = getattr(usage, "total_token_count", p_tok + o_tok)
                    st.session_state.token_info = {"prompt": p_tok, "output": o_tok, "total": t_tok}
                else:
                    st.session_state.token_info = None

                st.session_state.eval_report = response.text
                st.session_state.safe_student_text = safe_student_text
                st.session_state.target_title = f"{university}_{selected_subtype}_{major}"
                st.session_state.quant_summary_md = bench_info["summary_md"]
                st.session_state.major_notice_md = bench_info["major_notice"]
                st.session_state.applied_weights_info = f"[{detected_track}] 학업 {w_acad}% : 진로 {w_career}% : 공동체 {w_comm}% ({weight_desc})"

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# ==========================================
# 7. 결과 출력 및 다운로드 구역
# ==========================================
if st.session_state.eval_report:
    st.success(f"개인정보 보호 처리 및 심층 분석이 완료되었습니다! [{st.session_state.target_title.replace('_', ' ')}] 진단 리포트를 출력합니다.")

    if st.session_state.major_notice_md:
        st.info(f"🔎 **계열 및 모집단위 매칭 안내:** {st.session_state.major_notice_md}")

    if st.session_state.token_info:
        t_info = st.session_state.token_info
        tpm_percent = round((t_info["total"] / 250000) * 100, 1)
        st.info(
            f"📊 **이번 작업 토큰 소모량:** 총 **{t_info['total']:,} 토큰** "
            f"(입력: {t_info['prompt']:,} / 출력: {t_info['output']:,}) "
            f"— 분당 토큰 한도(25만)의 약 **{tpm_percent}%** 사용 | "
            f"**적용 기준:** {st.session_state.applied_weights_info}"
        )

    full_combined_report = (
        "## 🏫 [참고] 본교 최근 3개년(2024~2026 대입) 동일 계열 학생부종합전형 입시결과 (최근연도 5:3:2 가중 및 합불 보정)\n"
        + st.session_state.quant_summary_md
        + "\n\n---\n\n"
        + st.session_state.eval_report
    )

    st.markdown("#### 📄 진단 리포트 저장하기 (표·마스터진단표·수식 유지)")
    pdf_download_component(
        result_text=full_combined_report,
        title_prefix=st.session_state.target_title
    )

    st.divider()

    with st.expander("🏫 본교 최근 3개년(2024~2026 대입) 해당 대학·동일 계열 [학생부종합전형] 가중 입결 및 합격 사례 보기", expanded=True):
        st.markdown(st.session_state.quant_summary_md)

    st.divider()

    st.markdown(st.session_state.eval_report)

    st.download_button(
        label="📄 일반 텍스트로 백업 다운로드 (.txt)",
        data=full_combined_report,
        file_name=f"{st.session_state.target_title}_생기부진단리포트.txt",
        mime="text/plain"
    )

    with st.expander("보안 처리된 생기부 텍스트 확인하기 (클릭)"):
        st.text(st.session_state.safe_student_text[:1000] + "\n\n... (중략) ...")
