import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
import json
import time

st.set_page_config(page_title="생기부 역량 평가 및 보완점 진단기", page_icon="📊", layout="wide")

# ==========================================
# 0. 세션 상태 초기화 (저장 버튼 클릭 시 화면 초기화 방지)
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

# ==========================================
# 1. 지족고 3개년 입시결과 CSV 불러오기 및 5등급제 환산 함수
# ==========================================
g9_knots = np.array([1.00, 1.50, 1.85, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00])
g5_knots = np.array([1.00, 1.20, 1.38, 1.42, 1.65, 1.90, 2.15, 2.40, 3.00, 3.55, 4.10, 4.60, 5.00])

def convert_9_to_5(g9):
    return round(float(np.interp(g9, g9_knots, g5_knots)), 2)

@st.cache_data
def load_admission_data():
    try:
        df = pd.read_csv("jijok_admission_data.csv")
        if "내신등급_5등급환산" not in df.columns and "내신등급" in df.columns:
            df["내신등급_5등급환산"] = df["내신등급"].apply(convert_9_to_5)
        return df
    except Exception:
        return None

adm_df = load_admission_data()

# ==========================================
# 2. 개인정보 비식별화(마스킹) 함수
# ==========================================
def mask_personal_info(text, student_name=""):
    text = re.sub(r'\d{6}[-\s]*[1-4]\d{6}', '******-*******', text)
    text = re.sub(r'01[016789][-\s]*\d{3,4}[-\s]*\d{4}', '010-****-****', text)
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '***@***.***', text)
    if student_name and len(student_name.strip()) >= 2:
        text = text.replace(student_name.strip(), "OOO")
    return text

# ==========================================
# 3. 대학명 표준화 및 대학별 학종 반영 비율 자동 매칭 (미확인 시 4:4:2)
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

def get_auto_univ_weights(univ_input, major_input=""):
    """
    입력된 대학명과 모집단위를 바탕으로 학종 3대 역량(학업:진로:공동체) 반영 비율을 자동 산출합니다.
    특정 비율이 확인되지 않는 일반 대학은 기본 40 : 40 : 20 (4:4:2)으로 설정합니다.
    """
    if not univ_input.strip():
        return 40, 40, 20, "기본 표준 비율 (4 : 4 : 2)"

    u_key = normalize_univ_keyword(univ_input)
    m_key = major_input.strip()

    # 1) 교대·사범대 등 교직/인성 비중이 높은 대학·학과
    if any(k in u_key for k in ["교육", "교원"]) or "교육" in m_key or "사범" in m_key:
        return 35, 40, 25, f"{univ_input} 교원·사범계열 학종 기준 (3.5 : 4.0 : 2.5 - 공동체·교직인성 강화)"

    # 2) 학업·탐구역량 중심형 주요 대학
    academic_heavy = {
        "서울": (50, 30, 20, "서울대 일반/지균 학업·탐구 중심 기준 (5 : 3 : 2)"),
        "고려": (50, 30, 20, "고려대 학업우수형/계열적합 평균 기준 (5 : 3 : 2)"),
        "서강": (50, 30, 20, "서강대 일반형 학업역량 중심 기준 (5 : 3 : 2)"),
        "성균관": (50, 30, 20, "성균관대 융합형/탐구형 평균 기준 (5 : 3 : 2)"),
        "한양": (50, 35, 15, "한양대 추천형/서류형 학업·진로 중심 기준 (5.0 : 3.5 : 1.5)"),
        "이화여자": (50, 30, 20, "이화여대 미래인재 학업 중심 기준 (5 : 3 : 2)"),
        "과학기술원": (50, 35, 15, "과기원(KAIST/GIST/DGIST/UNIST) 학업·탐구 중심 기준 (5.0 : 3.5 : 1.5)"),
        "광주과학기술원": (50, 35, 15, "GIST 학업·연구역량 중심 기준 (5.0 : 3.5 : 1.5)"),
        "대구경북과학기술원": (50, 35, 15, "DGIST 학업·연구역량 중심 기준 (5.0 : 3.5 : 1.5)")
    }
    for k, val in academic_heavy.items():
        if k in u_key:
            return val

    # 3) 진로역량(전공적합성) 중심형 대학
    career_heavy = {
        "연세": (30, 50, 20, "연세대 활동우수형 진로·학업 반영 기준 (3 : 5 : 2)"),
        "경희": (30, 50, 20, "경희대 네오르네상스전형 공식 반영비율 (3 : 5 : 2)"),
        "건국": (30, 50, 20, "건국대 KU자기추천전형 공식 반영비율 (3 : 5 : 2)"),
        "동국": (30, 50, 20, "동국대 Do Dream전형 진로역량 중심 기준 (3 : 5 : 2)"),
        "서울시립": (35, 45, 20, "서울시립대 학생부종합Ⅰ 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "중앙": (35, 45, 20, "중앙대 다빈치/탐구형 평균 기준 (3.5 : 4.5 : 2.0)"),
        "세종": (35, 45, 20, "세종대 창의인재전형 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "숙명여자": (35, 45, 20, "숙명여대 숙명인재 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "서울과학기술": (35, 45, 20, "서울과기대 학교생활우수자/창의융합 기준 (3.5 : 4.5 : 2.0)"),
        "숭실": (35, 45, 20, "숭실대 SSU미래인재 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "국민": (35, 45, 20, "국민대 국민프런티어 진로역량 중심 기준 (3.5 : 4.5 : 2.0)"),
        "단국": (35, 45, 20, "단국대 DKU인재 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "아주": (35, 45, 20, "아주대 에이스전형 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "인하": (35, 45, 20, "인하대 인하미래인재 진로·학업 기준 (3.5 : 4.5 : 2.0)"),
        "한국기술교육": (35, 45, 20, "한국기술교육대 창의인재 진로·실천 중심 기준 (3.5 : 4.5 : 2.0)")
    }
    for k, val in career_heavy.items():
        if k in u_key:
            return val

    # 4) 거점국립대 및 기타 일반 대학 -> 표준 4 : 4 : 2 자동 적용
    return 40, 40, 20, f"{univ_input} 학생부종합 표준 균형 반영 비율 (4 : 4 : 2 자동 적용)"

# ==========================================
# 4. 지족고 3개년 입결 검색 (희망대학=학종 전용 / 추천대학=종합+교과 모두 고려)
# ==========================================
def extract_major_root_keywords(major_input):
    m = major_input.strip()
    keyword_groups = [
        ["수학", "통계", "수리", "데이터"],
        ["컴퓨터", "소프트웨어", "인공지능", "AI", "정보", "보안", "전산", "스마트", "IT"],
        ["기계", "자동차", "항공", "우주", "로봇", "메카트로닉스", "자율운항", "조선"],
        ["전기", "전자", "반도체", "정보통신", "제어", "광학"],
        ["화학", "화공", "신소재", "고분자", "재료", "에너지", "나노"],
        ["생명", "생물", "바이오", "유전", "식품", "농업", "원예", "산림", "수의"],
        ["물리", "천문", "지구", "환경", "기상", "해양"],
        ["건축", "토목", "도시", "건설", "조경", "실내"],
        ["간호", "보건", "물리치료", "임상병리", "작업치료", "방사선", "응급", "치위생", "의예", "약학", "한의"],
        ["교육", "사범", "초등"],
        ["경영", "경제", "금융", "회계", "무역", "통상", "물류", "비즈니스"],
        ["행정", "정치", "외교", "법학", "경찰", "사회", "복지", "심리", "언론", "미디어", "광고"],
        ["국어", "국문", "영어", "영문", "일어", "중어", "불어", "독어", "언어", "문헌", "사학", "역사", "철학"]
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

def analyze_school_admission_benchmark(df, univ_input, major_input):
    if df is None or df.empty:
        return {
            "match_level": "데이터 없음",
            "summary_md": "⚠️ `jijok_admission_data.csv` 파일을 찾을 수 없어 일반 전국 학생부종합 기준으로 진단합니다.",
            "prompt_context": "본교 3개년 입결 파일이 연결되지 않았습니다.",
            "similar_majors_context": "본교 데이터 없음"
        }

    u_key = normalize_univ_keyword(univ_input)
    m_key = major_input.strip().replace("학과", "").replace("학부", "").replace("전공", "").strip()

    # ★ 핵심: 희망 대학·학과 분석은 오직 '학생부위주(종합)' 데이터만 필터링하여 기준선 산출 ★
    jonghap_all_df = df[df['전형유형'].str.contains("종합", na=False)].copy()

    univ_mask = jonghap_all_df['대학명'].str.contains(u_key, na=False)
    if '검색용_대학명' in jonghap_all_df.columns:
        univ_mask = univ_mask | jonghap_all_df['검색용_대학명'].str.contains(u_key, na=False)
    univ_jonghap_df = jonghap_all_df[univ_mask]

    if not univ_jonghap_df.empty and m_key:
        exact_jonghap_df = univ_jonghap_df[univ_jonghap_df['모집단위'].str.contains(m_key, na=False)]
    else:
        exact_jonghap_df = pd.DataFrame()

    # 학종 기준 3단계 폴백(Fallback) 판정
    if not exact_jonghap_df.empty:
        target_df = exact_jonghap_df
        match_level = f"✅ 1단계 정확 일치 (학생부종합): 본교 3개년 [{univ_input} - {major_input}] 학종 합격 사례 ({len(target_df)}건)"
    elif not univ_jonghap_df.empty:
        target_df = univ_jonghap_df
        match_level = f"🔄 2단계 동일 대학 학종 보정: 본교 내 [{major_input}] 학종 직접 사례가 없어 [{univ_input} 학생부종합 전체 합격 사례 ({len(target_df)}건)] 기준으로 보정"
    else:
        target_df = pd.DataFrame()
        match_level = f"🌐 3단계 전국 학종 기준 보정: 본교 3개년 데이터 내 [{univ_input}] 학생부종합 합격 사례가 없어, 해당 대학의 전국 학종 입결 기준 및 본교 학종 전체 평균(9등급제 3.85등급 ➔ 5등급제 환산 2.33등급)으로 보정"

    # ★ 추천 대학 산출용 데이터: 유사 학과의 '학생부위주(종합)' + '학생부위주(교과)' 모두 추출 ★
    sim_keywords = extract_major_root_keywords(major_input)
    valid_types_df = df[df['전형유형'].isin(["학생부위주(종합)", "학생부위주(교과)"])].copy()
    if sim_keywords:
        pattern = "|".join([re.escape(k) for k in sim_keywords])
        sim_df = valid_types_df[valid_types_df['모집단위'].str.contains(pattern, na=False)].copy()
    else:
        sim_df = pd.DataFrame()

    similar_lines = []
    if not sim_df.empty:
        sim_grouped = sim_df.groupby(['대학명', '모집단위', '전형유형']).agg(
            cnt=('내신등급', 'count'),
            mean_9=('내신등급', 'mean'),
            mean_5=('내신등급_5등급환산', 'mean'),
            min_5=('내신등급_5등급환산', 'min'),
            max_5=('내신등급_5등급환산', 'max')
        ).reset_index().sort_values(by=['전형유형', 'mean_5'])

        # 종합전형 사례와 교과전형 사례를 골고루 포함하여 프롬프트에 전달
        jonghap_sims = sim_grouped[sim_grouped['전형유형'] == '학생부위주(종합)'].head(15)
        gyogwa_sims = sim_grouped[sim_grouped['전형유형'] == '학생부위주(교과)'].head(15)
        combined_sims = pd.concat([jonghap_sims, gyogwa_sims])

        for _, r in combined_sims.iterrows():
            similar_lines.append(
                f"- {r['대학명']} | {r['모집단위']} [{r['전형유형']}, {int(r['cnt'])}건]: 5등급환산 평균 {r['mean_5']:.2f}등급 (범위 {r['min_5']:.2f}~{r['max_5']:.2f} / 9등급제 {r['mean_9']:.2f}등급)"
            )
    similar_majors_context = "\n".join(similar_lines) if similar_lines else "본교 내 동일 키워드 학과 사례가 적어 전국 대학 기준(종합·교과 모두 포함)으로 추천하세요."

    if target_df.empty:
        return {
            "match_level": match_level,
            "summary_md": f"**{match_level}**\n\n* 본교 최근 3개년(2024~2026 대입) 수시 데이터에 해당 대학의 **학생부종합전형** 합격 사례가 없어, 해당 대학·학과의 일반적인 학종 합격선과 본교 학종 합격자 평균(9등급제 3.85등급 ➔ **5등급제 환산 2.33등급**)을 바탕으로 보정 평가합니다.",
            "prompt_context": f"본교 3개년 데이터에 {univ_input} {major_input} 학생부종합전형 직접 사례가 없습니다. 해당 대학·학과의 일반적인 학생부종합전형 수준과 본교 전체 학종 합격 평균(9등급제 3.85등급 / 5등급제 환산 2.33등급)을 기준으로 성적을 분석하세요.",
            "similar_majors_context": similar_majors_context
        }

    # 희망 대학 학생부종합전형 요약 표 생성
    rows_md = []
    prompt_lines = [f"[{match_level}]"]

    # 전체 학종 합계 및 단계별(최초합/추합) 통계
    for stage, stage_label in [("합격", "최초합격"), ("충원합격", "충원합격(추합)")]:
        st_sub = target_df[target_df['최종단계'] == stage]
        if st_sub.empty:
            continue
        cnt = len(st_sub)
        mean_9 = st_sub['내신등급'].mean()
        min_9 = st_sub['내신등급'].min()
        max_9 = st_sub['내신등급'].max()
        mean_5 = convert_9_to_5(mean_9)
        min_5 = convert_9_to_5(min_9)
        max_5 = convert_9_to_5(max_9)
        rows_md.append(
            f"| **학생부위주(종합)** | {stage_label} | {cnt}건 | **{mean_5:.2f} 등급** *(범위: {min_5:.2f}~{max_5:.2f})* | {mean_9:.2f} 등급 *(범위: {min_9:.2f}~{max_9:.2f})* |"
        )
        prompt_lines.append(
            f"- 학생부위주(종합) [{stage_label}, {cnt}건]: 9등급제 평균 {mean_9:.2f}등급(최고 {min_9:.2f}~최저 {max_9:.2f}) -> 현행 5등급제 환산 평균 {mean_5:.2f}등급(범위 {min_5:.2f}~{max_5:.2f}등급)"
        )

    # 최근 학종 합격 사례 상세 최대 5건 ('최저 없음/미기재' 문구 완전 제거, 세부전형명 표시)
    sample_cases = target_df[['대입학년도', '대학명', '세부유형', '모집단위', '최종단계', '내신등급', '내신등급_5등급환산']].head(5)
    case_lines = []
    for _, r in sample_cases.iterrows():
        sub_type = str(r['세부유형']).strip() if pd.notna(r['세부유형']) and str(r['세부유형']).strip() else "학생부종합"
        case_lines.append(
            f"  * [{int(r['대입학년도'])}대입] {r['모집단위']} ({sub_type} - {r['최종단계']}): **5등급환산 {r['내신등급_5등급환산']:.2f}등급** (9등급제 {r['내신등급']:.2f})"
        )

    table_header = (
        f"**{match_level}**\n\n"
        "| 전형유형 (학종 전용) | 합격단계 | 본교 건수 | **현행 5등급제 환산 평균 (범위)** | 선배 9등급제 원본 평균 (범위) |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    summary_md = table_header + "\n".join(rows_md) + "\n\n**📌 본교 최근 학생부종합 주요 합격 사례:**\n" + "\n".join(case_lines)
    prompt_context = "\n".join(prompt_lines) + "\n" + "\n".join(case_lines)

    return {
        "match_level": match_level,
        "summary_md": summary_md,
        "prompt_context": prompt_context,
        "similar_majors_context": similar_majors_context
    }

# ==========================================
# 5. 표·수식·인용구 완벽 지원 PDF 다운로드 컴포넌트
# ==========================================
def pdf_download_component(result_text: str, title_prefix: str):
    safe_text = json.dumps(result_text, ensure_ascii=False)
    safe_filename = json.dumps(f"{title_prefix}_생기부종합진단리포트.pdf", ensure_ascii=False)
    safe_title = json.dumps(f"{title_prefix} 생기부 역량 평가 및 추천대학 진단 리포트", ensure_ascii=False)

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
                line-height: 1.68; font-size: 10.5pt; background: #ffffff;
            }}
            #pdf-render-area .doc-header {{
                text-align: center; font-size: 17pt; font-weight: 700; color: #1E3A8A;
                border-bottom: 2px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px;
            }}
            #pdf-render-area h2, #pdf-render-area h3 {{
                color: #1E3A8A; margin-top: 18px; margin-bottom: 8px; page-break-after: avoid;
            }}
            #pdf-render-area table {{
                width: 100%; border-collapse: collapse; margin: 12px 0 18px 0;
                font-size: 9.5pt; page-break-inside: avoid;
            }}
            #pdf-render-area th, #pdf-render-area td {{
                border: 1px solid #CBD5E1; padding: 7px 10px; text-align: left;
            }}
            #pdf-render-area th {{ background-color: #EFF6FF; color: #1E3A8A; font-weight: 700; }}
            #pdf-render-area blockquote {{
                background-color: #F8FAFC; border-left: 4px solid #3B82F6;
                margin: 0 0 14px 0; padding: 10px 14px; border-radius: 4px;
                page-break-inside: avoid; break-inside: avoid;
            }}
            #pdf-render-area blockquote p {{ margin: 5px 0; }}
            #pdf-render-area hr {{ border: 0; border-top: 1px solid #E5E7EB; margin: 16px 0; }}
        </style>
    </head>
    <body>
        <div class="btn-container">
            <button class="pdf-btn" id="dl-btn" onclick="downloadDirectPDF()">📥 진단 리포트 PDF 바로 다운로드</button>
            <button class="pdf-btn print-mode" onclick="openPrintPDF()">🖨️ 고화질 PDF 저장 / 인쇄 (표·수식 완벽 권장)</button>
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
                    margin:       [12, 12, 12, 12],
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
                const printWin = window.open('', '_blank', 'width=950,height=850');
                printWin.document.write(`
                    <html>
                    <head>
                        <title>${{fileName}}</title>
                        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
                        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
                        <style>
                            body {{ font-family: 'Noto Sans KR', sans-serif; padding: 20px; line-height: 1.68; font-size: 10.5pt; color: #111827; }}
                            .doc-header {{ text-align: center; font-size: 17pt; font-weight: 700; color: #1E3A8A; border-bottom: 2px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px; }}
                            h2, h3 {{ color: #1E3A8A; margin-top: 18px; margin-bottom: 8px; page-break-after: avoid; }}
                            table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px 0; font-size: 9.5pt; page-break-inside: avoid; }}
                            th, td {{ border: 1px solid #CBD5E1; padding: 7px 10px; text-align: left; }}
                            th {{ background-color: #EFF6FF !important; color: #1E3A8A !important; font-weight: 700; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                            blockquote {{ background-color: #F8FAFC !important; border-left: 4px solid #3B82F6; margin: 0 0 14px 0; padding: 10px 14px; border-radius: 4px; page-break-inside: avoid; break-inside: avoid; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
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
# 화면 UI 구성
# ==========================================
st.title("📊 생기부 역량 평가 및 보완점 진단기")
st.markdown("학생의 **학교생활기록부 PDF**를 업로드하면 **① 현재 학기까지의 시수 반영 내신(5등급제) 산출**, **② 본교 3개년 학생부종합 입결 환산 비교**, **③ 대학별 맞춤 비율(기본 4:4:2) 기반 9대 세부 역량 심층 채점**, **④ 다음 학기 보완 처방전 및 맞춤 추천 대학(종합·교과 통합)**을 제시합니다.")
st.caption("하다가 어려운점이 있다면 제작자에게 문의주세요(지족고 송정훈)")

api_key = st.text_input("Gemini API 키를 입력하세요 (면접 생성기와 동일한 무료 키 사용 가능)", type="password")

st.markdown("#### 1️⃣ 희망 대학·학과 및 학생 기본 정보")
col1, col2, col3 = st.columns(3)
with col1:
    university = st.text_input("희망 대학", placeholder="예: 충남대, 동국대, 고려대")
with col2:
    major = st.text_input("모집 단위(학과)", placeholder="예: 수학교육과, 컴퓨터공학과")
with col3:
    student_name = st.text_input("학생 이름 (개인정보 가림 처리용)", placeholder="예: 홍길동")

st.markdown("#### 2️⃣ 희망 대학 학생부종합전형 반영 비율 설정")
weight_mode = st.radio(
    "반영 비율 적용 방식 선택",
    options=[
        "🔍 입력한 대학 기준 자동 매칭 (미확인 대학은 표준 4 : 4 : 2 자동 적용)",
        "🎛️ 슬라이더로 수동 비율 직접 입력"
    ],
    index=0,
    horizontal=True
)

if weight_mode.startswith("🔍"):
    w_acad, w_career, w_comm, weight_desc = get_auto_univ_weights(university, major)
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
    uploaded_file = st.file_uploader("나이스 학생부 PDF 파일 업로드 (1학년 또는 2학년 생기부 자동 감지)", type="pdf")
with u_col2:
    pdf_password = st.text_input("PDF 비밀번호 (생년월일 6자리 등, 없으면 비워둠)", type="password")

# ==========================================
# 6. 진단 시작 버튼 클릭 시 동작
# ==========================================
if st.button("🚀 생기부 심층 진단 및 추천 대학 산출하기", type="primary"):
    if not api_key:
        st.warning("API 키를 입력해주세요.")
    elif not uploaded_file:
        st.warning("생기부 PDF 파일을 업로드해주세요.")
    elif not university or not major:
        st.warning("희망 대학과 학과를 입력해주세요.")
    elif w_comm < 5:
        st.warning("공동체역량 비율이 최소 5% 이상이 되도록 학업·진로 비율을 조절해주세요.")
    else:
        with st.spinner("생기부 비식별화, 학종 기준 성적 비교, 9대 세부 역량 심층 분석 및 맞춤 추천 대학(종합·교과)을 도출 중입니다..."):
            try:
                # 1) PDF 텍스트 추출
                raw_student_text = ""
                with pdfplumber.open(uploaded_file, password=pdf_password if pdf_password else None) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_student_text += page_text + "\n"

                safe_student_text = mask_personal_info(raw_student_text, student_name)

                # 2) 본교 3개년 입시결과(희망대학=학종 전용 / 추천대학=종합+교과) 추출
                bench_info = analyze_school_admission_benchmark(adm_df, university, major)

                # 3) Gemini 3.8 Flash 호출
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name='gemini-3.8-flash',
                    generation_config={"max_output_tokens": 16384}
                )

                prompt = f"""
                당신은 {university} {major}의 학생부종합전형 수석 입학사정관이자 대입 진학 지도 전문가입니다.
                아래 제공된 [본교 최근 3개년 희망 대학 학생부종합전형 합격자 데이터(5등급제 환산 포함)], [본교 내 동일/유사 계열 타 대학 합격자 참고 데이터(종합+교과 포함)], 그리고 비식별화된 [학생 생활기록부 추출 내용]을 정밀 분석하여 **매우 상세하고 깊이 있는 종합 진단 리포트**를 작성하세요.

                [핵심 지시사항 1: 현재 기록된 학기 자동 감지 및 '학생부종합전형' 기준 성적 분석]
                - 업로드된 생기부 교과학습발달상황 표를 확인하여, 성적(석차등급 1~5등급)이 실제로 산출되어 있는 학기(예: 1학년 1학기까지, 1학년 전체, 2학년 1학기까지, 2학년 전체 등)까지만 자동으로 감지하세요.
                - 학년별 비율 없이, 현재까지 석차등급 숫자가 존재하는 모든 과목의 **이수단위(시수)를 가중치로 반영한 누적 평균 등급**($\\frac{{\\sum(\\text{{단위수}} \\times \\text{{석차등급}})}}{{\\sum \\text{{단위수}}}}$)을 전체 교과, 주요 교과(국·영·수·사·과), 희망 전공 관련 핵심 교과로 나누어 정밀하게 제시하세요.
                - **중요:** 사용자가 입력한 희망 대학([{university} {major}])에 대한 성적 및 유불리 분석은 **반드시 '학생부종합전형(학종)'을 기준으로만 평가**하세요. 단순 정량 내신 수치뿐만 아니라 전공 관련 과목 선택 이수 여부, 원점수/평균 대비 학업 역량, 학기별 성적 추이, 그리고 세특 역량으로 내신을 얼마나 만회(또는 강화)할 수 있는지를 종합전형 관점에서 판정하세요.

                [핵심 지시사항 2: 9개 세부 항목 각 10점 만점 채점 및 대학 반영 비율({w_acad}% : {w_career}% : {w_comm}%) 환산]
                - 적용된 반영 기준: {weight_desc} (학업역량 {w_acad}% : 진로역량 {w_career}% : 공동체역량 {w_comm}%)
                - 아래 9개 세부 항목을 각각 **10점 만점(소수점 첫째 자리까지, 예: 8.5/10)**으로 채점한 뒤, 반영 비율에 맞춰 100점 만점으로 환산하세요.
                  * 학업역량 환산 점수 = (항목 1+2+3 합계 / 30) × {w_acad}
                  * 진로역량 환산 점수 = (항목 4+5+6 합계 / 30) × {w_career}
                  * 공동체역량 환산 점수 = (항목 7+8+9 합계 / 30) × {w_comm}
                  * **최종 종합 총점(100점 만점)** = 위 3개 환산 점수의 합계

                [핵심 지시사항 3: 9개 항목별 분석을 '절대 간략히 요약하지 말고' 매우 상세하게 작성할 것 (가장 중요!)]
                - 1~2문장짜리 짧은 요약은 절대 금지합니다! 실제 담임교사와 학생이 상담 자료로 바로 쓸 수 있도록, 각 항목마다 다음 3가지를 **최소 8~10문장 이상**의 풍부한 분량으로 정밀 분석하세요:
                  1) **✅ 강점 및 긍정적 기록 심층 분석 (최소 4~5문장 이상)**: 학년·학기별 구체적인 과목명(원점수/평균/등급 포함)이나 창체 영역을 명시하고, 세특에 등장한 탐구 주제·수식·실험·이론이 왜 {major} 전공 관점에서 학술적 가치가 있는지 구체적인 문맥을 짚어 서술하세요.
                  2) **⚠️ 아쉬운 점·감점 요인 및 입학사정관 의구심 포인트 (최소 3~4문장 이상)**: 학기별 성적 하락 구간, 미이수 권장 과목, 지필 성적 대비 세특 용어의 과장(괴리감), 혹은 단순 조사·발표에 그친 얕은 탐구 기록이 어디인지 정확히 짚어 감점 사유를 분석하세요.
                  3) **💡 이 항목 점수를 10점 만점으로 끌어올리기 위한 핀셋 보완책 (2~3문장)**: 다음 학기(또는 면접 대비)에 구체적으로 어떤 과목/활동에서 무엇을 보완해야 하는지 항목별 맞춤 처방을 제시하세요.

                [핵심 지시사항 4: 추천 대학 및 유사 학과 산출 시에는 '학생부종합'과 '학생부교과'를 모두 고려할 것]
                - 5번 섹션에서 추천 대학 및 유사 학과를 제안할 때는 **학생부종합전형(학종)과 학생부교과전형(교과)을 모두 고려**하여 학생에게 가장 유리한 지원 전략을 짜주세요.
                - 예를 들어 내신 성적이 우수한 편이라면 교과전형과 학종 병행을 추천하고, 내신 대비 세특 탐구력이 뛰어나다면 학종 위주로 추천하는 등 **[🚀 상향·소신 추천 (2곳)]**, **[🎯 적정·맞춤 추천 (2곳)]**, **[🛡️ 안정·실속 추천 (2곳)]** 총 6곳을 구체적인 전형유형(학종/교과)과 함께 제시하세요.

                - 수학·과학 수식이나 기호가 등장할 때는 깨지지 않도록 표준 LaTeX 문법인 $수식$ 형태로 작성하세요. 인사말 없이 아래 [출력 양식]을 정확히 지켜 출력하세요.

                ---
                [출력 양식]
                ## 📈 1. 생기부 자동 감지 내신 산출 및 [{university} {major}] 학생부종합전형 기준 성적 분석
                > - **감지된 생기부 반영 범위**: (예: 1학년 1학기 ~ 2학년 1학기까지 총 OO단위 이수 기준)
                > - **학생 시수 반영 누적 내신 (5등급제)**:
                >   * **전체 전교과 평균**: **O.OO 등급** (1학년 O.OO등급 ➔ 2학년 O.OO등급 추이 설명)
                >   * **국·영·수·사·과 주요 교과 평균**: **O.OO 등급**
                >   * **희망 전공({major}) 관련 핵심 교과 평균**: **O.OO 등급** (주요 과목 원점수·성취도 및 선택과목 이수 특징 요약)
                > - **[{university} {major}] 학생부종합전형(학종) 기준 종합 판정**: **[🟢적정·안정권 / 🟡소신 경합권 / 🔴상향 도전권 / 🔵초과안정(상향권장) 중 택1]**
                >   * (본교 3개년 학종 5등급제 환산 합격선 및 전국 학종 기준 대비 현재 학생의 정량 내신 위치와, 학생의 세특·창체 정성 역량을 결합했을 때 학종에서 어떤 경쟁력을 갖는지 3~4문장으로 상세히 서술)

                ## 🏆 2. 대학 반영비율({w_acad}:{w_career}:{w_comm}) 적용 종합 평가 점수표
                > 📌 **적용 기준**: {weight_desc} (학업 {w_acad}% : 진로 {w_career}% : 공동체 {w_comm}%)

                | 평가 영역 | 세부 평가 항목 (9대 지표) | 원점수 (10점 만점) | 영역 소계 (30점 만점) | 대학 반영비율 환산 점수 |
                | :--- | :--- | :---: | :---: | :---: |
                | **📘 학업역량 ({w_acad}%)** | 1. 학업성취도 및 이수 성실성 | O.O / 10 | **OO.O / 30** | **OO.O / {w_acad}점** |
                | **📘 학업역량 ({w_acad}%)** | 2. 학업태도 및 자기주도성 | O.O / 10 | ↑ | ↑ |
                | **📘 학업역량 ({w_acad}%)** | 3. 교과 심화 탐구력 | O.O / 10 | ↑ | ↑ |
                | **📗 진로역량 ({w_career}%)** | 4. 전공 관련 교과 이수 노력 | O.O / 10 | **OO.O / 30** | **OO.O / {w_career}점** |
                | **📗 진로역량 ({w_career}%)** | 5. 전공 관련 교과 성취도 | O.O / 10 | ↑ | ↑ |
                | **📗 진로역량 ({w_career}%)** | 6. 진로 탐색 및 창체·세특 연계 | O.O / 10 | ↑ | ↑ |
                | **📙 공동체역량 ({w_comm}%)** | 7. 협업과 소통 능력 | O.O / 10 | **OO.O / 30** | **OO.O / {w_comm}점** |
                | **📙 공동체역량 ({w_comm}%)** | 8. 나눔과 배려 실천 | O.O / 10 | ↑ | ↑ |
                | **📙 공동체역량 ({w_comm}%)** | 9. 성실성·출결 및 규칙 준수 | O.O / 10 | ↑ | ↑ |
                | **🌟 최종 종합 총점** | **9개 항목 종합 ({university} {major} 맞춤 환산)** | - | **OO.O / 90** | **🎯 OO.O / 100점** |

                ## 🔍 3. 9개 세부 항목별 점수 판단 근거 및 심층 정밀 분석
                *(주의: 절대 1~2문장으로 짧게 요약하지 마세요. 1번부터 9번 항목까지 각각 아래 양식에 맞춰 과목명·원점수·세특 문맥을 풍부하게 인용하여 상세히 작성하세요.)*
                ### 📌 [항목 번호 및 이름] — **O.O / 10점**
                > - **✅ 강점 및 긍정적 기록 심층 분석**: (최소 4~5문장 이상. 학기별 과목 성취도와 세특에 기록된 구체적 탐구 주제·원리·수식·활동 과정을 상세히 인용하고 입학사정관이 높이 평가할 포인트를 구체적으로 서술)
                > - **⚠️ 아쉬운 점·감점 요인 및 보완 필요성**: (최소 3~4문장 이상. 성적 추이상 약점, 지필 성적과 세특 난이도의 괴리 여부, 누락된 탐구 단계나 아쉬운 기록 방식을 구체적으로 지적)
                > - **💡 항목별 핀셋 업그레이드 전략**: (2~3문장. 남은 학기 세특이나 창체에서 이 항목의 감점을 만회하기 위해 수행해야 할 구체적 활동 및 기록 방향 제시)

                ## 💡 4. 다음 학기 생기부 최종 보완 처방전 (Action Plan)
                > 🎯 **최우선 보완 타깃 항목**: (9개 항목 중 가장 시급히 보완해야 할 2~3개 항목과 핵심 이유)
                >
                > **1) [전공 관련 선택과목 및 교과 세특 심화 로드맵]**
                > - (남아있는 학기에 어떤 과목 세특에서 1~2학년의 탐구 키워드를 어떻게 확장할지 구체적인 심화 탐구 주제 3가지를 [과목명 - 추천 탐구 주제 - 핵심 원리 및 세특 작성 포인트] 형태로 상세히 제시)
                >
                > **2) [창의적 체험활동(자율·동아리·진로) 나선형 연결 전략]**
                > - (기존 창체 활동의 한계를 극복하고 전공 역량을 입증할 수 있는 구체적인 프로젝트·실험·독서·산출물 아이디어 제시)
                >
                > **3) [지필 성적 관리 및 세특 진정성(괴리 해소) 확보 전략]**
                > - (성적 추이 보완 목표치 및 지필 성적 수준에 부합하는 설득력 있는 세특 구성 전략)

                ## 🧭 5. 현재 학생 상황 맞춤 추천 대학 및 유사 학과 포트폴리오 (학종·교과 통합 고려)
                > - **희망 대학([{university} {major}]) 적합도 및 전형 전략 총평**: (현재 학생의 5등급제 내신과 생기부 역량을 종합할 때 희망 대학 지원이 적절한지, 그리고 학생에게 학생부종합전형과 학생부교과전형 중 어떤 비중으로 수시 6장 포트폴리오를 짜는 것이 유리한지 2~3문장으로 진단)

                | 지원 전략 구분 | 추천 대학명 | 추천 모집단위 (동일·유사 학과) | 추천 전형유형 (종합 / 교과) | 예상 합격권 (5등급제 환산 기준) | 학생 생기부·내신 기준 전형별 추천 사유 및 유불리 분석 |
                | :---: | :--- | :--- | :---: | :---: | :--- |
                | **🚀 상향·소신** | (대학 1) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (학생의 어떤 세특 강점 또는 내신 구조로 도전 가능한지 구체적 이유) |
                | **🚀 상향·소신** | (대학 2) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (구체적 추천 이유 및 보완 조건) |
                | **🎯 적정·맞춤** | (대학 3) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (현재 성적·세특 수준에서 가장 합격 확률이 높은 이유) |
                | **🎯 적정·맞춤** | (대학 4) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (본교 입결 데이터 및 학생 역량 매칭 분석) |
                | **🛡️ 안정·실속** | (대학 5) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (내신·최저 변수 대비 안전하게 확보 가능한 근거) |
                | **🛡️ 안정·실속** | (대학 6) | (유사 학과명) | 학생부종합 / 교과 중 유리한 전형 명시 | 약 O.OO ~ O.OO 등급 | (본교 다수 합격 사례 기반 추천 근거) |

                > - **📌 유사 학과 확장 가이드**: (학생이 희망하는 [{major}] 외에, 현재 생기부에 적힌 탐구 주제들을 그대로 활용하여 지원할 수 있는 유사 학과·융합 학과 3~4개를 추천하고 그 이유를 상세히 설명)

                ---
                [본교 최근 3개년(2024~2026 대입) 희망 대학·학과 '학생부종합전형' 합격자 데이터]
                {bench_info['prompt_context']}

                [본교 최근 3개년 동일/유사 계열 타 대학 합격자 참고 데이터 (학생부종합 + 학생부교과 통합)]
                {bench_info['similar_majors_context']}

                [학생 생활기록부 추출 내용]
                {safe_student_text}
                """

                # 429 한도 초과 방지를 위한 자동 재시도 로직
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

                # 토큰 사용량 추출
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
                st.session_state.target_title = f"{university}_{major}"
                st.session_state.quant_summary_md = bench_info["summary_md"]
                st.session_state.applied_weights_info = f"학업 {w_acad}% : 진로 {w_career}% : 공동체 {w_comm}% ({weight_desc})"

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# ==========================================
# 7. 결과 출력 및 PDF 다운로드 구역
# ==========================================
if st.session_state.eval_report:
    st.success(f"개인정보 보호 처리 및 심층 분석이 완료되었습니다! [{st.session_state.target_title.replace('_', ' ')}] 진단 리포트를 출력합니다.")

    if st.session_state.token_info:
        t_info = st.session_state.token_info
        tpm_percent = round((t_info["total"] / 250000) * 100, 1)
        st.info(
            f"📊 **이번 작업 토큰 소모량:** 총 **{t_info['total']:,} 토큰** "
            f"(입력: {t_info['prompt']:,} / 출력: {t_info['output']:,}) "
            f"— 무료 분당 토큰 한도(25만)의 약 **{tpm_percent}%** 사용 | "
            f"**적용 비율:** {st.session_state.applied_weights_info}"
        )

    # PDF 저장용 통합 문서
    full_combined_report = (
        "## 🏫 [참고] 본교 최근 3개년(2024~2026 대입) 학생부종합전형 입시결과 및 5등급제 환산 기준\n"
        + st.session_state.quant_summary_md
        + "\n\n---\n\n"
        + st.session_state.eval_report
    )

    st.markdown("#### 📄 진단 리포트 저장하기 (표·수식·서식 유지)")
    pdf_download_component(
        result_text=full_combined_report,
        title_prefix=st.session_state.target_title
    )

    st.divider()

    # 본교 3개년 입결 데이터 박스 출력 (학종 전용)
    with st.expander("🏫 본교 최근 3개년(2024~2026 대입) 해당 대학·학과 [학생부종합전형] 입시결과 및 5등급제 환산표 보기", expanded=True):
        st.markdown(st.session_state.quant_summary_md)

    st.divider()

    # AI 종합 진단 리포트 본문 출력
    st.markdown(st.session_state.eval_report)

    st.download_button(
        label="📄 일반 텍스트로 백업 다운로드 (.txt)",
        data=full_combined_report,
        file_name=f"{st.session_state.target_title}_생기부진단리포트.txt",
        mime="text/plain"
    )

    with st.expander("보안 처리된 생기부 텍스트 확인하기 (클릭)"):
        st.text(st.session_state.safe_student_text[:1000] + "\n\n... (중략) ...")
