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
# 3. 지족고 3개년 입결 검색 및 미지원 학과 3단계 자동 보정 함수
# ==========================================
def normalize_univ_keyword(univ_input):
    s = univ_input.strip()
    s = re.sub(r'\(.*?\)', '', s)
    s = s.replace("국립", "").replace("대학교", "").replace("대학", "").strip()
    # 흔히 쓰는 약칭 보정
    alias_map = {
        "충대": "충남", "한밭": "한밭", "공주": "공주", "충북": "충북",
        "외대": "한국외국어", "한국외대": "한국외국어", "한기대": "한국기술교육",
        "교원대": "한국교원", "교통대": "한국교통", "금오공대": "금오공과",
        "과기대": "서울과학기술", "시립대": "서울시립", "성대": "성균관",
        "숙대": "숙명여자", "이대": "이화여자", "고대": "고려", "연대": "연세"
    }
    return alias_map.get(s, s)

def analyze_school_admission_benchmark(df, univ_input, major_input):
    if df is None or df.empty:
        return {
            "match_level": "데이터 없음",
            "summary_md": "⚠️ `jijok_admission_data.csv` 파일을 찾을 수 없어 일반 전국 기준으로 진단합니다.",
            "prompt_context": "본교 3개년 입결 파일이 연결되지 않았습니다. 일반적인 해당 대학·학과 수준으로 평가하세요."
        }

    u_key = normalize_univ_keyword(univ_input)
    m_key = major_input.strip().replace("학과", "").replace("학부", "").replace("전공", "").strip()

    # 대학 매칭
    univ_mask = df['대학명'].str.contains(u_key, na=False)
    if '검색용_대학명' in df.columns:
        univ_mask = univ_mask | df['검색용_대학명'].str.contains(u_key, na=False)
    univ_df = df[univ_mask]

    # 학과 매칭
    if not univ_df.empty and m_key:
        exact_df = univ_df[univ_df['모집단위'].str.contains(m_key, na=False)]
    else:
        exact_df = pd.DataFrame()

    # 3단계 폴백(Fallback) 판정
    if not exact_df.empty:
        target_df = exact_df
        match_level = f"✅ 1단계 정확 일치: 본교 3개년 [{univ_input} - {major_input}] 합격 사례 ({len(target_df)}건)"
    elif not univ_df.empty:
        target_df = univ_df
        match_level = f"🔄 2단계 동일 대학 보정: 본교 내 [{major_input}] 직접 사례가 없어 [{univ_input} 전체 합격 사례 ({len(target_df)}건)] 기준으로 보정"
    else:
        target_df = pd.DataFrame()
        match_level = f"🌐 3단계 전국 기준 보정: 본교 3개년 데이터 내 [{univ_input}] 사례가 없어 AI 전국 대학 입결 기준 및 본교 전체 학종 분포(평균 3.85등급 / 5등급제 환산 약 2.33등급)를 참고하여 보정"

    if target_df.empty:
        return {
            "match_level": match_level,
            "summary_md": f"**{match_level}**\n\n* 본교 최근 3개년(2024~2026 대입) 수시 합격 데이터에 해당 대학 지원·합격 사례가 없어, 해당 대학·학과의 일반적인 학종 수준과 본교 학종 합격자 평균(9등급제 3.85등급 ➔ **5등급제 환산 2.33등급**)을 바탕으로 보정 평가합니다.",
            "prompt_context": f"본교 3개년 데이터에 {univ_input} {major_input} 사례가 없습니다. 본교 전체 학종 합격 평균은 9등급제 3.85등급(5등급제 환산 2.33등급)입니다. 해당 대학의 일반적인 위상과 비교하여 평가해 주세요."
        }

    # 전형유형 및 최종단계별 요약 표 생성
    rows_md = []
    prompt_lines = [f"[{match_level}]"]

    for t_type in ["학생부위주(종합)", "학생부위주(교과)"]:
        sub = target_df[target_df['전형유형'] == t_type]
        if sub.empty:
            continue
        for stage in ["합격", "충원합격"]:
            st_sub = sub[sub['최종단계'] == stage]
            if st_sub.empty:
                continue
            cnt = len(st_sub)
            mean_9 = st_sub['내신등급'].mean()
            min_9 = st_sub['내신등급'].min()
            max_9 = st_sub['내신등급'].max()
            mean_5 = convert_9_to_5(mean_9)
            min_5 = convert_9_to_5(min_9)
            max_5 = convert_9_to_5(max_9)
            stage_label = "최초합격" if stage == "합격" else "충원합격(추합)"
            rows_md.append(
                f"| **{t_type}** | {stage_label} | {cnt}건 | **{mean_5:.2f} 등급** *(범위: {min_5:.2f}~{max_5:.2f})* | {mean_9:.2f} 등급 *(범위: {min_9:.2f}~{max_9:.2f})* |"
            )
            prompt_lines.append(
                f"- {t_type} ({stage_label}, {cnt}건): 9등급제 평균 {mean_9:.2f}등급(최고 {min_9:.2f}~최저 {max_9:.2f}) -> 현행 5등급제 환산 평균 {mean_5:.2f}등급(범위 {min_5:.2f}~{max_5:.2f}등급)"
            )

    # 최근 합격 사례 상세(모집단위, 최저기준) 최대 5건
    sample_cases = target_df[['대입학년도', '대학명', '전형유형', '모집단위', '최종단계', '내신등급', '내신등급_5등급환산', '최저학력기준']].head(5)
    case_lines = []
    for _, r in sample_cases.iterrows():
        min_std = str(r['최저학력기준']) if pd.notna(r['최저학력기준']) and str(r['최저학력기준']).strip() else "최저 없음/미기재"
        case_lines.append(f"  * [{int(r['대입학년도'])}대입] {r['모집단위']} ({r['전형유형']} - {r['최종단계']}): **5등급환산 {r['내신등급_5등급환산']:.2f}등급** (9등급제 {r['내신등급']:.2f}) | {min_std}")

    table_header = (
        f"**{match_level}**\n\n"
        "| 전형유형 | 합격단계 | 본교 건수 | **현행 5등급제 환산 평균 (범위)** | 선배 9등급제 원본 평균 (범위) |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    summary_md = table_header + "\n".join(rows_md) + "\n\n**📌 본교 최근 주요 합격 사례 (참고용):**\n" + "\n".join(case_lines)
    prompt_context = "\n".join(prompt_lines) + "\n" + "\n".join(case_lines)

    return {
        "match_level": match_level,
        "summary_md": summary_md,
        "prompt_context": prompt_context
    }

# ==========================================
# 4. 표·수식·인용구 완벽 지원 PDF 다운로드 컴포넌트 (높이 100px 적용)
# ==========================================
def pdf_download_component(result_text: str, title_prefix: str):
    safe_text = json.dumps(result_text, ensure_ascii=False)
    safe_filename = json.dumps(f"{title_prefix}_생기부종합진단리포트.pdf", ensure_ascii=False)
    safe_title = json.dumps(f"{title_prefix} 생기부 역량 평가 및 보완점 진단 리포트", ensure_ascii=False)

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
                line-height: 1.65; font-size: 10.5pt; background: #ffffff;
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
                            body {{ font-family: 'Noto Sans KR', sans-serif; padding: 20px; line-height: 1.65; font-size: 10.5pt; color: #111827; }}
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
st.markdown("학생의 **학교생활기록부 PDF**를 업로드하면 **① 현재 기록된 학기까지의 시수 반영 내신(5등급제) 산출**, **② 본교 3개년(2024~2026 대입) 입결 환산 비교**, **③ 9대 세부 역량 점수(100점 만점) 및 판단 근거**, **④ 다음 학기 맞춤 보완 처방전**을 생성합니다.")
st.caption("하다가 어려운점이 있다면 제작자에게 문의주세요(지족고 송정훈)")

api_key = st.text_input("Gemini API 키를 입력하세요 (면접 생성기와 동일한 무료 키 사용 가능)", type="password")

st.markdown("#### 1️⃣ 희망 대학·학과 및 학생 기본 정보")
col1, col2, col3 = st.columns(3)
with col1:
    university = st.text_input("희망 대학", placeholder="예: 충남대, 공주대, 고려대")
with col2:
    major = st.text_input("모집 단위(학과)", placeholder="예: 수학교육과, 컴퓨터공학과")
with col3:
    student_name = st.text_input("학생 이름 (개인정보 가림 처리용)", placeholder="예: 홍길동")

st.markdown("#### 2️⃣ 희망 대학 평가 반영 비율 설정 (합계 100% 맞춤)")
st.caption("9개 세부 항목은 각각 10점 만점으로 채점된 뒤, 아래에서 설정한 대학별 역량 반영 비율(%)에 맞춰 100점 만점 총점으로 자동 환산됩니다.")
w_col1, w_col2, w_col3 = st.columns(3)
with w_col1:
    w_acad = st.slider("📘 학업역량 반영 비율 (%)", min_value=10, max_value=70, value=40, step=5)
with w_col2:
    w_career = st.slider("📗 진로역량 반영 비율 (%)", min_value=10, max_value=70, value=45, step=5)
with w_col3:
    w_comm = 100 - (w_acad + w_career)
    if w_comm >= 5:
        st.metric("📙 공동체역량 반영 비율 (자동 계산)", f"{w_comm}%")
    else:
        st.error(f"학업+진로 합계가 95%를 초과했습니다! (현재 공동체역량: {w_comm}%) 슬라이더를 조절해주세요.")

st.markdown("#### 3️⃣ 학생 생기부 PDF 업로드")
u_col1, u_col2 = st.columns([2, 1])
with u_col1:
    uploaded_file = st.file_uploader("나이스 학생부 PDF 파일 업로드 (1학년 또는 2학년 생기부 자동 감지)", type="pdf")
with u_col2:
    pdf_password = st.text_input("PDF 비밀번호 (생년월일 6자리 등, 없으면 비워둠)", type="password")

# ==========================================
# 5. 진단 시작 버튼 클릭 시 동작
# ==========================================
if st.button("🚀 생기부 정밀 진단 및 점수 산출하기", type="primary"):
    if not api_key:
        st.warning("API 키를 입력해주세요.")
    elif not uploaded_file:
        st.warning("생기부 PDF 파일을 업로드해주세요.")
    elif not university or not major:
        st.warning("희망 대학과 학과를 입력해주세요.")
    elif w_comm < 5:
        st.warning("공동체역량 비율이 최소 5% 이상이 되도록 학업·진로 비율을 조절해주세요.")
    else:
        with st.spinner("생기부 비식별화, 시수 반영 성적 산출, 본교 3개년 입결 비교 및 9대 역량 채점을 진행 중입니다..."):
            try:
                # 1) PDF 텍스트 및 성적표 추출
                raw_student_text = ""
                with pdfplumber.open(uploaded_file, password=pdf_password if pdf_password else None) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_student_text += page_text + "\n"

                safe_student_text = mask_personal_info(raw_student_text, student_name)

                # 2) 본교 3개년 입시결과 매칭 및 5등급제 환산 분석
                bench_info = analyze_school_admission_benchmark(adm_df, university, major)

                # 3) Gemini 3.8 Flash 호출
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name='gemini-3.8-flash',
                    generation_config={"max_output_tokens": 16384}
                )

                prompt = f"""
                당신은 {university} {major}의 학생부종합전형 수석 입학사정관이자 고등학교 진학 지도 전문가입니다.
                아래 제공된 [본교 최근 3개년(2024~2026 대입) 합격자 데이터(5등급제 환산 포함)]와 비식별화된 [학생 생활기록부 추출 내용]을 정밀 분석하여 종합 진단 리포트를 작성하세요.

                [핵심 지시사항 1: 현재 기록된 학기 자동 감지 및 시수(단위수) 반영 성적 산출]
                - 업로드된 생기부 교과학습발달상황 표를 확인하여, 성적(석차등급 1~5등급)이 실제로 산출되어 있는 학기(예: 1학년 1학기까지, 1학년 전체, 2학년 1학기까지, 2학년 전체 등)까지만 자동으로 감지하세요.
                - 아직 기록되지 않은 학년/학기는 절대 임의로 추정하지 마세요.
                - 학년별 비율 없이, 현재까지 석차등급 숫자가 존재하는 모든 과목의 **이수단위(시수)를 가중치로 반영한 누적 평균 등급**($\\frac{{\\sum(\\text{{단위수}} \\times \\text{{석차등급}})}}{{\\sum \\text{{단위수}}}}$)을 전체 교과와 주요 교과(국·영·수·사·과)로 나누어 계산해 제시하세요.
                - [본교 최근 3개년 합격자 데이터]의 **현행 5등급제 환산 합격선**과 학생의 현재 5등급제 내신을 비교하여 정량적 위치(🟢안정~적정권 / 🟡소신 경합권 / 🔴상향 역전필요권)를 명확히 판정하세요.

                [핵심 지시사항 2: 9개 세부 항목 각 10점 만점 채점 및 대학 반영 비율({w_acad}% : {w_career}% : {w_comm}%) 환산]
                - 아래 9개 세부 항목을 각각 **10점 만점(소수점 첫째 자리까지 허용, 예: 8.5/10)**으로 엄격하고 객관적으로 채점하세요.
                  * 📘 **학업역량 (3개 항목, 합계 30점 만점 -> {w_acad}점 만점으로 환산)**
                    1. 학업성취도 및 이수 성실성 (10점)
                    2. 학업태도 및 자기주도성 (10점)
                    3. 교과 심화 탐구력 (10점)
                  * 📗 **진로역량 (3개 항목, 합계 30점 만점 -> {w_career}점 만점으로 환산)**
                    4. 전공(계열) 관련 교과 이수 노력 (10점)
                    5. 전공(계열) 관련 교과 성취도 (10점)
                    6. 진로 탐색 활동 및 창체·세특 연계성 (10점)
                  * 📙 **공동체역량 (3개 항목, 합계 30점 만점 -> {w_comm}점 만점으로 환산)**
                    7. 협업과 소통 능력 (10점)
                    8. 나눔과 배려 실천 (10점)
                    9. 성실성·출결 및 규칙 준수 (10점)
                - **환산 공식:**
                  * 학업역량 환산 점수 = (항목 1+2+3 합계 / 30) × {w_acad}
                  * 진로역량 환산 점수 = (항목 4+5+6 합계 / 30) × {w_career}
                  * 공동체역량 환산 점수 = (항목 7+8+9 합계 / 30) × {w_comm}
                  * **최종 종합 총점(100점 만점)** = 위 3개 환산 점수의 합계 (소수점 첫째 자리까지 표기)

                [핵심 지시사항 3: 채점 판단 근거 및 다음 학기 맞춤 보완 처방전(Action Plan)]
                - 각 항목의 점수를 부여할 때는 반드시 **생기부에 실제로 적힌 구체적인 활동·과목 세특·성적 추이를 인용**하여 왜 그 점수를 주었는지(잘한 점과 감점 요인)를 명확히 서술하세요.
                - 단순 요약형 세특("~을 조사하여 발표함")은 감점 요인으로 지적하고, 호기심 -> 수학적·과학적·논리적 원리 탐구 -> 결론 도출로 이어진 기록은 가점 요인으로 평가하세요.
                - 감지된 학생의 현재 학년(1학년이면 2~3학년 대비, 2학년이면 3학년 1학기 마무리 대비)에 맞춰, **앞으로 남은 학기 동안 어느 과목 세특과 어느 창체 영역(자율·동아리·진로)에 어떤 구체적인 탐구 주제를 채워 넣어야 본교 합격 선배 수준을 넘어설 수 있는지** 실천적인 처방전을 작성하세요.
                - 수학·과학 수식이나 계산식이 등장할 때는 깨지지 않도록 표준 LaTeX 문법인 $수식$ 형태로 작성하세요. 인사말이나 불필요한 서론 없이 아래 [출력 양식]을 정확히 지켜 출력하세요.

                ---
                [출력 양식]
                ## 📈 1. 생기부 자동 감지 내신 산출 및 본교 3개년 입결(5등급제 환산) 비교
                > - **감지된 생기부 반영 범위**: (예: 1학년 1학기 ~ 2학년 1학기까지 총 OO단위 이수 기준)
                > - **학생 시수 반영 누적 내신 (5등급제)**: **전체 교과 O.OO 등급** / **주요 교과(국영수사과) O.OO 등급** (학년·학기별 추이 요약)
                > - **본교 3개년 입결 대비 판정**: **[🟢안정·적정권 / 🟡소신 경합권 / 🔴상향 도전권 중 택1]** - (본교 5등급제 환산 합격선 대비 유불리 및 최저학력기준 고려사항 2~3문장 요약)

                ## 🏆 2. 대학 반영비율({w_acad}:{w_career}:{w_comm}) 적용 종합 평가 점수표
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

                ## 🔍 3. 9개 세부 항목별 점수 판단 근거 및 정밀 분석
                *(1번부터 9번 항목까지 각각 아래 인용구 양식으로 빠짐없이 작성하세요)*
                ### 📌 [항목 번호 및 이름] — **O.O / 10점**
                > - **✅ 강점 및 긍정적 기록 근거**: (생기부의 실제 과목명·활동 내용을 인용하여 구체적으로 서술)
                > - **⚠️ 아쉬운 점 및 감점 요인**: (부족한 깊이, 누락된 권장 과목, 단순 나열식 기재 등 감점 사유를 명확히 지적)

                ## 💡 4. 다음 학기 생기부 최종 보완 처방전 (Action Plan)
                > 🎯 **최우선 보완 타깃 항목**: (9개 항목 중 가장 시급히 점수를 올려야 할 2~3개 항목 명시)
                >
                > **1) [전공 관련 선택과목 및 교과 세특 보완 전략]**
                > - (현재 학년 이후 남아있는 학기에 어떤 과목 세특에서 어떤 구체적 주제·수식·원리·실험으로 심화 탐구를 연결할지 구체적인 예시 2~3가지 제시)
                >
                > **2) [창의적 체험활동(자율·동아리·진로) 나선형 심화 전략]**
                > - (1~2학년 기존 활동의 아쉬운 점을 이어받아 다음 학기 동아리/진로활동에서 수행할 후속 심화 탐구 주제 및 산출물 제안)
                >
                > **3) [본교 3개년 입결 대비 맞춤형 지원·학습 전략]**
                > - (현재 산출된 5등급제 내신과 본교 5등급제 환산 합격선을 비교하여, 남은 학기 목표 내신 등급 및 수능 최저 대비 조언)

                ---
                [본교 최근 3개년(2024~2026 대입) 합격자 데이터]
                {bench_info['prompt_context']}

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

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# ==========================================
# 6. 결과 출력 및 PDF 다운로드 구역
# ==========================================
if st.session_state.eval_report:
    st.success(f"개인정보 보호 처리 및 분석이 완료되었습니다. [{st.session_state.target_title.replace('_', ' ')}] 맞춤형 생기부 진단 리포트를 출력합니다!")

    if st.session_state.token_info:
        t_info = st.session_state.token_info
        tpm_percent = round((t_info["total"] / 250000) * 100, 1)
        st.info(
            f"📊 **이번 작업 토큰 소모량:** 총 **{t_info['total']:,} 토큰** "
            f"(입력: {t_info['prompt']:,} / 출력: {t_info['output']:,}) "
            f"— 무료 분당 토큰 한도(25만)의 약 **{tpm_percent}%** 사용"
        )

    # PDF 저장할 때는 본교 3개년 입결 요약표와 AI 진단 리포트를 합쳐서 한 번에 출력
    full_combined_report = (
        "## 🏫 [참고] 본교 최근 3개년(2024~2026 대입) 입시결과 및 5등급제 환산 기준\n"
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

    # 본교 3개년 입결 데이터 박스 출력
    with st.expander("🏫 본교 최근 3개년(2024~2026 대입) 해당 대학·학과 입시결과 및 5등급제 환산표 보기", expanded=True):
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
