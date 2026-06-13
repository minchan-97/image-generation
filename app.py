"""
당신의 업무만을 위한 AI — 이미지 일관성 생성
================================================
참조 이미지/설정 → CoreAI 학습 → 가드레일 → DALL-E 생성
"""
import streamlit as st
import os, io, base64, pickle, time
from openai import OpenAI

st.set_page_config(
    page_title="이미지 일관성 AI",
    page_icon="🎨",
    layout="wide",
)

st.markdown("""
<style>
body { background: #0f0f0f; color: #f0f0f0; }
.stApp { background: #0f0f0f; }
.title { font-size: 1.6rem; font-weight: 800;
         background: linear-gradient(90deg,#a78bfa,#60a5fa);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #6b7280; font-size: 0.85rem; margin-bottom: 1rem; }
.card { background: #1a1a2e; border-radius: 12px; padding: 1rem;
        border: 1px solid #2d2d44; margin-bottom: 0.8rem; }
.tag-pass  { background:#064e3b; color:#6ee7b7;
             padding:2px 8px; border-radius:999px; font-size:0.75rem; }
.tag-warn  { background:#78350f; color:#fcd34d;
             padding:2px 8px; border-radius:999px; font-size:0.75rem; }
.tag-fatal { background:#7f1d1d; color:#fca5a5;
             padding:2px 8px; border-radius:999px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── CoreAI v2 로드 ─────────────────────────────────────────────
try:
    from core_ai_v2_engine import CoreAIv2Engine
    ENGINE_OK = True
except Exception:
    ENGINE_OK = False

# ── 세션 ───────────────────────────────────────────────────────
for k,v in {
    "engine": None, "trained": False,
    "corpus": "", "history": [],
    "train_stats": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 사이드바 ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎨 이미지 일관성 AI")
    st.caption("당신의 업무만을 위한 AI")
    st.markdown("---")

    try:
        _key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY",""))
    except Exception:
        _key = os.getenv("OPENAI_API_KEY","")

    api_key = st.text_input("OpenAI API Key", value=_key,
                             type="password", placeholder="sk-...")
    model_img = st.selectbox("이미지 모델", ["dall-e-3","dall-e-2"])
    img_size  = st.selectbox("크기", ["1024x1024","1792x1024","1024x1792"])
    logp_thr  = st.slider("가드레일 민감도", -16.0, -8.0, -11.5, 0.5)

    st.markdown("---")
    st.markdown("### 📚 설정 학습")

    # 텍스트 설정 직접 입력
    setting_text = st.text_area(
        "캐릭터/세계관 설정 (텍스트)",
        placeholder="예: 파란 눈, 흰 머리, 중세 기사 갑옷\n사막 배경, 황금빛 조명\n사실적인 스타일, 영화적 구도",
        height=120,
    )

    # 참조 이미지 업로드 (선택)
    ref_images = st.file_uploader(
        "참조 이미지 (선택, 여러 장 가능)",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
    )

    n_clusters = st.slider("클러스터 수", 2, 8, 3)

    if st.button("🚀 학습 시작", use_container_width=True,
                 disabled=not (api_key and (setting_text or ref_images))):
        with st.spinner("설정 분석 중..."):
            try:
                client = OpenAI(api_key=api_key)
                corpus_lines = []

                # 텍스트 설정 추가
                if setting_text.strip():
                    corpus_lines.append(setting_text.strip())

                # 참조 이미지 → 텍스트 변환 (GPT-4V)
                if ref_images:
                    for img_file in ref_images:
                        img_b64 = base64.b64encode(img_file.read()).decode()
                        ext = img_file.name.split('.')[-1].lower()
                        mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"

                        resp = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type":"image_url","image_url":{
                                        "url": f"data:{mime};base64,{img_b64}"
                                    }},
                                    {"type":"text","text":
                                        "이 이미지의 시각적 특성을 상세히 설명해줘. "
                                        "색상, 스타일, 조명, 구도, 캐릭터 특징, 분위기를 "
                                        "각각 한 줄씩 명확하게. 이미지 생성 프롬프트에 "
                                        "재사용할 수 있게 구체적으로."}
                                ]
                            }],
                            max_tokens=500,
                        )
                        desc = resp.choices[0].message.content.strip()
                        corpus_lines.append(desc)
                        st.caption(f"✅ {img_file.name} 분석 완료")

                corpus = "\n".join(corpus_lines)
                st.session_state.corpus = corpus

                # CoreAI 학습
                engine = CoreAIv2Engine(n_clusters=n_clusters)
                t0 = time.perf_counter()
                stats = engine.train(corpus)
                elapsed = (time.perf_counter()-t0)*1000

                st.session_state.engine = engine
                st.session_state.trained = True
                st.session_state.train_stats = stats or {}
                st.success(f"✅ 학습 완료 ({elapsed:.0f}ms)")

            except Exception as e:
                st.error(f"학습 실패: {e}")

    st.markdown("---")
    # pkl 저장/로드
    if st.session_state.trained and st.session_state.engine:
        nm = st.session_state.engine.nm_engine
        save_data = {
            "n_clusters": st.session_state.engine.n_clusters,
            "global_vocab": st.session_state.engine.global_vocab,
            "corpus": st.session_state.corpus,
            "train_stats": st.session_state.train_stats,
            "emb_emb": st.session_state.engine.embedder.emb,
            "emb_vocab": st.session_state.engine.embedder.vocab,
            "emb_dim": st.session_state.engine.embedder.dim,
            "cluster_sentences": dict(st.session_state.engine.decomposer.cluster_sentences),
            "cluster_tokens": dict(st.session_state.engine.decomposer.cluster_tokens),
            "cluster_keywords": st.session_state.engine.decomposer.cluster_keywords,
            "decomp_vocab": st.session_state.engine.decomposer.vocab,
            "decomp_W": st.session_state.engine.decomposer.W,
            "markovs": {
                k: {"uni":dict(m.uni),
                    "bi": {k2:dict(v) for k2,v in m.bi.items()},
                    "tri":{k2:dict(v) for k2,v in m.tri.items()},
                    "total":m.total}
                for k,m in st.session_state.engine.markovs.items()
            },
            "nm_engine": {
                "uni":   dict(nm.uni),
                "bi":    {k2:dict(v) for k2,v in nm.bi.items()},
                "tri":   {k2:dict(v) for k2,v in nm.tri.items()},
                "total": nm.total,
            } if nm.is_trained else None,
        }
        st.download_button(
            "💾 설정 저장 (.pkl)",
            data=pickle.dumps(save_data),
            file_name="image_style.pkl",
            mime="application/octet-stream",
        )

    st.markdown("### 💾 설정 불러오기")
    pkl_file = st.file_uploader("설정 파일 (.pkl)", type=None, key="pkl_up")
    if pkl_file and pkl_file.name.endswith('.pkl'):
        try:
            data = pickle.loads(pkl_file.read())
            engine = CoreAIv2Engine.load_from_dict(data)
            st.session_state.engine = engine
            st.session_state.trained = True
            st.session_state.corpus = data.get("corpus","")
            st.session_state.train_stats = data.get("train_stats",{})
            st.success("✅ 설정 로드 완료")
            st.rerun()
        except Exception as e:
            st.error(f"로드 실패: {e}")

    if st.session_state.trained:
        if st.button("🔄 초기화", use_container_width=True):
            for k in ["engine","trained","corpus","history","train_stats"]:
                st.session_state[k] = None if k=="engine" else ([] if k=="history" else ("" if k=="corpus" else {}))
            st.session_state.trained = False
            st.rerun()

# ── 메인 ───────────────────────────────────────────────────────
st.markdown('<div class="title">🎨 이미지 일관성 AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">당신의 설정으로 학습 → 가드레일 → 일관성 있는 이미지 생성</div>',
            unsafe_allow_html=True)

if not st.session_state.trained:
    st.info("← 사이드바에서 설정을 입력하고 학습해주세요.")
    if st.session_state.corpus == "":
        with st.expander("💡 사용 방법", expanded=True):
            st.markdown("""
**1. 설정 입력**
- 텍스트로 캐릭터/세계관 설명
- 참조 이미지 업로드 (GPT-4V가 자동 분석)

**2. 학습**
- CoreAI가 설정을 학습
- 가드레일이 일관성 기준 설정

**3. 이미지 생성**
- 요청 → 가드레일 체크 → PASS면 DALL-E 호출
- 설정에서 벗어난 요청은 재시도

**활용 예시**
- 웹툰 캐릭터 일관성 유지
- 게임 세계관 아트 생성
- 브랜드 이미지 일관화
""")
    st.stop()

# ── 학습 완료 상태 ─────────────────────────────────────────────
col_info, col_main = st.columns([1, 3])

with col_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**학습된 설정**")
    stats = st.session_state.train_stats or {}
    st.metric("클러스터", stats.get("n_clusters", "?"))
    st.metric("어휘", stats.get("vocab_size", len(st.session_state.engine.global_vocab)))
    if st.session_state.engine.decomposer.cluster_keywords:
        st.markdown("**키워드**")
        for k, kws in st.session_state.engine.decomposer.cluster_keywords.items():
            if kws:
                st.caption(f"C{k}: {' · '.join(kws[:4])}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 설정 원문
    with st.expander("📄 학습 설정 원문"):
        st.text(st.session_state.corpus[:600])

with col_main:
    st.markdown("### ✏️ 이미지 요청")
    prompt = st.text_area(
        "어떤 이미지를 원하나요?",
        placeholder="예: 같은 캐릭터가 말을 타고 사막을 달리는 장면",
        height=80,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([2,1])
    with col_btn1:
        generate = st.button("🎨 생성하기", use_container_width=True,
                             disabled=not (api_key and prompt.strip()))
    with col_btn2:
        max_retry = st.selectbox("재시도", [1,2,3], index=1, label_visibility="collapsed")

    if generate and prompt.strip():
        if not api_key:
            st.error("API Key를 입력하세요")
            st.stop()

        client = OpenAI(api_key=api_key)

        with st.spinner("가드레일 체크 중..."):
            # 1. 가드레일 체크
            result = st.session_state.engine.evaluate(prompt, logp_thr=logp_thr)
            verdict = result.get("verdict","SKIP")
            logp    = result.get("logp", 0.0)

            # 2. 일관성 프롬프트 강화
            corpus_hint = st.session_state.corpus[:400]
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Style consistency rules:\n{corpus_hint}\n\n"
                f"Maintain exact visual consistency with the above settings."
            ) if verdict in ("PASS","WARNING") else None

            # 3. FATAL이면 재시도
            attempts = 1
            final_verdict = verdict
            final_prompt  = enhanced_prompt or prompt

            if verdict == "FATAL" and max_retry > 1:
                for i in range(max_retry - 1):
                    # LLM으로 프롬프트 수정
                    fix_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":
                                f"다음 설정을 반드시 지켜서 이미지 프롬프트를 수정해줘:\n{corpus_hint}"},
                            {"role":"user","content":
                                f"원본 요청: {prompt}\n\n"
                                f"설정에 맞게 수정된 영어 이미지 프롬프트만 반환해줘."}
                        ],
                        max_tokens=200,
                    )
                    new_prompt = fix_resp.choices[0].message.content.strip()
                    new_result = st.session_state.engine.evaluate(new_prompt, logp_thr=logp_thr)
                    attempts += 1
                    final_verdict = new_result.get("verdict","SKIP")
                    final_prompt  = new_prompt
                    if final_verdict in ("PASS","WARNING"):
                        break

        # 4. 판정 표시
        tag_cls = {"PASS":"tag-pass","WARNING":"tag-warn","FATAL":"tag-fatal"}.get(final_verdict,"")
        st.markdown(f'<span class="{tag_cls}">{final_verdict}</span> '
                    f'logP: {logp:+.2f} | {attempts}회 시도',
                    unsafe_allow_html=True)

        if final_verdict == "FATAL":
            st.warning("⚠️ 설정과 맞지 않는 요청이에요. 설정에 맞게 요청을 수정해보세요.")
        else:
            # 5. DALL-E 호출
            with st.spinner("이미지 생성 중..."):
                try:
                    img_resp = client.images.generate(
                        model=model_img,
                        prompt=final_prompt,
                        size=img_size,
                        quality="standard",
                        n=1,
                    )
                    img_url = img_resp.data[0].url

                    st.image(img_url, use_container_width=True)
                    st.caption(f"프롬프트: {final_prompt[:120]}...")

                    # 기록에 추가
                    st.session_state.history.insert(0, {
                        "prompt":   prompt,
                        "enhanced": final_prompt,
                        "verdict":  final_verdict,
                        "url":      img_url,
                        "logp":     logp,
                    })

                except Exception as e:
                    st.error(f"이미지 생성 실패: {e}")

    # ── 히스토리 ───────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🖼️ 생성 기록")
        cols = st.columns(3)
        for i, h in enumerate(st.session_state.history[:6]):
            with cols[i % 3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                tag_cls = {"PASS":"tag-pass","WARNING":"tag-warn","FATAL":"tag-fatal"}.get(h["verdict"],"")
                st.markdown(f'<span class="{tag_cls}">{h["verdict"]}</span>', unsafe_allow_html=True)
                if h.get("url"):
                    st.image(h["url"], use_container_width=True)
                st.caption(h["prompt"][:50])
                st.markdown('</div>', unsafe_allow_html=True)
