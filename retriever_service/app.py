from typing import Literal

import streamlit as st
from retriever.retriever import QdrantRetriever, RetrievalResult

SearchMode = Literal["hybrid", "dense", "sparse"]


# ── Конфиг страницы ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Поиск по нормативной базе",
    page_icon="🏗️",
    layout="wide",
)


@st.cache_resource(show_spinner="Подключение к Qdrant и загрузка моделей…")
def get_retriever():
    return QdrantRetriever()


retriever = get_retriever()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Параметры поиска")

    mode = st.selectbox(
        "Режим поиска",
        options=["hybrid", "dense", "sparse"],
        index=0,
        help=(
            "**hybrid** — dense + sparse Prefetch → ColBERT rerank (рекомендуется)\n\n"
            "**dense** — только multilingual-mpnet-base-v2 (ANN).\\n\\n"
            "**sparse** — только BM25 (ключевые слова)."
        ),
    )

    top_k = st.slider("top_k — финальных результатов", 1, 20, 10)

    prefetch_k = top_k * 4
    if mode == "hybrid":
        prefetch_k = st.slider(
            "prefetch_k — кандидатов для ColBERT rerank",
            min_value=top_k,
            max_value=100,
            value=min(top_k * 4, 100),
            help="Кандидатов из dense+sparse перед rerank. Больше = точнее, медленнее.",
        )

    st.divider()

    if mode == "hybrid":
        st.markdown(
            "**Схема:**\n"
            "```\n"
            "dense  Prefetch ─┐\n"
            "                 ├► ColBERT rerank → top_k\n"
            "sparse Prefetch ─┘\n"
            "```\n"
            "ColBERT — reranker, не ANN-индекс."
        )
    else:
        st.info(f"Режим **{mode}**: прямой поиск без rerank.")

    st.divider()
    st.markdown("**Фильтры**")

    table_filter = st.radio(
        "Тип чанков",
        options=["Все", "Только текст", "Только таблицы"],
        index=0,
        horizontal=True,
    )
    only_tables = {"Все": None, "Только текст": False, "Только таблицы": True}[
        table_filter
    ]

    filename_filter = (
        st.text_input(
            "Фильтр по файлу (filename)",
            placeholder="sp_63_13330",
        )
        or None
    )

    st.divider()
    st.caption(
        f"🔗 `{retriever.collection}`\\n\\n📦 mpnet-multilingual · BM25 · ColBERTv2"
    )

# ── Основной контент ──────────────────────────────────────────────────────────
st.title("🏗️ Поиск по нормативной базе")
st.caption("СП / ГОСТ / СНиП — hybrid search + ColBERT rerank через Qdrant")

query = st.text_input(
    "Запрос", placeholder="арматура для железобетона", label_visibility="collapsed"
)

col_btn, col_clear, _ = st.columns([1, 1, 6])
with col_btn:
    search_btn = st.button("🔍 Найти", type="primary", use_container_width=True)
with col_clear:
    if st.button("✕ Сбросить", use_container_width=True):
        st.session_state.pop("results", None)
        st.session_state.pop("meta", None)
        st.rerun()

# ── Выполнить поиск ───────────────────────────────────────────────────────────
if search_btn and query.strip():
    label = (
        f"hybrid + ColBERT rerank (prefetch_k={prefetch_k})"
        if mode == "hybrid"
        else mode
    )
    with st.spinner(f"Режим: {label}, top_k={top_k}…"):
        results = retriever.search(
            query=query,
            top_k=top_k,
            prefetch_k=prefetch_k,
            mode=mode,
            only_tables=only_tables,
            filename_filter=filename_filter,
        )
    st.session_state["results"] = results
    st.session_state["meta"] = dict(
        query=query,
        mode=mode,
        top_k=top_k,
        prefetch_k=prefetch_k if mode == "hybrid" else None,
        only_tables=only_tables,
        filename_filter=filename_filter,
    )

# ── Результаты ────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results: list[RetrievalResult] = st.session_state["results"]
    meta = st.session_state.get("meta", {})

    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Найдено", len(results))
    m2.metric("Режим", meta.get("mode", "—").upper())
    m3.metric(
        "prefetch_k",
        meta.get("prefetch_k", "—") if meta.get("mode") == "hybrid" else "—",
    )
    m4.metric("Таблиц", sum(1 for r in results if r.is_table))
    m5.metric("Макс. score", f"{max((r.score for r in results), default=0):.4f}")

    st.markdown(f"##### Результаты для: *«{meta.get('query', '')}»*")

    if not results:
        st.info("Ничего не найдено.")
    else:
        for idx, r in enumerate(results, start=1):
            mode_used = meta.get("mode", "hybrid")
            if mode_used == "hybrid":
                dot = "🟢" if r.score >= 15 else ("🟡" if r.score >= 8 else "🔴")
            else:
                dot = "🟢" if r.score >= 0.02 else ("🟡" if r.score >= 0.01 else "🔴"))
            kind = "📊 Таблица" if r.is_table else "📄 Текст"
            lbl = (
                f"{dot} **#{idx}** &nbsp; `score: {r.score:.4f}` &nbsp;·&nbsp; "
                f"{kind} &nbsp;·&nbsp; `{r.filename}` &nbsp;·&nbsp; chunk #{r.chunk_index}"
            )
            with st.expander(lbl, expanded=(idx <= 3)):
                if r.headings:
                    st.markdown(" › ".join(f"**{h}**" for h in r.headings))

                st.markdown("**Текст чанка**")
                if r.is_table:
                    st.code(r.text, language=None)
                else:
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-left:3px solid #4CAF50;'
                        f'padding:10px 14px;border-radius:4px;font-size:0.93rem;">'
                        f"{r.text}</div>",
                        unsafe_allow_html=True,
                    )

                if r.refs:
                    st.markdown("**Нормативные ссылки:**")
                    badges = " &nbsp; ".join(
                        f'<code style="background:#e3f2fd;padding:2px 6px;border-radius:3px;">{ref}</code>'
                        for ref in r.refs
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                with st.popover("🔍 Метаданные (JSON)"):
                    st.json(
                        {
                            "id": r.id,
                            "score": r.score,
                            "filename": r.filename,
                            "chunk_index": r.chunk_index,
                            "is_table": r.is_table,
                            "headings": r.headings,
                            "refs": r.refs,
                        }
                    )
else:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#888;">'
        '<div style="font-size:3rem;">🔍</div>'
        '<p style="font-size:1.1rem;margin-top:12px;">'
        "Введите запрос и нажмите <strong>Найти</strong></p>"
        '<p style="font-size:0.85rem;">hybrid (dense + sparse → ColBERT rerank) · dense · sparse</p>'
        "</div>",
        unsafe_allow_html=True,
    )
