from typing import Literal

import streamlit as st
import structlog
from openai import OpenAI
from retriever.retriever import QdrantRetriever, RetrievalResult

SearchMode = Literal["hybrid", "dense", "sparse"]
SCORE_THRESHOLD = 4.0
log = structlog.get_logger(__name__)


@st.cache_resource(show_spinner="Подключение к Qdrant и загрузка моделей…")
def _load_retriever() -> QdrantRetriever:
    return QdrantRetriever()


def _score_dot(score: float, mode: SearchMode) -> str:
    if mode == "hybrid":
        return "🟢" if score >= 15 else ("🟡" if score >= 8 else "🔴")
    return "🟢" if score >= 0.02 else ("🟡" if score >= 0.01 else "🔴")


class QueryRewriter:
    """
    Reformulates the user request using vllm-light.
    If the model is unhealthy, return the original query.
    """

    REWRITE_SYSTEM_PROMPT = """
    Ты — ассистент по переформулированию поисковых запросов для системы RAG.

    Твоя задача: преобразовать вопрос пользователя в точный поисковый запрос,
    пригодный для векторного поиска по нормативным документам (СП, ГОСТ, СНиП).

    Правила:
    - Верни ТОЛЬКО переформулированный запрос, без пояснений и кавычек.
    - Убери разговорные обороты («расскажи мне», «хочу узнать» и т.п.).
    - Сохрани все технические термины, номера стандартов, классы материалов.
    - Добавь релевантные синонимы и уточнения области применения, если очевидны.
    - Длина ответа — не более двух предложений.
    """
    REWRITER_BASE_URL = "http://vllm-light:8020/v1"
    REWRITER_MODEL = "query-rewriter"

    def __init__(self, timeout: float = 5.0) -> None:
        self.client = OpenAI(
            base_url=self.REWRITER_BASE_URL,
            api_key="",
            timeout=timeout,
        )

    def rewrite(self, query: str) -> tuple[str, bool]:
        try:
            resp = self.client.chat.completions.create(
                model=self.REWRITER_MODEL,
                messages=[
                    {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            if not rewritten:
                return query, False
            return rewritten, True
        except Exception as e:
            log.error("rewriter_error", query=query, error=str(e))
            return query, False


class SidebarParams:
    """Считывает все параметры поиска из боковой панели."""

    def __init__(self) -> None:
        with st.sidebar:
            st.title("Параметры поиска")

            self.mode: SearchMode = st.selectbox(
                "Режим поиска",
                options=["hybrid", "dense", "sparse"],
                index=0,
                help=(
                    "**hybrid** — dense + sparse Prefetch → ColBERT rerank (рекомендуется)\n\n"
                    "**dense** — только multilingual-mpnet-base-v2 (ANN).\n\n"
                    "**sparse** — только BM25 (ключевые слова)."
                ),
            )

            self.top_k: int = st.slider("top_k — финальных результатов", 1, 20, 10)
            self.prefetch_k: int = self._render_prefetch_k()

            # st.divider()
            # self._render_mode_schema()

            st.divider()
            self.only_tables, self.filename_filter = self._render_filters()

            st.divider()
            self.use_rewriter: bool = st.toggle(
                "Переформулировать запрос",
                value=True,
                help="Использует vllm-light для преобразования вопроса в поисковый запрос.",
            )

            st.divider()
            st.caption(
                f"Коллекция: `{_load_retriever().collection}`\n\n"
                "mpnet-multilingual · BM25 · ColBERTv2"
            )

    def _render_prefetch_k(self) -> int:
        if self.mode != "hybrid":
            return self.top_k * 4
        return st.slider(
            "prefetch_k — кандидатов для ColBERT rerank",
            min_value=self.top_k,
            max_value=100,
            value=min(self.top_k * 4, 100),
            help="Кандидатов из dense+sparse перед rerank. Больше = точнее, медленнее.",
        )

    def _render_mode_schema(self) -> None:
        if self.mode == "hybrid":
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
            st.info(f"Режим **{self.mode}**: прямой поиск без rerank.")

    def _render_filters(self) -> tuple[bool | None, str | None]:
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
            st.text_input("Фильтр по файлу (filename)", placeholder="sp_63_13330")
            or None
        )
        return only_tables, filename_filter


class SearchBar:
    """Поле ввода запроса и кнопки управления."""

    def __init__(self) -> None:
        self.query: str = st.text_input(
            "Запрос",
            placeholder="арматура для железобетона",
            label_visibility="collapsed",
        )
        col_search, col_clear, _ = st.columns([1, 1, 6])
        with col_search:
            self.search_clicked = st.button(
                "Найти", type="primary", use_container_width=True
            )
        with col_clear:
            if st.button("Сбросить", use_container_width=True):
                st.session_state.pop("results", None)
                st.session_state.pop("meta", None)
                st.rerun()

    @property
    def should_search(self) -> bool:
        return self.search_clicked and bool(self.query.strip())


class SearchRunner:
    """
    Опционально переформулирует запрос через QueryRewriter,
    затем выполняет поиск и кладёт результаты в session_state.
    """

    def __init__(self, retriever: QdrantRetriever) -> None:
        self._retriever = retriever
        self._rewriter = QueryRewriter()

    def run(self, query: str, params: SidebarParams) -> None:
        effective_query, rewritten = self._maybe_rewrite(query, params)
        self._show_rewrite_info(query, effective_query, rewritten)

        results = self._fetch_results(effective_query, params)
        self._validate_and_store(results, query, effective_query, params)

    def _maybe_rewrite(self, query: str, params: SidebarParams) -> tuple[str, bool]:
        if not params.use_rewriter:
            return query, False

        with st.spinner("Переформулирование запроса…"):
            return self._rewriter.rewrite(query)

    @staticmethod
    def _show_rewrite_info(original: str, effective: str, was_rewritten: bool) -> None:
        if was_rewritten:
            st.info(
                f"**Исходный запрос:** {original}\n\n"
                f"**Переформулированный:** {effective}"
            )

    def _fetch_results(
        self, query: str, params: SidebarParams
    ) -> list[RetrievalResult]:
        label = (
            f"hybrid + ColBERT rerank (prefetch_k={params.prefetch_k})"
            if params.mode == "hybrid"
            else params.mode
        )
        with st.spinner(f"Режим: {label}, top_k={params.top_k}…"):
            return self._retriever.search(
                query=query,
                top_k=params.top_k,
                prefetch_k=params.prefetch_k,
                mode=params.mode,
                only_tables=params.only_tables,
                filename_filter=params.filename_filter,
            )

    @staticmethod
    def _validate_and_store(
        results: list[RetrievalResult],
        original_query: str,
        effective_query: str,
        params: SidebarParams,
    ) -> None:
        if not results or results[0].score < SCORE_THRESHOLD:
            st.warning("Документ по этой теме отсутствует в базе")
            if results:
                st.caption(
                    f"Лучший скор: {results[0].score:.4f} (порог: {SCORE_THRESHOLD})"
                )
            st.stop()

        st.session_state["results"] = results
        st.session_state["meta"] = dict(
            query=original_query,
            effective_query=effective_query,
            mode=params.mode,
            top_k=params.top_k,
            prefetch_k=params.prefetch_k if params.mode == "hybrid" else None,
            only_tables=params.only_tables,
            filename_filter=params.filename_filter,
        )


class ResultsView:
    """Отображает метрики и список найденных чанков."""

    def render(self) -> None:
        if "results" not in st.session_state:
            self._render_empty_prompt()
            return

        results: list[RetrievalResult] = st.session_state["results"]
        meta: dict = st.session_state.get("meta", {})

        st.divider()
        self._render_metrics(results, meta)

        label_query = meta.get("effective_query") or meta.get("query", "")
        st.markdown(f"##### Результаты для: *«{label_query}»*")

        if not results:
            st.info("Ничего не найдено.")
        else:
            for idx, result in enumerate(results, start=1):
                self._render_chunk(idx, result, meta.get("mode", "hybrid"))

    def _render_metrics(self, results: list[RetrievalResult], meta: dict) -> None:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Найдено", len(results))
        m2.metric("Режим", meta.get("mode", "—").upper())
        m3.metric(
            "prefetch_k",
            meta.get("prefetch_k", "—") if meta.get("mode") == "hybrid" else "—",
        )
        m4.metric("Таблиц", sum(1 for r in results if r.is_table))
        m5.metric("Макс. score", f"{max((r.score for r in results), default=0):.4f}")

    def _render_chunk(
        self, idx: int, result: RetrievalResult, mode: SearchMode
    ) -> None:
        dot = _score_dot(result.score, mode)
        kind = "Таблица" if result.is_table else "Текст"
        label = (
            f"{dot} **#{idx}** &nbsp; `score: {result.score:.4f}` &nbsp;·&nbsp; "
            f"{kind} &nbsp;·&nbsp; `{result.filename}` &nbsp;·&nbsp; chunk #{result.chunk_index}"
        )
        with st.expander(label, expanded=(idx <= 3)):
            self._render_headings(result)
            self._render_text(result)
            self._render_refs(result)
            self._render_metadata_popover(result)

    def _render_headings(self, result: RetrievalResult) -> None:
        if result.headings:
            st.markdown(" › ".join(f"**{h}**" for h in result.headings))

    def _render_text(self, result: RetrievalResult) -> None:
        st.markdown("**Текст чанка**")
        if result.is_table:
            st.code(result.text, language=None)
        else:
            st.markdown(
                '<div style="background:#f8f9fa;border-left:3px solid #4CAF50;'
                'padding:10px 14px;border-radius:4px;font-size:0.93rem;">'
                f"{result.text}</div>",
                unsafe_allow_html=True,
            )

    def _render_refs(self, result: RetrievalResult) -> None:
        if not result.refs:
            return
        st.markdown("**Нормативные ссылки:**")
        badges = " &nbsp; ".join(
            f'<code style="background:#e3f2fd;padding:2px 6px;border-radius:3px;">{ref}</code>'
            for ref in result.refs
        )
        st.markdown(badges, unsafe_allow_html=True)

    def _render_metadata_popover(self, result: RetrievalResult) -> None:
        with st.popover("Метаданные (JSON)"):
            st.json(
                {
                    "id": result.id,
                    "score": result.score,
                    "filename": result.filename,
                    "chunk_index": result.chunk_index,
                    "is_table": result.is_table,
                    "headings": result.headings,
                    "refs": result.refs,
                }
            )

    @staticmethod
    def _render_empty_prompt() -> None:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#888;">'
            '<p style="font-size:1.1rem;margin-top:12px;">'
            "Введите запрос и нажмите <strong>Найти</strong></p>"
            '<p style="font-size:0.85rem;">hybrid (dense + sparse → ColBERT rerank) · dense · sparse</p>'
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Поиск по нормативной базе",
        page_icon="📋",
        layout="wide",
    )

    retriever = _load_retriever()
    params = SidebarParams()

    st.title("Поиск по нормативной базе")
    st.caption("СП / ГОСТ / СНиП — hybrid search + ColBERT rerank через Qdrant")

    search_bar = SearchBar()

    if search_bar.should_search:
        SearchRunner(retriever).run(search_bar.query, params)

    ResultsView().render()


if __name__ == "__main__":
    main()
