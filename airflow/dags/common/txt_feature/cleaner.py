import logging
import re

MAX_TOKENS = 512
MIN_WORDS = 8
MIN_WORDS_MERGE = 15

_NOISE_HEADINGS = re.compile(
    r"сведения о (стандарте|своде правил|нормативном документе|документе)|"
    r"предисловие|foreword|"
    r"библиография|bibliography|"
    r"^приложение\s*[а-яёa-z]?$|"
    r"дата введения|"
    r"термины и определения",
    re.IGNORECASE,
)
_FIGURE_CAPTION = re.compile(r"^\d+\s*[-–-]\s+\S+")


class ChunkCleaner:
    """
    A utility class for cleaning, filtering,
    and merging chunks before indexing them in Qdrant.

    """

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Remove common OCR artefacts from chunk text before vectorisation.

        Normalises excess whitespace, collapses repeated newlines, fixes
        hyphenated number ranges, and strips repeated pipe characters.

        Parameters
        ----------
        text : str
            Raw OCR text from a document chunk.

        Returns
        -------
        str
            Cleaned text, or the original value if it was empty / falsy.
        """
        if not text:
            return text
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1–\2", text)
        text = re.sub(r"[|]{2,}", "", text)
        return text.strip()

    @classmethod
    def extract_refs(cls, text: str) -> list[str]:
        """
        Extract normative document references from chunk text.

        Searches for Russian building-code patterns: СП, СНиП, ГОСТ,
        paragraph numbers, table / figure references, and appendix labels.

        Parameters
        ----------
        text : str
            Plain text of a document chunk.

        Returns
        -------
        list of str
            Deduplicated list of matched reference strings.
            Returns an empty list when no patterns are found.
        """
        patterns = [
            r"[Сс][Пп]\s*\d+[\.\d]*",  # СП 63.13330
            r"[СсГг][НнОо][ИиСс][ПпТт]\s*[\d\.\-]+",  # СНиП, ГОСТ
            r"[Пп]\.?\s*\d+[\.д]*",  # п. 3.45
            r"[Тт]абл(?:ица|\.)\s*\d+",  # таблица 7
            r"[Рр]ис(?:унок|\.)\s*\d+",  # рисунок 3
            r"[Пп]риложени[еяй]\s*[А-ЯA-Z\d]+",  # приложение А
        ]
        refs: list[str] = []
        for pat in patterns:
            refs.extend(re.findall(pat, text))
        return list(set(refs))

    @classmethod
    def count_tokens(cls, text: str) -> int:
        return int(len(text.split()) * 1.3)

    @classmethod
    def strip_heading_prefix(cls, text: str, headings: list[str]) -> str:
        if not headings:  # ← защита от None и []
            return text or ""
        if not text:
            return ""
        for h in headings:
            if h and text.startswith(h):
                text = text[len(h) :].strip()
        return text

    @classmethod
    def is_noise(cls, chunk: dict) -> bool:
        """
        True if the chunk is garbage.

         Criteria:
        1. Too short (< MIN_WORDS words after removing the heading-prefix)
        2. Header section (preface, bibliography, etc.)
        3. Formula fragment: >60% of tokens are Latin/Cyrillic
           variables with a length of ≤2 characters (E s 0, R s w)
        """
        headings = chunk.get("headings", [])
        text = cls.strip_heading_prefix(chunk.get("text", ""), headings)
        words = text.split()

        # 1. Слишком короткий
        if len(words) < MIN_WORDS:
            return True

        # 2. Служебный заголовок
        for h in headings:
            if _NOISE_HEADINGS.search(h):
                return True
            if _FIGURE_CAPTION.match(h):
                return True

        # 3. Обломки формул
        alpha_words = [re.sub(r"[^а-яёa-z]", "", w.lower()) for w in words]
        short = sum(1 for w in alpha_words if len(w) <= 2)
        if len(words) > 0 and short / len(words) > 0.6:
            return True

        return False

    @classmethod
    def merge_by_section(
        cls,
        chunks: list[dict],
        max_tokens: int = MAX_TOKENS,
        min_words: int = MIN_WORDS_MERGE,
    ) -> list[dict]:
        """
        Объединяет соседние чанки одного раздела (одинаковые headings)
        пока суммарный размер < max_tokens токенов.

        Таблицы (is_table=True) никогда не мержатся - идут отдельно.
        """
        result: list[dict] = []
        buffer: dict | None = None

        for chunk in chunks:
            # Tables are always separate
            if chunk.get("is_table"):
                if buffer is not None:
                    result.append(buffer)
                    buffer = None
                result.append(chunk)
                continue

            if buffer is None:
                buffer = {
                    **chunk,
                    "headings": chunk.get("doc_items", []),
                    "doc_items": list(chunk.get("doc_items", [])),
                }
                continue

            same_section = buffer.get("headings") == chunk.get("headings")
            buf_tokens = cls.count_tokens(buffer.get("text", ""))
            new_tokens = cls.count_tokens(chunk.get("text", ""))

            if same_section and (buf_tokens + new_tokens) < max_tokens:
                # Только содержательная часть без heading
                addition = cls.strip_heading_prefix(
                    chunk.get("text", ""), chunk.get("headings", [])
                )
                buffer["text"] += "\n" + addition
                buffer["doc_items"].extend(chunk.get("doc_items", []))
                buffer["refs"] = list(
                    set(buffer.get("refs", []) + chunk.get("refs", []))
                )
            else:
                if len(buffer.get("text", "").split()) >= min_words:
                    result.append(buffer)
                buffer = {**chunk, "doc_items": list(chunk.get("doc_items", []))}

        if buffer and len(buffer.get("text", "").split()) >= min_words:
            result.append(buffer)

        return result

    @classmethod
    def process(cls, chunks: list[dict]) -> list[dict]:
        """
        A fully pipeline for cleaning chanks by a single document.

        Steps
        -------------
        1. clean_text       - remove OCR artifacts from the text
        2. extract_refs     - (re)calculate the reference norms
        3. merge_by_section - glue micro-chunks of one section
        4. filter noise     - remove garbage chunks
        5. reindex          - recalculate the chunk_index
        6. count_tokens     - write the num_tokens to each chunk

        Returns
        -------
        list[dict]
            A cleaned list of chunks ready for indexing.
        """
        # Step 1 - 2
        for c in chunks:
            c["text"] = cls.clean_text(c.get("text", ""))
            c["refs"] = cls.extract_refs(c.get("text", ""))
            c["headings"] = c.get("headings", [])
            c["doc_items"] = c.get("doc_items", [])

        before = len(chunks)

        # Step 3
        chunks = cls.merge_by_section(chunks)

        # Step 4
        chunks = [c for c in chunks if not cls.is_noise(c)]

        # Step 5
        for idx, c in enumerate(chunks):
            c["chunk_index"] = idx
            c["num_tokens"] = cls.count_tokens(c.get("text", ""))

        logging.info(
            f"ChunkCleaner.process: {before} → {len(chunks)} chunks ({before - len(chunks)} removed/merged)"
        )
        return chunks
