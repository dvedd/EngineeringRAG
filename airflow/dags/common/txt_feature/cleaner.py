import logging
import re

from transformers import AutoTokenizer

MAX_TOKENS = 512
MIN_WORDS = 8
MIN_WORDS_MERGE = 15

NOISE_HEADINGS = re.compile(
    r"сведения о (стандарте|своде правил|нормативном документе|документе)|"
    r"предисловие|foreword|"
    r"библиография|bibliography|"
    r"^приложение\s*[а-яёa-z]?$|"
    r"дата введения|"
    r"термины и определения",
    re.IGNORECASE,
)
FIGURE_CAPTION = re.compile(r"^\d+\s*[-–-]\s+\S+")
TECHEXPERT_WATERMARKS: tuple[str, ...] = (
    r"Внимание!\s*Документ включен в доказательную базу технического регламента\."
    r"ИС\s*«Техэксперт:[^»]*»\s*Интранет[^\n]*",
    r"Дополнительную информацию см\. в ярлыке\s*[«\"]Примечания[»\"][^\n]*",
)

MANDATORY_PATTERNS: list[str] = [
    # СНиП, ГОСТ
    r"[СсГг][НнОо][ИиСс][ПпТт]\s*[\d\.\-]+\.[\d\.]+(?:\s*\((?:пункт|п\.?|таблиц[аеу]|табл\.?)\s*[\d\.]+\))?",
    r"[СсГг][НнОо][ИиСс][ПпТт]\s*[\d\.\-]+",
    # СП — отдельно, иначе CROSS_PATTERNS съедает «П» как п.3.45
    r"(?:(?:пункт[аеу]?|п\.)\s*[\d]+(?:\.[\d]+)*\s+)?СП\s*[\d]+(?:\.[\d]+)*(?:\s*\((?:пункт[аеу]?|п\.?|таблиц[аеу]|табл\.?)\s*[\d\.]+\))?",
    r"СП\s*[\d]+(?:\.[\d]+)*",
    # СанПиН
    r"СанПи[нН]\s*[\d]+(?:\.[\d\-]+)*",
]

CROSS_PATTERNS: list[str] = [
    r"[Тт]абл(?:иц[аеу]|(?:иц)?\.)\s*\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*\b",
    r"[пП]\.\s*\d+(?:\.\d+)*(?:\.?[дД])?",
    # r"[Рр]ис(?:унок|\.)\s*\d+(?:\.\d+)*",  # рисунок 3
    r"[Пп]риложени[еяй]\s*[А-ЯA-Z\d]+",
]
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")


class ChunkCleaner:
    """
    A utility class for cleaning, filtering,
    and merging chunks before indexing them in Qdrant.

    """

    @classmethod
    def clean_text(cls, text: str, headings: list[str]) -> str:
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

        text = cls.normalize_headings(text)
        for pat in TECHEXPERT_WATERMARKS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)

        if headings:
            for h in headings:
                nh = re.escape(cls.normalize_headings(h))
                text = re.sub(rf"^{nh}\s*\n?", "", text, flags=re.MULTILINE)
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1–\2", text)
        text = re.sub(r"[|]{2,}", "", text)
        return text.strip()

    @classmethod
    def normalize_headings(cls, s: str) -> str:
        return s.replace("–", "-").replace("—", "-").replace("\xa0", " ")

    @classmethod
    def extract_mandatory_refs(cls, text: str) -> list[str]:
        """
        Extract all mandatoryreferences from chunk text
        По обязательным будет вестиcь углубленный поиск для углубдения контекста
        """
        refs: list[str] = []
        for pat in MANDATORY_PATTERNS + CROSS_PATTERNS:
            refs.extend(re.findall(pat, text))
        return list(set(refs))

    @classmethod
    def extract_cross_refs(cls, text: str) -> list[str]:
        """
        Extract only cross references.

        Кросс метрики долджны будут попать в контекст в неизменном виде.
        Рисунки потом будут тронсфармироваться в прямые ссылки. Mineru может созранять фото.
        Parameters
        ----------
        text : str
            Plain text of a document chunk.

        Returns
        -------
        list of str
            Deduplicated list of cross-reference strings.
        """
        refs: list[str] = []
        for pat in CROSS_PATTERNS:
            refs.extend(re.findall(pat, text))
        return list(set(refs))

    @classmethod
    def count_tokens(cls, text: str) -> int:
        return len(TOKENIZER.encode(text, add_special_tokens=False))

    @classmethod
    def strip_heading_prefix(cls, text: str, headings: list[str]) -> str:
        if not headings:
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

        if chunk.get("is_table"):
            return False

        if len(words) < MIN_WORDS:
            return True

        # Служебный заголовок
        for h in headings:
            if NOISE_HEADINGS.search(h):
                return True
            if FIGURE_CAPTION.match(h):
                return True

        # Обломки формул
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
                    "headings": chunk.get("headings", []),
                    "doc_items": list(chunk.get("doc_items", [])),
                }
                continue

            same_section = cls.normalize_headings(
                buffer.get("headings", [])
            ) == cls.normalize_headings(chunk.get("headings", []))
            buf_tokens = cls.count_tokens(buffer.get("text", ""))
            new_tokens = cls.count_tokens(chunk.get("text", ""))

            if same_section and (buf_tokens + new_tokens) < max_tokens:
                buffer["text"] += "\n" + chunk.get("text", "")
                buffer["doc_items"].extend(chunk.get("doc_items", []))
                buffer["refs"] = list(
                    set(buffer.get("refs", []) + chunk.get("refs", []))
                )
            else:
                if len(buffer.get("text", "").split()) >= min_words:
                    result.append(buffer)
                    buffer = {
                        **chunk,
                        "headings": chunk.get("headings", []),
                        "doc_items": list(chunk.get("doc_items", [])),
                        "refs": list(chunk.get("refs", [])),
                        "man_refs": list(chunk.get("man_refs", [])),
                        "cross_refs": list(chunk.get("cross_refs", [])),
                    }

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
            c["text"] = cls.clean_text(c.get("text", ""), c.get("headings", []))
            c["man_refs"] = cls.extract_mandatory_refs(c.get("text", ""))
            c["cross_refs"] = cls.extract_cross_refs(c.get("text", ""))
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
