"""
src/preprocessing/corpus_builder.py
────────────────────────────────────
Extracts and structures the Bangladesh Constitution from PDF into
article-level JSON chunks ready for retrieval indexing and annotation.

Usage:
    python -m src.preprocessing.corpus_builder \
        --pdf data/raw/constitution_bn.pdf \
        --output data/processed/corpus.json \
        --lang bn

    # No PDF? Use built-in sample:
    python -m src.preprocessing.corpus_builder \
        --sample \
        --output data/processed/corpus.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

# PyMuPDF — only needed for PDF mode, not for --sample
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# csebuetnlp Bangla normalizer — optional but recommended
try:
    from normalizer import normalize
    HAS_NORMALIZER = True
except ImportError:
    HAS_NORMALIZER = False
    logger.warning(
        "csebuetnlp normalizer not found — Unicode normalization skipped.\n"
        "Install with: pip install git+https://github.com/csebuetnlp/normalizer"
    )


# ── Bangla Unicode range ─────────────────────────────────────────────────────
BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]+')

# ── Article header patterns ───────────────────────────────────────────────────
# Bangla: matches "অনুচ্ছেদ ১।"  "অনুচ্ছেদ ১ক।"
ARTICLE_PATTERN_BN = re.compile(
    r'অনুচ্ছেদ\s+([\u09E6-\u09EF\d]+[কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ]?)[।\.]',
    re.UNICODE
)
# English: matches "Article 7A. Title" or "7A. Title"
ARTICLE_PATTERN_EN = re.compile(
    r'(?:Article\s+)?(\d+[A-Z]?)\.\s+([A-Z][^\n]+)',
    re.IGNORECASE
)

PART_PATTERN_BN = re.compile(
    r'(প্রথম|দ্বিতীয়|তৃতীয়|চতুর্থ|পঞ্চম|ষষ্ঠ|সপ্তম|অষ্টম|নবম|দশম|একাদশ)\s*ভাগ',
    re.UNICODE
)
PART_PATTERN_EN = re.compile(r'PART\s+([IVXLC]+)\s*[\n\r]', re.IGNORECASE)


class ConstitutionCorpusBuilder:
    """Parses the Bangladesh Constitution PDF into structured article chunks."""

    ARTICLE_TITLES_EN: Dict[str, str] = {
        "1":    "Name and territory of the Republic",
        "2":    "State religion",
        "2A":   "State religion",
        "3":    "State language",
        "4":    "National anthem, flag and emblem",
        "5":    "Capital",
        "6":    "Citizenship",
        "7":    "Supremacy of the Constitution",
        "7A":   "Offence of abeyance of the Constitution",
        "8":    "Fundamental principles",
        "9":    "Nationalism",
        "10":   "Socialism",
        "11":   "Democracy and human rights",
        "12":   "Secularism and freedom of religion",
        "13":   "Principles of ownership",
        "14":   "Emancipation of peasants and workers",
        "15":   "Provision of basic necessities",
        "16":   "Rural development and agricultural revolution",
        "17":   "Free and compulsory education",
        "18":   "Public health and morality",
        "18A":  "Protection and improvement of environment and biodiversity",
        "19":   "Equality of opportunity",
        "20":   "Work as a right and duty",
        "21":   "Duties of citizens and of public servants",
        "22":   "Separation of judiciary from executive",
        "23":   "National culture",
        "23A":  "Culture of tribes, minor races, ethnic sects and communities",
        "24":   "National monuments, etc.",
        "25":   "Promotion of international peace, security and solidarity",
        "26":   "Laws inconsistent with fundamental rights to be void",
        "27":   "Equality before law",
        "28":   "Discrimination on grounds of religion, etc.",
        "29":   "Equality of opportunity in public employment",
        "30":   "Prohibition of foreign titles, etc.",
        "31":   "Right to protection of law",
        "32":   "Protection of right to life and personal liberty",
        "33":   "Safeguards as to arrest and detention",
        "34":   "Prohibition of forced labour",
        "35":   "Protection in respect of trial and punishment",
        "36":   "Freedom of movement",
        "37":   "Freedom of assembly",
        "38":   "Freedom of association",
        "39":   "Freedom of thought and conscience, and of speech",
        "40":   "Freedom of profession or occupation",
        "41":   "Freedom of religion",
        "42":   "Rights to property",
        "43":   "Protection of home and correspondence",
        "44":   "Enforcement of fundamental rights",
        "45":   "Modification of rights in respect of disciplinary law",
        "46":   "Power to indemnify",
        "47":   "Saving for certain laws",
        "47A":  "Inapplicability of certain articles",
        "48":   "The President",
        "49":   "Power of pardon, etc.",
        "50":   "Term of office of President",
        "65":   "Establishment of Parliament",
        "66":   "Qualifications and disqualifications for election to Parliament",
        "70":   "Vacation of seat on resignation or voting against party",
        "72":   "Sessions of Parliament",
        "80":   "Legislative procedure",
        "94":   "Establishment of Supreme Court",
        "95":   "Appointment of Judges",
        "96":   "Tenure of office of Judges",
        "99":   "Prohibition of practice in courts",
        "100":  "Seat of Supreme Court",
        "102":  "Powers of High Court Division",
        "116":  "Control and discipline of subordinate courts",
        "116A": "Independence of judiciary",
        "117":  "Administrative Tribunals",
        "118":  "Election Commission",
        "119":  "Functions of Election Commission",
        "141A": "Proclamation of emergency",
        "142":  "Amendment of the Constitution",
        "143":  "Property of the Republic",
        "152":  "Interpretation",
        "153":  "Commencement, citation and authentication",
    }

    PART_TITLES: Dict[str, str] = {
        "I":    "The Republic",
        "II":   "Fundamental Principles of State Policy",
        "III":  "Fundamental Rights",
        "IV":   "The Executive",
        "V":    "The Legislature",
        "VI":   "The Judiciary",
        "VII":  "Elections",
        "VIII": "The Comptroller and Auditor-General",
        "IX":   "The Civil Service of Bangladesh",
        "IXA":  "The Office of Ombudsman",
        "X":    "Amendment of Constitution",
        "XI":   "Miscellaneous",
    }

    def __init__(self, lang: str = "bn"):
        self.lang = lang
        self.articles: List[Dict] = []

    # ── PDF extraction ────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from all pages of a PDF using PyMuPDF."""
        if not HAS_FITZ:
            raise ImportError(
                "PyMuPDF is required to read PDFs.\n"
                "Install with: pip install PyMuPDF"
            )

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(path))
        pages = []
        total = len(doc)
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(text)
            logger.debug(f"Extracted page {page_num + 1}/{total}")
        doc.close()

        full_text = "\n".join(pages)
        logger.info(f"Extracted {len(full_text):,} characters from {total} pages")
        return full_text

    # ── Text normalisation ────────────────────────────────────────────────────

    def normalize_bangla(self, text: str) -> str:
        """
        Normalise Bangla Unicode text.
        Uses csebuetnlp normalizer if available, otherwise does basic cleanup.
        """
        if HAS_NORMALIZER:
            try:
                return normalize(text)
            except Exception as e:
                logger.warning(f"Normalizer failed ({e}), falling back to basic cleanup")
        # Basic fallback: collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ── Article splitting ─────────────────────────────────────────────────────

    def split_into_articles(self, text: str) -> List[Dict]:
        """
        Split raw extracted text into article-level chunks.
        Detects article headers in both English and Bangla PDFs.
        """
        lines = text.split('\n')
        articles: List[Dict] = []
        current_article: Optional[Dict] = None
        current_text_lines: List[str] = []
        current_part = "I"
        current_part_title = self.PART_TITLES.get("I", "")

        def _save_current():
            """Flush current_text_lines into current_article and append."""
            if current_article is not None:
                current_article["text"] = self.normalize_bangla(
                    " ".join(current_text_lines)
                )
                articles.append(current_article)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ── Detect Part boundary ──────────────────────────────────────────
            part_match = PART_PATTERN_EN.search(line)
            if part_match:
                current_part = part_match.group(1).upper()
                current_part_title = self.PART_TITLES.get(current_part, line)
                continue

            # ── Detect article header — English PDF ──────────────────────────
            art_en = ARTICLE_PATTERN_EN.match(line)
            if art_en:
                _save_current()
                current_text_lines = []

                art_num = art_en.group(1).upper()
                art_title = (
                    art_en.group(2).strip()
                    if art_en.group(2)
                    else self.ARTICLE_TITLES_EN.get(art_num, f"Article {art_num}")
                )
                current_article = {
                    "article_number":    art_num,
                    "part":              current_part,
                    "part_title":        current_part_title,
                    "title_en":          art_title,
                    "title_bn":          f"অনুচ্ছেদ {art_num}",
                    "text":              "",
                    "clauses":           [],
                }
                continue

            # ── Detect article header — Bangla PDF ───────────────────────────
            if self.lang == "bn":
                art_bn = ARTICLE_PATTERN_BN.search(line)
                if art_bn:
                    _save_current()
                    current_text_lines = []

                    art_num_bn = art_bn.group(1)
                    art_num_ar = self._bn_digit_to_arabic(art_num_bn)

                    current_article = {
                        "article_number":    art_num_ar,
                        "article_number_bn": art_num_bn,
                        "part":              current_part,
                        "part_title":        current_part_title,
                        "title_en":          self.ARTICLE_TITLES_EN.get(
                                                 art_num_ar, f"Article {art_num_ar}"
                                             ),
                        "title_bn":          f"অনুচ্ছেদ {art_num_bn}",
                        "text":              "",
                        "clauses":           [],
                    }
                    continue

            # ── Accumulate body text ──────────────────────────────────────────
            if current_article is not None:
                current_text_lines.append(line)

        # Save the very last article
        _save_current()

        logger.info(f"Extracted {len(articles)} articles")
        return articles

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bn_digit_to_arabic(self, text: str) -> str:
        """Convert Bangla-Indic digits ০১২৩৪৫৬৭৮৯ → 0123456789."""
        mapping = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
        return text.translate(mapping)

    def _enrich(self, articles: List[Dict]) -> List[Dict]:
        """Add id, seq, and passage fields to every article."""
        for i, art in enumerate(articles):
            art["id"]  = f"art_{art['article_number']}"
            art["seq"] = i
            art["passage"] = (
                f"অনুচ্ছেদ {art.get('article_number_bn', art['article_number'])} "
                f"— {art.get('title_en', '')}। {art['text']}"
            )
        return articles

    # ── Public build methods ──────────────────────────────────────────────────

    def build_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Full pipeline: PDF file → cleaned, enriched article list."""
        raw_text = self.extract_text_from_pdf(pdf_path)
        articles = self.split_into_articles(raw_text)
        self.articles = self._enrich(articles)
        return self.articles

    def build_from_manual_json(self, json_path: str) -> List[Dict]:
        """
        Load a manually structured corpus JSON.
        Use this when the PDF extractor gives poor results (e.g. scanned PDFs).

        Required fields per article: article_number, text, part
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

        with open(path, encoding="utf-8") as f:
            articles = json.load(f)

        for i, art in enumerate(articles):
            art.setdefault("id",  f"art_{art['article_number']}")
            art.setdefault("seq", i)
            art.setdefault(
                "title_en",
                self.ARTICLE_TITLES_EN.get(art["article_number"], "")
            )
            art["text"]    = self.normalize_bangla(art.get("text", ""))
            art["passage"] = (
                f"অনুচ্ছেদ {art.get('article_number_bn', art['article_number'])} "
                f"— {art.get('title_en', '')}। {art['text']}"
            )

        self.articles = articles
        return self.articles

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_path: str) -> None:
        """Write the article list to a UTF-8 JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.articles)} articles → {output_path}")

    def get_stats(self) -> Dict:
        """Return a summary of the current corpus."""
        parts: Dict[str, int] = {}
        total_chars = 0
        for art in self.articles:
            p = art.get("part", "?")
            parts[p] = parts.get(p, 0) + 1
            total_chars += len(art.get("text", ""))
        return {
            "total_articles":     len(self.articles),
            "parts":              parts,
            "total_characters":   total_chars,
            "avg_article_length": total_chars // max(len(self.articles), 1),
        }


# ── Built-in sample corpus (15 key articles) ─────────────────────────────────
# Used when the actual Constitution PDF is not available.

SAMPLE_CORPUS: List[Dict] = [
    {
        "article_number": "1",
        "article_number_bn": "১",
        "part": "I",
        "part_title": "The Republic",
        "title_en": "Name and territory of the Republic",
        "title_bn": "প্রজাতন্ত্রের নামকরণ ও রাজ্যসীমা",
        "text": "বাংলাদেশ একটি একক, স্বাধীন ও সার্বভৌম প্রজাতন্ত্র, যা 'গণপ্রজাতন্ত্রী বাংলাদেশ' নামে পরিচিত হইবে।",
    },
    {
        "article_number": "2A",
        "article_number_bn": "২ক",
        "part": "I",
        "part_title": "The Republic",
        "title_en": "State religion",
        "title_bn": "রাষ্ট্রধর্ম",
        "text": "প্রজাতন্ত্রের রাষ্ট্রধর্ম ইসলাম, তবে হিন্দু, বৌদ্ধ, খ্রীষ্টানসহ অন্যান্য ধর্ম পালনে রাষ্ট্র সমমর্যাদা ও সমঅধিকার নিশ্চিত করিবে।",
    },
    {
        "article_number": "3",
        "article_number_bn": "৩",
        "part": "I",
        "part_title": "The Republic",
        "title_en": "State language",
        "title_bn": "রাষ্ট্রভাষা",
        "text": "প্রজাতন্ত্রের রাষ্ট্রভাষা বাংলা।",
    },
    {
        "article_number": "7",
        "article_number_bn": "৭",
        "part": "I",
        "part_title": "The Republic",
        "title_en": "Supremacy of the Constitution",
        "title_bn": "সংবিধানের প্রাধান্য",
        "text": "প্রজাতন্ত্রের সকল ক্ষমতার মালিক জনগণ; এবং জনগণের পক্ষে সেই ক্ষমতার প্রয়োগ কেবল এই সংবিধানের অধীনে ও কর্তৃত্বে কার্যকর হইবে। এই সংবিধান প্রজাতন্ত্রের সর্বোচ্চ আইন এবং অন্য কোনো আইন যদি এই সংবিধানের সহিত অসামঞ্জস্যপূর্ণ হয়, তাহা হইলে সেই আইন যতখানি অসামঞ্জস্যপূর্ণ, ততখানি বাতিল হইবে।",
    },
    {
        "article_number": "27",
        "article_number_bn": "২৭",
        "part": "III",
        "part_title": "Fundamental Rights",
        "title_en": "Equality before law",
        "title_bn": "আইনের দৃষ্টিতে সমতা",
        "text": "সকল নাগরিক আইনের দৃষ্টিতে সমান এবং আইনের সমান আশ্রয় লাভের অধিকারী।",
    },
    {
        "article_number": "28",
        "article_number_bn": "২৮",
        "part": "III",
        "part_title": "Fundamental Rights",
        "title_en": "Discrimination on grounds of religion etc.",
        "title_bn": "ধর্ম প্রভৃতি কারণে বৈষম্য",
        "text": "কেবল ধর্ম, গোষ্ঠী, বর্ণ, নারী-পুরুষভেদ বা জন্মস্থানের কারণে কোনো নাগরিকের প্রতি রাষ্ট্র বৈষম্য প্রদর্শন করিবেন না।",
    },
    {
        "article_number": "32",
        "article_number_bn": "৩২",
        "part": "III",
        "part_title": "Fundamental Rights",
        "title_en": "Protection of right to life and personal liberty",
        "title_bn": "জীবন ও ব্যক্তিস্বাধীনতার অধিকাররক্ষণ",
        "text": "আইনানুযায়ী ব্যতীত জীবন ও ব্যক্তিস্বাধীনতা হইতে কোনো ব্যক্তিকে বঞ্চিত করা যাইবে না।",
    },
    {
        "article_number": "39",
        "article_number_bn": "৩৯",
        "part": "III",
        "part_title": "Fundamental Rights",
        "title_en": "Freedom of thought and conscience, and of speech",
        "title_bn": "চিন্তা ও বিবেকের স্বাধীনতা এবং বাক্স্বাধীনতা",
        "text": "চিন্তা ও বিবেকের স্বাধীনতার নিশ্চয়তা দান করা হইল। রাষ্ট্রের নিরাপত্তা, বিদেশী রাষ্ট্রসমূহের সহিত বন্ধুত্বপূর্ণ সম্পর্ক, জনশৃঙ্খলা, শালীনতা বা নৈতিকতার স্বার্থে কিংবা আদালত-অবমাননা, মানহানি বা অপরাধে প্ররোচনা সম্পর্কে আইনের দ্বারা আরোপিত যুক্তিসঙ্গত বাধানিষেধ-সাপেক্ষে প্রত্যেক নাগরিকের বাক্ ও প্রকাশের স্বাধীনতার অধিকারের এবং সংবাদক্ষেত্রের স্বাধীনতার নিশ্চয়তা দান করা হইল।",
    },
    {
        "article_number": "41",
        "article_number_bn": "৪১",
        "part": "III",
        "part_title": "Fundamental Rights",
        "title_en": "Freedom of religion",
        "title_bn": "ধর্মীয় স্বাধীনতা",
        "text": "আইন, জনশৃঙ্খলা ও নৈতিকতা-সাপেক্ষে প্রত্যেক নাগরিকের যেকোনো ধর্ম অবলম্বন, পালন বা প্রচারের অধিকার রহিয়াছে। প্রত্যেক ধর্মীয় সম্প্রদায় ও উপ-সম্প্রদায়ের নিজস্ব ধর্মীয় প্রতিষ্ঠান স্থাপন, রক্ষণ ও ব্যবস্থাপনার অধিকার রহিয়াছে।",
    },
    {
        "article_number": "48",
        "article_number_bn": "৪৮",
        "part": "IV",
        "part_title": "The Executive",
        "title_en": "The President",
        "title_bn": "রাষ্ট্রপতি",
        "text": "বাংলাদেশের একজন রাষ্ট্রপতি থাকিবেন, যিনি আইন-অনুযায়ী সংসদ-সদস্যদের দ্বারা নির্বাচিত হইবেন। রাষ্ট্রপতি পদে নির্বাচিত হইবার যোগ্য হইবেন না কোনো ব্যক্তি যদি তিনি পঁয়ত্রিশ বৎসর বয়স পূর্ণ না করিয়া থাকেন।",
    },
    {
        "article_number": "56",
        "article_number_bn": "৫৬",
        "part": "IV",
        "part_title": "The Executive",
        "title_en": "Cabinet",
        "title_bn": "মন্ত্রিসভা",
        "text": "প্রধানমন্ত্রীকে লইয়া একটি মন্ত্রিসভা থাকিবে এবং প্রধানমন্ত্রী ও অন্যান্য মন্ত্রীগণ রাষ্ট্রপতি কর্তৃক নিযুক্ত হইবেন। রাষ্ট্রপতি প্রধানমন্ত্রীকে নিয়োগ করিবেন এবং তাঁহার পরামর্শে অন্যান্য মন্ত্রীদের নিয়োগ করিবেন।",
    },
    {
        "article_number": "65",
        "article_number_bn": "৬৫",
        "part": "V",
        "part_title": "The Legislature",
        "title_en": "Establishment of Parliament",
        "title_bn": "সংসদ-প্রতিষ্ঠা",
        "text": "জাতীয় সংসদ নামে বাংলাদেশের একটি সংসদ থাকিবে এবং এই সংবিধানের বিধানাবলী-সাপেক্ষে প্রজাতন্ত্রের আইন-প্রণয়ন ক্ষমতা সংসদের উপর ন্যস্ত থাকিবে। সংসদ তিনশত আসন লইয়া গঠিত হইবে এবং আসনগুলিতে প্রত্যক্ষ নির্বাচনের মাধ্যমে নির্বাচিত সদস্যগণ থাকিবেন। পঞ্চাশটি আসন কেবলমাত্র মহিলা সদস্যদের জন্য সংরক্ষিত থাকিবে।",
    },
    {
        "article_number": "94",
        "article_number_bn": "৯৪",
        "part": "VI",
        "part_title": "The Judiciary",
        "title_en": "Establishment of Supreme Court",
        "title_bn": "সুপ্রীম কোর্ট-প্রতিষ্ঠা",
        "text": "বাংলাদেশ সুপ্রীম কোর্ট নামে বাংলাদেশের একটি সর্বোচ্চ আদালত থাকিবে। আপীল বিভাগ ও হাইকোর্ট বিভাগ লইয়া সুপ্রীম কোর্ট গঠিত হইবে। প্রধান বিচারপতি ও আপীল বিভাগে নিযুক্ত বিচারকগণকে লইয়া আপীল বিভাগ এবং প্রধান বিচারপতি ও অন্যান্য বিচারকগণকে লইয়া হাইকোর্ট বিভাগ গঠিত হইবে।",
    },
    {
        "article_number": "116A",
        "article_number_bn": "১১৬ক",
        "part": "VI",
        "part_title": "The Judiciary",
        "title_en": "Independence of judiciary",
        "title_bn": "বিচারকার্যে বিচারকগণের স্বাধীনতা",
        "text": "বিচারকার্য পরিচালনার ক্ষেত্রে বিচারকগণ স্বাধীন থাকিবেন এবং কেবলমাত্র এই সংবিধান ও আইনের অধীন হইবেন।",
    },
    {
        "article_number": "118",
        "article_number_bn": "১১৮",
        "part": "VII",
        "part_title": "Elections",
        "title_en": "Establishment of Election Commission",
        "title_bn": "নির্বাচন কমিশন প্রতিষ্ঠা",
        "text": "বাংলাদেশের একটি নির্বাচন কমিশন থাকিবে এবং উক্ত বিষয়ে প্রণীত যেকোনো আইনের বিধানাবলী-সাপেক্ষে রাষ্ট্রপতি প্রধান নির্বাচন কমিশনার ও অনধিক চারজন নির্বাচন কমিশনার নিয়োগ করিবেন।",
    },
    {
        "article_number": "142",
        "article_number_bn": "১৪২",
        "part": "X",
        "part_title": "Amendment of Constitution",
        "title_en": "Amendment of the Constitution",
        "title_bn": "সংবিধান-সংশোধন",
        "text": "সংসদের আইন দ্বারা এই সংবিধানের কোনো বিধান সংযোজন, পরিবর্তন, প্রতিস্থাপন বা রহিতকরণ দ্বারা সংশোধন করা যাইবে। সংবিধান সংশোধনের জন্য সংসদের মোট সদস্যসংখ্যার অন্যূন দুই-তৃতীয়াংশ ভোটের প্রয়োজন।",
    },
]


def build_sample_corpus(output_path: str) -> None:
    """Build and save the built-in sample corpus (no PDF needed)."""
    builder = ConstitutionCorpusBuilder()
    articles = builder._enrich(
        [{**art} for art in SAMPLE_CORPUS]   # shallow copy to avoid mutating the constant
    )
    builder.articles = articles
    builder.save(output_path)
    logger.info(f"Sample corpus: {len(articles)} articles saved → {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Bangladesh Constitution corpus JSON."
    )
    parser.add_argument("--pdf",         default=None,  help="Path to Constitution PDF")
    parser.add_argument("--manual-json", default=None,  help="Path to manually structured JSON")
    parser.add_argument("--sample",      action="store_true",
                        help="Use built-in sample corpus (no PDF needed)")
    parser.add_argument("--output",      default="data/processed/corpus.json",
                        help="Output path for corpus JSON")
    parser.add_argument("--lang",        default="bn", choices=["bn", "en"],
                        help="Language of the PDF (bn = Bangla, en = English)")
    args = parser.parse_args()

    if args.sample:
        build_sample_corpus(args.output)

    elif args.pdf:
        builder = ConstitutionCorpusBuilder(lang=args.lang)
        builder.build_from_pdf(args.pdf)
        logger.info(builder.get_stats())
        builder.save(args.output)

    elif args.manual_json:
        builder = ConstitutionCorpusBuilder(lang=args.lang)
        builder.build_from_manual_json(args.manual_json)
        logger.info(builder.get_stats())
        builder.save(args.output)

    else:
        parser.print_help()
        logger.error(
            "\nNo input specified. Use one of:\n"
            "  --sample                        (quick start, no PDF needed)\n"
            "  --pdf path/to/constitution.pdf  (full PDF extraction)\n"
            "  --manual-json path/to/data.json (pre-structured JSON)"
        )