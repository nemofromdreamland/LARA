"""Pre-retrieval input-classification gate for LARA chat.

Routes each incoming chat message BEFORE any embedding/retrieval/LLM work so that
non-medical and unsafe messages short-circuit the RAG pipeline. The gate is a
pure, synchronous, dependency-free function — deterministic, sub-millisecond, and
independently testable.

Priority (first match wins):
    1. degenerate input            → OFF_TOPIC   (gentle clarify prompt)
    2. safety: self-harm / harm    → SAFETY_SELF_HARM  (crisis reply)
    3. safety: medical emergency   → SAFETY_EMERGENCY  (emergency reply)
    4. prompt-injection / jailbreak→ OFF_TOPIC   (scope redirect, no prompt leak)
    5. non-English (non-safety)    → NON_ENGLISH
    6. meta / capability           → META
    7. greeting / thanks / closing → GREETING
    8. out-of-scope medical        → OUT_OF_SCOPE_MED
    9. off-topic / non-medical     → OFF_TOPIC
   10. else                        → MEDICAL   (run the RAG pipeline)

Safety detection (steps 2-3) is intentionally multilingual (EN + PT + ES) and
runs BEFORE the language check, so a non-English cry for help is never swallowed
by the English-only redirect. Safety/non-medical *replies* are fixed English
text, so the English-only answer policy still holds; only safety *detection* is
multilingual.

Injection (step 4) is checked before the scope checks so a non-session drug name
in a jailbreak payload ("...prescribe me xanax") can't reroute it; the gate never
reveals the system prompt or role-plays a prescriber.

Pattern-based safety detection is a best-effort mitigation, not a guarantee. The
curated patterns favor precision on idioms ("this headache is killing me",
"kill the pain", "dying to know" do NOT trigger) and recall on genuine signals
(err toward showing help). Likewise the off-topic patterns are deliberately
specific: anything not positively recognised falls through to MEDICAL, so a real
medical question is never misrouted (worst case it hits the existing retrieval
fallback).
"""

import re
from dataclasses import dataclass
from enum import Enum

from app.utils import whole_word_match


class Route(str, Enum):
    SAFETY_SELF_HARM = "safety_self_harm"  # rule 1 (incl. harm facilitation)
    SAFETY_EMERGENCY = "safety_emergency"  # rule 2
    NON_ENGLISH = "non_english"  # rule 3
    META = "meta"  # rule 4
    GREETING = "greeting"  # rule 5 (greeting / thanks / closing)
    OUT_OF_SCOPE_MED = (
        "out_of_scope_medical"  # rule 6 (drug-not-here / dx / rec / cost)
    )
    OFF_TOPIC = "off_topic"  # rule 7 (incl. injection / jailbreak / degenerate)
    MEDICAL = "medical"  # rule 8 → run RAG (default)


@dataclass(frozen=True)
class GateResult:
    route: Route
    reply: str | None  # canned English text; None only for Route.MEDICAL


# ── Canned English replies ───────────────────────────────────────────────────
# Safety replies are generic — they intentionally carry NO hardcoded phone
# numbers (those are region-specific and go stale); they point to local services.

_REPLY_CRISIS = (
    "If you're thinking about harming yourself, please reach out right now to "
    "your local emergency services or a crisis helpline in your area — you don't "
    "have to face this alone. I can't help with anything that could cause harm, "
    "but I'm here for questions about your uploaded leaflets when you're ready."
)
_REPLY_EMERGENCY = (
    "This may be a medical emergency. Please contact your local emergency services "
    "right away — I can't provide emergency care. Once you're safe, I can answer "
    "questions about your uploaded leaflets."
)
_REPLY_NON_ENGLISH = (
    "This information is not available in the provided leaflets. I can only answer "
    "questions in English about your uploaded drug leaflets."
)
_REPLY_GREETING = (
    "Hi! Ask me anything about your uploaded drug leaflets — dosage, side effects, "
    "warnings, or interactions."
)
_REPLY_THANKS = (
    "You're welcome! Let me know if you have any other questions about your leaflets."
)
_REPLY_CLOSING = "Take care! Come back anytime with questions about your medications."
_REPLY_DIAGNOSIS = (
    "I can't diagnose conditions or assess symptoms — please talk to your doctor or "
    "pharmacist. I can only share what your uploaded leaflets say."
)
_REPLY_RECOMMENDATION = (
    "I can't recommend which medication to take — please check with your doctor or "
    "pharmacist. I can only tell you what your uploaded leaflets say."
)
_REPLY_COST = (
    "Cost, insurance, and pharmacy details aren't covered in the drug leaflets — "
    "your pharmacy can help with those. I can answer questions about what the "
    "leaflets say."
)
_REPLY_OFF_TOPIC = (
    "I can only help with questions about your uploaded drug leaflets — things like "
    "dosage, side effects, warnings, or interactions."
)
_REPLY_DEGENERATE = (
    "I didn't catch a question there. Try asking about your uploaded leaflets — for "
    'example, "What are the side effects?" or "Can I take these together?"'
)


def _meta_reply(drugs_found: list[str]) -> str:
    base = (
        "Ask me about dosage, side effects, warnings, or interactions. I'm not a "
        "substitute for professional medical advice."
    )
    if drugs_found:
        listed = ", ".join(drugs_found)
        return (
            "I'm LARA. I answer questions using the official FDA leaflets for the "
            f"medications in your prescription. In this session I can tell you about: "
            f"{listed}. {base}"
        )
    return (
        "I'm LARA. Upload a prescription and I'll read the official FDA leaflets so I "
        f"can answer questions about your medications. {base}"
    )


def _foreign_drug_reply(drug: str, drugs_found: list[str]) -> str:
    listed = ", ".join(drugs_found)
    return (
        f"I don't have a leaflet for {drug} in this session. I can answer questions "
        f"about: {listed}."
    )


# ── Degenerate input ─────────────────────────────────────────────────────────

# Any Unicode letter (incl. non-Latin scripts) — absence ⇒ emoji/punctuation/digits.
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_PLACEHOLDER = {
    "test",
    "testing",
    "asdf",
    "asdfg",
    "asdfgh",
    "qwerty",
    "hjkl",
    "aaa",
    "abc",
    "blah",
    "foo",
    "bar",
}


def _is_degenerate(q: str, low: str) -> bool:
    if not q:
        return True
    if not _LETTER_RE.search(q):  # emoji / punctuation / digits only
        return True
    stripped = re.sub(r"[^\w\s]", "", low).strip()
    return stripped in _PLACEHOLDER


# ── Safety: self-harm / harm facilitation (multilingual) ─────────────────────
# Patterns require the reflexive/possessive object ("kill MYSELF", "MY life") or a
# specific phrase, so "this headache is killing me" / "kill the pain" / "dying to
# know" do NOT match.

_SELF_HARM_RE = re.compile(
    # English — self-harm intent
    r"\bkill(?:ing)?\s+my\s*self\b"
    r"|\bkilling\s+myself\b"
    r"|\bend(?:ing)?\s+(?:my|my\s+own)\s+life\b"
    r"|\btake\s+my\s+(?:own\s+)?life\b"
    r"|\b(?:want|wanna|going|going\s+to|trying)\s+to\s+die\b"
    r"|\bno\s+reason\s+to\s+live\b"
    r"|\bdon'?t\s+want\s+to\s+(?:live|be\s+here|exist|be\s+alive)\b"
    r"|\bbetter\s+off\s+dead\b"
    r"|\bsuicid\w*\b"
    r"|\b(?:harm|hurt|cut)\s+myself\b"
    r"|\bself[\s-]?harm\b"
    r"|\bend\s+it\s+all\b"
    # Portuguese
    r"|\bme\s+matar\b|\bme\s+suicidar\b|\bme\s+machucar\b"
    r"|\bquero\s+morrer\b|\bvou\s+me\s+matar\b"
    r"|\btirar\s+(?:a\s+)?minha\s+vida\b"
    r"|\bsuic[íi]d\w*\b"
    r"|\bn[ãa]o\s+quero\s+(?:mais\s+)?viver\b"
    r"|\bacabar\s+com\s+tudo\b"
    # Spanish
    r"|\bmatarme\b|\bsuicidarme\b|\bquiero\s+morir\b"
    r"|\bquitarme\s+la\s+vida\b|\bhacerme\s+da[ñn]o\b|\bno\s+quiero\s+vivir\b",
    re.IGNORECASE,
)

_HARM_FACILITATION_RE = re.compile(
    # English — facilitate overdose / lethality / getting high
    r"\b(?:lethal|fatal|deadly|toxic)\s+dose\b"
    r"|\bhow\s+(?:much|many)\b[^?.!]{0,40}\b(?:kill|overdose|over\s*dose|die|fatal|lethal|od)\b"
    r"|\bhow\s+(?:do\s+i\s+|can\s+i\s+|to\s+)?overdose\b"
    r"|\boverdose\s+on\b"
    r"|\benough\b[^?.!]{0,30}\bto\s+kill\b"
    r"|\bhow\s+much\b[^?.!]{0,30}\bis\s+(?:lethal|fatal|deadly)\b"
    r"|\bhow\s+to\s+get\s+high\b|\bget\s+high\s+(?:on|off|from)\b"
    r"|\bhow\s+(?:much|many)\b[^?.!]{0,40}\bget\s+high\b"
    # Portuguese
    r"|\bdose\s+(?:letal|fatal|mortal)\b"
    r"|\bquant[oa]s?\b[^?.!]{0,30}\b(?:para\s+)?morrer\b"
    r"|\bcomo\s+(?:tomar\s+)?(?:uma\s+)?overdose\b"
    # Spanish
    r"|\bdosis\s+(?:letal|mortal|fatal)\b"
    r"|\bsobredosis\b"
    r"|\bcu[áa]nt[oa]s?\b[^?.!]{0,30}\bpara\s+morir\b",
    re.IGNORECASE,
)

# ── Safety: medical emergency (multilingual) ─────────────────────────────────

_EMERGENCY_RE = re.compile(
    # English
    r"\bcan'?t\s+breathe\b|\bcannot\s+breathe\b|\bunable\s+to\s+breathe\b"
    r"|\b(?:trouble|difficulty|hard|struggling)\s+(?:to\s+)?breath\w*\b"
    r"|\bchest\s+pain\b|\bpain\s+in\s+my\s+chest\b|\btight(?:ness\s+in\s+my)?\s+chest\b"
    r"|\bheart\s+attack\b"
    r"|\banaphyla\w*\b|\bsevere\s+allergic\s+reaction\b"
    r"|\bthroat\s+(?:is\s+)?(?:closing|swelling|swollen)\b|\bcan'?t\s+swallow\b"
    r"|\bunconscious\b|\bunresponsive\b|\bpass(?:ed|ing)\s+out\b|\bfaint(?:ed|ing)\b"
    r"|\bcollaps\w+\b"
    r"|\b(?:having|signs?\s+of)\s+a\s+stroke\b|\bstroke\b"
    r"|\bslurred\s+speech\b|\bface\s+(?:is\s+)?drooping\b"
    r"|\bsevere\s+bleeding\b|\bwon'?t\s+stop\s+bleeding\b"
    r"|\bbleeding\s+(?:a\s+lot|heavily|badly|nonstop)\b|\bcan'?t\s+stop\s+the\s+bleeding\b"
    # overdose-as-emergency (accidental or intentional "I took too much")
    r"|\btook\s+too\s+(?:much|many)\b|\btook\s+an?\s+extra\b|\baccidentally\s+took\b"
    r"|\btook\s+\d+\b[^?.!]{0,40}\b(?:by\s+mistake|by\s+accident|too\s+many|not\s+sure)\b"
    r"|\bi\s+(?:think\s+i\s+)?overdosed\b|\bdouble[\s-]?dosed\b"
    # Portuguese
    r"|\bn[ãa]o\s+consigo\s+respirar\b|\bdor\s+no\s+peito\b"
    r"|\bdesmai\w*\b|\binconsciente\b|\bataque\s+card\w*\b|\bderrame\b"
    r"|\btomei\s+(?:demais|muitos?|d\w+\s+demais)\b"
    r"|\btomei\s+\d+\b[^?.!]{0,40}\b(?:por\s+engano|sem\s+querer|por\s+acidente)\b"
    r"|\brea[çc][ãa]o\s+al[ée]rgica\s+grave\b|\bsangramento\s+(?:grave|forte|intenso)\b"
    # Spanish
    r"|\bno\s+puedo\s+respirar\b|\bdolor\s+en\s+el\s+pecho\b"
    r"|\bdesmay\w*\b|\bataque\s+al\s+coraz[óo]n\b|\btom[ée]\s+(?:demasiado|muchos?)\b",
    re.IGNORECASE,
)

# ── Prompt-injection / jailbreak ─────────────────────────────────────────────

_INJECTION_RE = re.compile(
    r"\bignore\s+(?:all\s+|your\s+|the\s+|any\s+|previous\s+|prior\s+|above\s+)*"
    r"(?:instructions?|prompts?|rules?|guidelines?|directions?)\b"
    r"|\bdisregard\s+(?:all\s+|your\s+|the\s+|previous\s+)*(?:instructions?|rules?|prompts?)\b"
    r"|\bforget\s+(?:all\s+|your\s+|the\s+|everything\s+|previous\s+)*"
    r"(?:instructions?|rules?|what\s+you)\b"
    r"|\b(?:system|initial|original)\s+prompt\b"
    r"|\b(?:print|show|reveal|repeat|display|tell\s+me|give\s+me|what(?:'s| is)|share)"
    r"\b[^?.!]{0,30}\b(?:your\s+)?"
    r"(?:system\s+prompt|prompt|instructions?|rules?|guidelines?)\b"
    r"|\bact\s+as\b[^?.!]{0,30}\b(?:doctor|physician|prescrib\w+|pharmacist|nurse)\b"
    r"|\bpretend\s+(?:you(?:'re| are)|to\s+be)\b"
    r"|\brole[\s-]?play\b|\bjailbreak\b|\bdeveloper\s+mode\b"
    r"|\byou\s+are\s+now\b|\bprescribe\s+me\b|\bwrite\s+me\s+a\s+prescription\b",
    re.IGNORECASE,
)

# ── Language (heuristic; default-to-English) ─────────────────────────────────

_NON_LATIN_RE = re.compile(
    r"[Ѐ-ӿͰ-Ͽ؀-ۿ֐-׿"
    r"ऀ-ॿ぀-ヿㇰ-ㇿ一-鿿가-힯฀-๿]"
)
# Diacritics distinctive to Portuguese/Spanish (rare in English) — fire alone.
_PTES_STRONG_DIACRITIC_RE = re.compile(r"[ãõñç¿¡]", re.IGNORECASE)
_ACCENT_RE = re.compile(r"[áéíóúâêôü]", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zà-ÿ]+")

_STRONG_FOREIGN = {
    "não",
    "nao",
    "você",
    "voce",
    "gravidez",
    "efeitos",
    "efeito",
    "obrigado",
    "obrigada",
    "dosagem",
    "remédio",
    "remedio",
    "comprimido",
    "saúde",
    "saude",
    "médico",
    "medico",
    "gracias",
    "embarazo",
    "medicamento",
    "síntomas",
    "sangre",
}
_WEAK_FOREIGN = {
    "tem",
    "em",
    "com",
    "para",
    "posso",
    "devo",
    "pode",
    "quero",
    "que",
    "qué",
    "como",
    "cuánto",
    "cuanto",
    "está",
    "esta",
    "dosis",
    "tomar",
    "fazer",
    "isso",
    "meu",
    "minha",
    "são",
    "sao",
    "muito",
    "mais",
    "sem",
    "tomei",
    "quais",
    "qual",
}


def _looks_non_english(q: str) -> bool:
    if _NON_LATIN_RE.search(q):
        return True
    if _PTES_STRONG_DIACRITIC_RE.search(q):
        return True
    tokens = set(_WORD_RE.findall(q.lower()))
    if tokens & _STRONG_FOREIGN:
        return True
    weak = len(tokens & _WEAK_FOREIGN)
    if weak >= 2:
        return True
    if weak >= 1 and _ACCENT_RE.search(q):
        return True
    return False


# ── Meta / capability ────────────────────────────────────────────────────────

_META_RE = re.compile(
    r"\bwhat\s+can\s+you\s+do\b|\bwhat\s+do\s+you\s+do\b"
    r"|\bwhat\s+(?:can|could|should)\s+i\s+ask\b"
    r"|\bwhat\s+(?:questions?|things?)\s+can\s+i\s+ask\b"
    r"|\bhow\s+(?:can|do)\s+you\s+help\b|\bwhat\s+can\s+you\s+help\s+(?:me\s+)?with\b"
    r"|\bwhat\s+(?:drugs?|medications?|meds?|medicines?|leaflets?)\s+"
    r"(?:do\s+you\s+(?:know|have)|are\s+(?:loaded|uploaded|available|here)|"
    r"have\s+i\s+uploaded)\b"
    r"|\bwhich\s+(?:drugs?|medications?|meds?)\b[^?]{0,30}\b(?:know|have|loaded|uploaded)\b"
    r"|\blist\s+(?:the\s+|my\s+)?(?:drugs?|medications?|meds?|leaflets?)\b",
    re.IGNORECASE,
)

# ── Greeting / thanks / closing (whole-message, anchored) ─────────────────────
# Anchored so "thanks, what about pregnancy?" is NOT treated as a greeting.

_GREETING_RE = re.compile(
    r"^\W*(?:hi|hello|hey|hiya|heya|howdy|yo|sup|greetings|gm"
    r"|good\s+(?:morning|afternoon|evening|day)"
    r"|how\s+are\s+you|how'?s\s+it\s+going)"
    r"(?:\s+(?:there|lara|all|everyone|guys|folks))?[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\W*(?:thanks|thank\s+you|thank\s+u|thankyou|thx|ty|tysm|cheers"
    r"|much\s+appreciated|appreciate\s+it"
    r"|ok|okay|okey|alright|cool|nice|great|awesome|perfect|got\s+it"
    r"|good\s+to\s+know|sounds\s+good|makes\s+sense)"
    r"(?:\s+(?:you|so\s+much|very\s+much|a\s+lot|lara|then))?[\s!.,?]*$",
    re.IGNORECASE,
)
_CLOSING_RE = re.compile(
    r"^\W*(?:bye|goodbye|good\s*bye|see\s+(?:you|ya)(?:\s+later)?|take\s+care"
    r"|cya|farewell|later|good\s*night|gn|that'?s\s+(?:all|it)|that\s+is\s+all"
    r"|i'?m\s+done|we'?re\s+done)"
    r"(?:\s+(?:you|lara|then|now|for\s+now))?[\s!.,?]*$",
    re.IGNORECASE,
)

# ── Out-of-scope medical ─────────────────────────────────────────────────────

_DIAGNOSIS_RE = re.compile(
    r"\bdo\s+i\s+have\s+(?:a|an|some|any)\b"
    r"|\bdiagnos\w+\b"
    r"|\bis\s+(?:this|it|that)\s+(?:a\s+)?(?:symptom|sign|normal|serious)\b"
    r"|\bwhat'?s\s+wrong\s+with\s+me\b"
    r"|\bcould\s+(?:it|this|i)\s+(?:be|have)\b"
    r"|\bshould\s+i\s+be\s+worried\b"
    r"|\bam\s+i\s+(?:having|getting|sick)\b"
    r"|\bwhy\s+do\s+i\s+(?:have|feel|keep|get)\b"
    r"|\bis\s+something\s+wrong\b",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(
    r"\bwhat\s+(?:should|can|could)\s+i\s+take\s+for\b"
    r"|\bwhat\s+(?:do|would)\s+you\s+recommend\b"
    r"|\brecommend\s+(?:me\s+)?(?:a|an|some|something|any)\b"
    r"|\bwhich\s+(?:drug|medication|medicine|pill)\s+(?:should\s+i|is\s+best|for|would)\b"
    r"|\bbest\s+(?:drug|medicine|medication|treatment)\s+for\b"
    r"|\bwhat\s+(?:medicine|drug|medication)\s+(?:should\s+i\s+take|for|is\s+good\s+for)\b"
    r"|\bgive\s+me\s+something\s+for\b"
    r"|\bwhat\s+can\s+i\s+take\s+to\b",
    re.IGNORECASE,
)
_COST_RE = re.compile(
    r"\bcost\b|\bprice\b|\bpricing\b"
    r"|\binsurance\b|\bcopay\b|\bco-pay\b|\bdeductible\b|\bout\s+of\s+pocket\b"
    r"|\bpharmacy\s+(?:hours|open|near|location)\b"
    r"|\bwhere\s+(?:can|do)\s+i\s+(?:buy|get|fill|purchase)\b",
    re.IGNORECASE,
)

# Common OTC/Rx names + a case-insensitive pharmaceutical-suffix regex, used only
# to spot a drug the session has no leaflet for. Best-effort; misses degrade to
# the existing retrieval fallback.
_COMMON_DRUGS = {
    "ibuprofen",
    "ibuprofeno",
    "aspirin",
    "aspirina",
    "acetaminophen",
    "paracetamol",
    "tylenol",
    "advil",
    "motrin",
    "aleve",
    "naproxen",
    "amoxicillin",
    "penicillin",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "lipitor",
    "amlodipine",
    "omeprazole",
    "prilosec",
    "prednisone",
    "gabapentin",
    "losartan",
    "albuterol",
    "ventolin",
    "insulin",
    "warfarin",
    "coumadin",
    "xanax",
    "valium",
    "ativan",
    "klonopin",
    "adderall",
    "ritalin",
    "oxycodone",
    "oxycontin",
    "hydrocodone",
    "vicodin",
    "codeine",
    "morphine",
    "fentanyl",
    "tramadol",
    "viagra",
    "cialis",
    "benadryl",
    "zyrtec",
    "claritin",
    "melatonin",
}
_DRUG_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}(?:cillin|mycin|cycline|prazole|statin|sartan|pril|olol|dipine"
    r"|azepam|azolam|oxetine|caine|profen|fenac|tinib|parin|dronate|triptan|setron"
    r"|coxib|gliptin|glitazone|floxacin|conazole|barbital)\b",
    re.IGNORECASE,
)


def _detect_drug_mentions(question: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for name in _COMMON_DRUGS:
        if name not in seen and whole_word_match(name, question):
            seen.add(name)
            found.append(name)
    for m in _DRUG_SUFFIX_RE.finditer(question):
        w = m.group(0).lower()
        if w not in seen:
            seen.add(w)
            found.append(w)
    return found


def _foreign_drug(question: str, drugs_found: list[str]) -> str | None:
    """A drug named in the question that this session has no leaflet for, else None.

    Returns None when the session has no drugs, or when a session drug is also
    mentioned (let the RAG pipeline handle that case).
    """
    if not drugs_found:
        return None
    if any(whole_word_match(d, question) for d in drugs_found):
        return None
    session = {d.lower() for d in drugs_found}
    for cand in _detect_drug_mentions(question):
        if cand not in session:
            return cand
    return None


# ── Off-topic / non-medical (positive evidence required) ─────────────────────
# Kept specific so real medical questions fall through to MEDICAL; anything not
# positively recognised here defaults to the RAG pipeline.

_OFF_TOPIC_RE = re.compile(
    r"\bweather\b|\bforecast\b|\btemperature\s+(?:outside|today)\b"
    r"|\b(?:football|soccer|basketball|baseball|cricket|hockey|tennis|nba|nfl|mlb"
    r"|world\s+cup|super\s+bowl|olympics)\b"
    r"|\bwrite\s+(?:me\s+)?(?:a|an|some)\s+(?:poem|song|story|essay|haiku|joke"
    r"|limerick|rap|script|code)\b"
    r"|\btell\s+me\s+a\s+joke\b"
    r"|\b(?:python|javascript|typescript|c\+\+|sql|html|css)\b"
    r"|\b(?:write|debug|compile|refactor)\b[^?.!]{0,30}\bcode\b"
    r"|\btranslat(?:e|ion)\b"
    r"|\bcapital\s+of\b|\bwho\s+(?:won|is\s+the\s+president|painted|wrote|invented)\b"
    r"|\bhow\s+(?:tall|old|far|big)\s+is\b"
    r"|\b(?:stock\s+price|bitcoin|crypto|ethereum)\b"
    r"|\b(?:recipe|how\s+to\s+(?:cook|bake))\b"
    r"|\b(?:who\s+(?:are|made|created|built|trained|designed)\s+you"
    r"|are\s+you\s+(?:a\s+)?(?:robot|bot|human|real|an?\s+ai|chatgpt|gpt|llm))\b"
    r"|\bwhat\s+(?:ai\s+)?model\s+(?:are\s+you|do\s+you\s+use)\b"
    r"|\b(?:sue|lawsuit|lawyer|attorney|legal\s+advice)\b"
    r"|\b(?:my|our)\s+(?:dog|cat|puppy|kitten|pet|horse)\b|\bveterinar\w+\b"
    r"|\bi\s+(?:love|hate)\s+you\b"
    r"|\byou(?:'re| are)\s+(?:useless|stupid|dumb|amazing|the\s+best|awesome)\b",
    re.IGNORECASE,
)


def classify_input(question: str, drugs_found: list[str]) -> GateResult:
    """Route a chat message. See module docstring for the priority order."""
    q = (question or "").strip()
    low = q.lower()

    # 1. Degenerate input.
    if _is_degenerate(q, low):
        return GateResult(Route.OFF_TOPIC, _REPLY_DEGENERATE)

    # 2-3. Safety FIRST (multilingual), before the language check.
    if _SELF_HARM_RE.search(q) or _HARM_FACILITATION_RE.search(q):
        return GateResult(Route.SAFETY_SELF_HARM, _REPLY_CRISIS)
    if _EMERGENCY_RE.search(q):
        return GateResult(Route.SAFETY_EMERGENCY, _REPLY_EMERGENCY)

    # 4. Prompt-injection / jailbreak — before scope checks; never leak the prompt.
    if _INJECTION_RE.search(q):
        return GateResult(Route.OFF_TOPIC, _REPLY_OFF_TOPIC)

    # 5. Non-English (heuristic).
    if _looks_non_english(q):
        return GateResult(Route.NON_ENGLISH, _REPLY_NON_ENGLISH)

    # 6. Meta / capability.
    if _META_RE.search(q):
        return GateResult(Route.META, _meta_reply(drugs_found))

    # 7. Greeting / thanks / closing.
    if _GREETING_RE.search(q):
        return GateResult(Route.GREETING, _REPLY_GREETING)
    if _THANKS_RE.search(q):
        return GateResult(Route.GREETING, _REPLY_THANKS)
    if _CLOSING_RE.search(q):
        return GateResult(Route.GREETING, _REPLY_CLOSING)

    # 8. Out-of-scope medical.
    if _DIAGNOSIS_RE.search(q):
        return GateResult(Route.OUT_OF_SCOPE_MED, _REPLY_DIAGNOSIS)
    if _RECOMMENDATION_RE.search(q):
        return GateResult(Route.OUT_OF_SCOPE_MED, _REPLY_RECOMMENDATION)
    if _COST_RE.search(q):
        return GateResult(Route.OUT_OF_SCOPE_MED, _REPLY_COST)
    foreign = _foreign_drug(q, drugs_found)
    if foreign is not None:
        return GateResult(
            Route.OUT_OF_SCOPE_MED, _foreign_drug_reply(foreign, drugs_found)
        )

    # 9. Off-topic / non-medical.
    if _OFF_TOPIC_RE.search(q):
        return GateResult(Route.OFF_TOPIC, _REPLY_OFF_TOPIC)

    # 10. Default: a medical question about uploaded drugs → run the RAG pipeline.
    return GateResult(Route.MEDICAL, None)
