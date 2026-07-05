from groq import Groq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_GROQ_AVAILABLE = bool(GROQ_API_KEY)

if _GROQ_AVAILABLE:
    client = Groq(api_key=GROQ_API_KEY, timeout=15)
else:
    client = None
    print("* WARNING: GROQ_API_KEY not set — skipping AI explanations")

def translate_clause(clause_text, clause_type, risk_level):
    if not _GROQ_AVAILABLE:
        return f"This is a {clause_type.replace('_', ' ')} clause ({risk_level} risk). Review it carefully in context of the full agreement."

    prompt = f"""You are a professional legal analyst. Explain this clause in clear, professional language.

    Clause text:
    ---
    {clause_text}
    ---

    Clause Type: {clause_type} | Risk Level: {risk_level}

    Provide:
    1. A concise, professional explanation of what this clause means in plain English.
    2. What the party is agreeing to or giving up.
    3. A practical note or recommendation.

    Keep it under 70 words. Use professional, neutral language. No emojis."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "You are a professional legal analyst who explains contract clauses in clear, professional plain English. Do not use emojis or casual language."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def translate_all(clauses):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def translate_one(clause):
        try:
            explanation = translate_clause(
                clause.get('full_text', ''),
                clause.get('type', 'general'),
                clause.get('risk_level', 'low')
            )
            try:
                from rag_healer import is_bad_explanation, heal_explanation
                if is_bad_explanation(explanation):
                    explanation = heal_explanation(
                        clause.get('full_text', ''),
                        clause.get('type', 'general'),
                        clause.get('risk_level', 'low'),
                        explanation
                    )
            except Exception:
                pass
        except Exception as e:
            explanation = clause.get('plain_english', 'Could not generate explanation.')
        return {**clause, 'ai_explanation': explanation}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(translate_one, c): i for i, c in enumerate(clauses)}
        ordered = [None] * len(clauses)
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()
    return ordered
