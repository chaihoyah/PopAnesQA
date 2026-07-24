"""Prompts used in the three PopAnesQA construction stages."""

STAGE1_SYSTEM = (
    "You are a clinical text reviewer specializing in Anesthesiology."
)

STAGE1_USER = """### Instruction:
Analyze the following OCR-extracted guideline text. Determine whether it contains
**practical clinical directives or recommendations relevant to Anesthesiology**
for either Pediatric or Adult patients.

A text is considered RELEVANT only if it includes at least one of the following:
- perioperative assessment or optimization
- airway evaluation or airway device selection/management
- anesthetic or sedative drug use, dosing, adjustment, or monitoring
- intraoperative or postoperative management related to anesthesia care
- fluid, blood product, or anticoagulant management in the perioperative setting
- pain management where anesthesiology is responsible
- anesthetic emergency or complication recognition/management

The text is NOT relevant if it is:
- purely administrative (billing, scheduling, consent wording, references, citations)
- general pathophysiology without anesthetic management implications
- nursing workflow not specific to anesthesia practice
- purely equipment vendor description
- background educational material without clinical action

If relevant, classify the text into ONE best-fit category:

### Categories
{{
  "Pre-op Assessment": "NPO guidelines, anxiety, premedication, cardiac/pulmonary risk evaluation",
  "Pharmacology & Fluid": "Drug dosing, FLUID therapy, transfusion, anticoagulants/DOACs",
  "Airway & Equipment": "Tube size/depth, airway approach, difficult airway plans, ventilator settings",
  "Crisis & Complication": "Laryngospasm, bronchospasm, LAST, malignant hyperthermia, arrest, hypotension",
  "Post-op & Pain": "Pain control, PONV, emergence delirium/agitation, PACU discharge"
}}

### Target Population Classification

Classify the target population as one of:
"Pediatric", "Adult", or "General".

Use the following principles:

1) Pediatric
- Choose "Pediatric" if the text explicitly refers to neonates, infants, children, or adolescents,
  OR if dosing, device sizes, thresholds, or workflows clearly imply pediatric practice
  (e.g., weight-based dosing, age-specific tube sizes, neonatal physiology).

2) Adult
- Choose "Adult" if the text explicitly refers to adults, elderly, geriatric patients,
  OR if the clinical context is clearly incompatible with routine pediatric practice,
  even when the word "adult" is not explicitly used.

  This includes situations where:
  - The diseases, indications, or surgical contexts are overwhelmingly adult-specific.
  - The management strategies are derived from adult-only or predominantly adult populations.
  - Applying the directive directly to children would be clinically unusual, unsafe,
    or outside standard pediatric practice.

  Examples of such adult-oriented contexts include (for illustration only):
  - Conditions predominantly seen in adults (e.g., atrial fibrillation, acute coronary syndrome).
  - Surgeries rarely encountered in pediatrics (e.g., degenerative joint replacement, most acquired cardiac surgery).
  - Age thresholds or terms such as "elderly", "geriatric", or ≥65–75 years.
  - Fixed-dose regimens derived from adult clinical trials without pediatric adaptation.

3) General
- Choose "General" ONLY if ALL of the following are true:
  - No age group is explicitly mentioned, AND
  - There is no strong adult-specific or pediatric-specific clinical context, AND
  - The directive represents a general anesthesia principle or workflow
    that would reasonably and safely apply to both pediatric and adult patients.

  Examples of truly "General" content include:
  - Basic perioperative monitoring principles.
  - Standard airway preparation or safety checks without age-specific features.
  - High-level preoperative assessment frameworks not tied to specific diseases or age groups.

When in doubt:
- Ask: "Could this directive be naturally and safely applied to both children and adults
  without substantial modification?"
  - If yes → General
  - If no, and it is not pediatric → Adult

Ignore obvious OCR noise (broken words, headers, page numbers).

### Output JSON format
{{
  "is_relevant": boolean,
  "target_population": "Pediatric" | "Adult" | "General",
  "category": "Selected Category or null",
  "reason": "Short concise justification"
}}

### Text to analyze:
{input_text}
"""


STAGE2_SYSTEM = """You are a Medical Data Auditor specializing in Anesthesiology.
Your role is to extract ONLY explicit, verifiable clinical logic from text.
Do NOT infer or invent missing information."""

STAGE2_USER = """### Instruction:
Based on the previously-assigned classification:

- target_population = "{target_population}"
- category = "{category}"

Extract ONLY clinical logic that is explicit, actionable, and rule-based.

A statement qualifies as a clinical rule ONLY if it clearly changes
how a clinician should prescribe, dose, time, monitor, or perform a
clinical procedure.

---

### VERY IMPORTANT STRICT RULES

1. NO EXTERNAL KNOWLEDGE OR INFERENCE
   - Extract ONLY what is explicitly stated in the text. Statements using words such as "should", "must", "do not", "is recommended" count as explicit clinical actions if they clearly change management.
   - If the text does not clearly specify a clinical instruction or decision rule, return an empty rules list.
   - Do NOT extract rules whose action is only a caution-level statement (e.g., "use caution", "monitor closely", "be aware"), even if these phrases explicitly appear in the text,
     unless the text also specifies a concrete management change (e.g., dose, timing, hold/restart interval, monitoring frequency, procedural step).

2. ALLOWED CONTENT
   A statement qualifies as a clinical rule if it clearly defines
   a decision or instruction that changes clinical management.

   This typically includes, but is NOT limited to:
   - when to start, stop, hold, or restart a drug
   - how much drug to give
   - how timing or dose changes based on a condition
   - measurable thresholds that trigger an action
   - procedural or crisis response steps
   - airway or device-related settings or selection rules
   - postoperative treatment or monitoring instructions

3. EXCLUDED CONTENT
   Do NOT extract:
   - descriptive background information
   - statistical risk associations
   - mechanistic / pharmacokinetic explanations
   - general statements about risk or safety
   - statements that lack a clear clinical action
   - ambiguous or incomplete numeric content

4. CATEGORY-BOUND EXTRACTION
   - Extract ONLY rules that belong to the assigned category and target population. Ignore unrelated numeric or descriptive content.
   - If a rule spans multiple categories, extract it only if its primary action clearly belongs to the assigned category.

5. DIFFERENT VARIABLES MUST NOT BE MERGED
   Treat timing, restart intervals, half-life, washout, maintenance dose,
   bolus dose, etc. as separate concepts when explicitly stated.

6. RULE GROUPING
   If multiple conditions all lead to the SAME action,
   represent them as ONE rule object with multiple triggers.
   (Triggers are OR-conditions unless the text explicitly states otherwise.)

7. NO APPROXIMATION
   - Preserve inequality symbols and numeric precision.
   - Do NOT convert units or restate numbers.

8. OCR NOISE RULE
   If a value or condition is ambiguous, corrupted, or incomplete, EXCLUDE it.

9. CONDITION_BASE ANCHORING RULE
   - Anchor the condition to the exact drug, device, procedure, or population explicitly linked to the action.
   - Whenever the text implies a patient context (e.g., "patients receiving X", "use of X", "treatment with X", "undergoing procedure Y"), express `condition_base` in a patient-centered form, such as:
     - "patients receiving X"
     - "patients treated with X"
     - "patients using device X"
     - "patients undergoing procedure Y"
   - Do NOT invent a patient context if the text does not explicitly or implicitly
     indicate one.
   - Do NOT generalize to a broader class. if the rule applies only to a specific target.
   - Match the scope of `condition_base` to the narrowest entity for which the instruction is explicitly stated.

---

### Output JSON format
Return JSON ONLY:

- Each rule object must include:
  - who the rule applies to
  - the explicit clinical action
  - a list of one or more triggering conditions
  - any numeric thresholds exactly as written
  - a short quote from the text supporting the rule
  - strictly text-based clarifying notes

{{
  "rules": [
    {{
      "condition_base": "...",
      "action": "...",
      "triggers": [
        {{ "criterion": "..." }}
      ],
      "source_fragment": "...",
      "notes": "text-based clarification only"
    }}
  ]
}}

Do NOT include invented rules.
Do NOT return commentary outside the JSON.

---

### Now extract rules strictly for {target_population} in this category: {category}

SOURCE TEXT:
{input_text}
"""


STAGE3_SYSTEM = """You are a Board Exam Question Creator specializing in Anesthesiology.
Your job is to:
- Create realistic, high-quality multiple-choice clinical vignettes.
- Strictly follow the provided guideline logic.
- Never invent medical rules, numbers, or thresholds.
- Never contradict the provided source text.
- Prioritize clinical realism and educational value.

You must:
- Treat all provided rules as authoritative.
- Use only the given rules and source text.
- Avoid adding outside knowledge.
- Ensure the correct answer is unambiguously supported by the rules and source text.

Multi-rule policy (general, non-case-specific):
- If multiple rules are provided, combine them into ONE MCQ only when they can naturally occur in a single realistic scenario WITHOUT adding assumptions.
- Do NOT force unrealistic co-occurrence. If rules seem incompatible, mutually exclusive, or would require invented conditions, do NOT merge; instead build a single-rule MCQ using the most clearly supported rule.
- When merging is feasible, design the item to require rule-interaction reasoning (prioritization, exceptions, conflicts, completeness), not mere keyword matching.
- Strict grounding: never invent facts/thresholds/management steps beyond the provided rules/source text.
"""

STAGE3_USER = """### Task Context:
You must create an anesthesiology board-style MCQ that applies ONLY to:
- Target population: {target_population}
- Topic Category: {category}

Do NOT use rules from other populations or categories.

---

### Guideline Logic (Authoritative):
This is the extracted clinical logic you MUST follow:
{rules_json}

Each rule contains:
The payload contains either a single rule or a list of rules:
- If "rules" is a list, each rule contains:
- condition_base: who/what the rule applies to
- action: what must be done
- triggers: a list of one or more triggering conditions
- source_fragment: exact supporting text

- You may also see:
  - merge_preference: guidance on whether to merge rules
  - merge_goal: what "good merging" means

You may ONLY use these rules to build the case.

---

### Source Text (For Validation)
Use this to verify wording and numbers:
{input_text}

If there is any mismatch between rules and text, always trust the SOURCE TEXT.

---

### Instruction:
You are a Board Exam Question Creator for {target_population} Anesthesiology.
Based on the provided Guideline Logic and Source Text, create ONE high-quality MCQ.

Requirements:

0. Multi-rule decision (only if multiple rules are provided)
   - FIRST decide if you can realistically combine two or more rules into a single vignette using ONLY the given triggers/conditions and source text.
   - If YES: create a System-2-like MCQ that requires applying at least TWO rules together via rule-interaction reasoning:
       - prioritization (which action matters most when multiple triggers apply)
       - exception handling (when one rule modifies/limits another)
       - conflict resolution (avoid choices that satisfy one rule but violate another)
       - completeness (best option addresses all applicable triggers; distractors miss one key trigger)
   - If NO: do NOT force merging. Create a single-rule MCQ using the rule with the clearest triggers and strongest direct support in the source text.

1. Patient Profile
   Create a realistic {target_population} patient.
   - If Pediatric: specify age (e.g., 4yo), weight (e.g., 16kg).
   - If Adult: specify age (e.g., 65yo), comorbidities, weight if relevant.

2. Scenario
   - Must strictly follow the extracted rules.
   - Must clearly activate at least one rule via its triggers.
   - Do not add extra conditions not present in the rules.
   - If merging rules: ensure the scenario naturally activates multiple rules without introducing assumptions beyond the given rules/source text.

3. Distractors
   - Create three plausible but incorrect options.
   - Try to vary error types when possible:
     - Wrong time, dose, or threshold
     - Applying a rule from a different population or context
     - Outdated or superseded practice
     - Overly aggressive or unsafe action
     - Ignoring a key condition or trigger

   - Do NOT force rigid categories.
   - Choose the most realistic and educational errors.

   - When natural:
     - Include at least one common clinical mistake.
     - Include at least one unsafe or contraindicated option.

   - If artificial, prioritize realism.

---

### Wording constraint (MUST):
- Do NOT use phrases like "According to the guideline(s)", "per the guideline", "based on the guideline", "current guidelines recommend", or similar wording in the question stem, options, or scenario.
- Write the MCQ as a self-contained board-style question that can be answered using only the information implicitly contained in the vignette (even though it is derived from the provided rules/source text).

### Validation Step:
Before finalizing the correct answer:
- Double-check numbers against SOURCE TEXT.
- If rule says "24 hours" but text says "2 days", use "2 days".

### Grounding constraint for Explanation (MUST):
- In the explanation, do NOT assert any medical facts, contraindications, preferences, or rationales that are NOT explicitly supported by the provided rules_json or source_fragment/source text.
- If a distractor is incorrect because it is outside the provided rules, say so explicitly (e.g., "This is not supported by the provided source text/rules") rather than introducing outside clinical knowledge.

---

### Output JSON format (Return JSON ONLY):

{{
  "question_type": "Management" | "Calculation" | "Diagnosis",
  "target_population": "{target_population}",
  "category": "{category}",
  "patient_profile": "...",
  "scenario": "...",
  "question": "...",
  "options": {{
    "a": "...",
    "b": "...",
    "c": "...",
    "d": "..."
  }},
  "correct_answer": "a" | "b" | "c" | "d",
  "explanation": "Explain using the guideline logic and source text.",
  "difficulty": "Hard" | "Medium" | "Easy"
}}
"""
