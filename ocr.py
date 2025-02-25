import openai
import spacy
import difflib

# Load the NLP model for spacy (will be used for POS tagging) 
# is this the best model/technique?
nlp = spacy.load("en_core_web_sm")

client = openai.OpenAI(api_key="[-YOUR-API-KEY-HERE-]")

# System message to instruct ChatGPT on transcription rules
SYSTEM_PROMPT = """You are an expert in transcribing early modern English printed texts. 
You receive raw OCR text and correct it based on human-verified transcriptions. 

Rules:
- Preserve original spelling.
- Do not translate Latin text into English.
- Expand abbreviations (e.g., 'wᵗʰ' → 'with').
- Convert 'ſ' (long s) to 's'.
- Separate ligatures into individual letters.
- Mark marginal notes as [MARGINAL NOTE: text].
- Retain formatting and line breaks where possible.

Here are some examples of corrections:"""

#few-shot training -- the more the better here
EXAMPLES = [
    {"ocr": "all things that shall be treated secretlie at y® Counsell",
     "corrected": "all things that shall be treated secretlie at the counsell"},
    
    {"ocr": "w shall be resolued vpon",
     "corrected": "which shall be resolved upon"}
]

# iterate over the few-shot data and format them as instructions (this will be sent to
# ChatGPT immediately after the SYSTEM_PROMPT text):
def format_few_shot_prompt(examples):
    """Formats the prompt using OCR and human-corrected transcriptions."""
    formatted_examples = "\n".join(
        f"Original OCR:\n{ex['ocr']}\nCorrected Transcription:\n{ex['corrected']}\n---"
        for ex in examples
    )
    return formatted_examples

# New OCR text to be corrected (hard-coded for demonstration purposes; this needs to be
# modular and accept a directory of text files as an argument
new_ocr_text = """
 Fleete of .8.
shippes

one Pinnace

600. land men

Spaine

12 RECORDS OF THE VIRGINIA COMPANY

W ByYRpDE

[Indorsed:] Dnus ffranciscus Popham miles et Dna Anna Popha é Johem
Havercomb §Browne§ Williamson.

Lecta lata et pmulgata fuit hec Snia p Dim Thoma Crompton militem

sup!me Curie Admi* Anglie Judicem xxiij Junii 1608.

V. Virernta Councinu. ‘SINSTRUCCONS ORDERS AND CONSTITUCONS
* * * vo Sk THomas GATES KNIGHT GOVERNOR OF VIRGINIA”

May, 1609

Ashmolean Manuscripts, 1147, folios 175-190a. A contemporary copy
Document in Bodleian Library, Oxford University
List of Records No. 10

Instrucéons orders and constituéons by way of advise sett downe declared
and ppounded to 8* Thomas Gates knight Governo’ of Virginia and
of the Colony there planted, and to be planted, and of all the inhabi-
tants thereof, by vs his maiesties Counsell for the Direcéon of the
affaires of that Countrey for his better disposinge and pceedinge in
the government thereof §accordinge§ to the authority and power giuen
vnto vs by by virtue of his Ma"'*: lres Patents:

1 Hauinge considered the greate sufficiency and zealous affecton w** you
S' Thomas Gates haue many waies manifested vnto vs, and hauinge
therefore by our Commission vnder o' hand¢ and seales constituted and
ordayned you to be the governor of Virginia, wee his Mate Counsell for
that plantatéon, haue consulted and advised vppon diuers instrucéons for
yo" safer and more delibate p’ceedinge, therein, And therefore doe requier
and charge §you§ accordinge to the Comission in that behalf Directed vnto
you, presently with all convenient speede to take the charge and of our
fleete Consistinge of eight good shippes and one Pinnace and of sixe
hundred land men to be transported vnder yo’ Comaund, and with the
NFO
first winde to sett sayle for virginia. And in yo" passage thither you shall
not land nor touch any of §the Kinge of§ Spaines his Dominions quetly
possessed, without the leaue or licence of the governor of such place as
you shal by accident or contrary windes, be forced into. Yo" shall also
hold Counsell with the M™ and Pilotts and men of the best experience
"""

def correct_transcription(ocr_text):
    """Sends OCR text to ChatGPT with few-shot learning to improve accuracy."""
    prompt = SYSTEM_PROMPT + "\n" + format_few_shot_prompt(EXAMPLES) + f"\nPlease correct the following:\n{ocr_text}\nCorrected Transcription:"

    response = client.chat.completions.create(
        model="gpt-4-turbo", # TODO: try other models? turbo is cheap and fast, but can more expensive, "better" models help?
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2, # i.e., be more deterministic, less "creative"; change to experiment
        stream=True # show results as they appear
    )

    # Collect the streamed tokens:
    corrected_text = ""
    print("\n[INFO] Streaming response from ChatGPT...\n")
    for chunk in response:
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            text_part = chunk.choices[0].delta.content
            if text_part:  # is null at the end of the stream
                print(text_part, end="", flush=True)  # print tokens as they arrive
                corrected_text += text_part  # and accumulate tokens into the final string

    print("\n\n[INFO] Transcription correction complete.")
    return corrected_text.strip()

# Run the correction process and show corrected text
corrected_text = correct_transcription(new_ocr_text)

def generate_diff(ocr_text, corrected_text):
    diff = difflib.ndiff(ocr_text.split(), corrected_text.split())
    highlighted_diff = []

    for word in diff:
        if word.startswith("+"):
            highlighted_diff.append(f"\033[32m{word[2:]}\033[0m")  # green = addition
        elif word.startswith("-"):
            highlighted_diff.append(f"\033[31m{word[2:]}\033[0m")  # red = deletion
        else:
            highlighted_diff.append(word[2:])  # no color for unchanged words
    return " ".join(highlighted_diff)

# show changes and run NLP
print("\n[INFO] Comparing OCR text with corrected version...")
diff_output = generate_diff(new_ocr_text, corrected_text)
print("\n[INFO] Changes Highlighted:\n", diff_output)

doc = nlp(corrected_text)
print("Named Entities, Phrases, and Concepts:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
