import os
import json
import random
import textwrap
from flask import Flask, render_template, request, redirect, url_for, session
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ==================== CONFIG ====================

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
VISITS_PER_CLIENT = 3
WRAP_WIDTH = 80
MAX_HISTORY = 6  # keep session small

# Rotate secret key each run → fresh instance every launch
SECRET_KEY = os.urandom(32)

client = InferenceClient(api_key=API_KEY)

with open("./system", "r") as f:
    SYSTEM_PROMPT = f.read()

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ==================== HELPERS ====================

def generate_name():
    first = ["John", "Maria", "Theo", "Linda", "Jack", "James", "Nick", "Marley"]
    last = ["Smith", "Lewis", "White", "Thomas", "Waters", "McCormick"]
    return f"{random.choice(first)} {random.choice(last)}"


def wrap(text):
    return "<br>".join(textwrap.wrap(text, width=WRAP_WIDTH))


def med_link(name):
    url = "https://www.drugs.com/search.php?searchterm=" + name.replace(" ", "+")
    return f'<a href="{url}" target="_blank">{name}</a>'


def call_model(messages):
    out = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024,
    )
    return out.choices[0].message.content


def parse_json_or_nil(text):
    if text == "NIL":
        return None
    try:
        return json.loads(text)
    except:
        return None


def trim_history(messages):
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    others = others[-MAX_HISTORY:]
    return system + others


def init_client():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "NEW CONVERSATION"}
    ]

    response = call_model(messages)
    info = parse_json_or_nil(response)
    if info is None:
        return None, None, None

    messages.append({"role": "assistant", "content": response})
    messages = trim_history(messages)

    name = generate_name()

    session["messages"] = messages
    session["patient_name"] = name
    session["visit"] = 1
    session["history"] = []  # store visit-by-visit results
    session["score"] = 3.0   # cumulative score start

    return name, 1, info


def update_cumulative_score(move_quality):
    score = session.get("score", 3.0)
    score = (score * 0.7) + (move_quality * 0.3)
    session["score"] = score
    return score


def is_question(text):
    return "?" in text.strip()


def classify(info, user_input):
    rt = info.get("response_type", "").lower()
    if rt == "question_response":
        return True
    if rt == "action_response":
        return False
    return is_question(user_input)


# ==================== ROUTES ====================

@app.route("/new")
def new_client():
    session.clear()
    return redirect(url_for("index"))


@app.route("/final")
def final_page():
    return render_template(
        "final.html",
        name=session.get("patient_name"),
        score=session.get("score", 0),
        visits=session.get("history", []),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    error = None

    if "messages" not in session:
        name, visit, info = init_client()
        if info is None:
            return "Error initializing client."
    else:
        name = session["patient_name"]
        visit = session["visit"]
        messages = session["messages"]
        info = None

        if request.method == "POST":
            user_input = request.form.get("action", "").strip()
            if not user_input:
                return redirect(url_for("index"))

            messages.append({"role": "user", "content": user_input})
            messages = trim_history(messages)

            response = call_model(messages)

            if response == "NIL":
                error = "Input rejected."
            else:
                new_info = parse_json_or_nil(response)
                if new_info is None:
                    error = "Invalid JSON returned."
                else:
                    messages.append({"role": "assistant", "content": response})
                    messages = trim_history(messages)
                    session["messages"] = messages
                    info = new_info

                    is_q = classify(new_info, user_input)

                    if not is_q:
                        # update cumulative score
                        update_cumulative_score(new_info["move_quality"])

                        # store visit data
                        session["history"].append({
                            "visit": visit,
                            "info": new_info
                        })

                        visit += 1
                        session["visit"] = visit

                        if visit > VISITS_PER_CLIENT:
                            return redirect(url_for("final_page"))

        if info is None:
            # load last assistant JSON
            for m in reversed(messages):
                if m["role"] == "assistant":
                    info = parse_json_or_nil(m["content"])
                    break

    # prepare display
    meds = info.get("current_medications", [])
    meds_html = ", ".join(med_link(m) for m in meds) if meds else "None"

    last_user = None
    for m in reversed(session["messages"]):
        if m["role"] == "user":
            last_user = m["content"]
            break

    is_q = classify(info, last_user)

    show_score = (not is_q) and (visit > 1)
    show_vitals = not is_q
    show_meds = not is_q

    return render_template(
        "main.html",
        name=name,
        visit=min(visit, VISITS_PER_CLIENT),
        max_visits=VISITS_PER_CLIENT,
        info=info,
        meds_html=meds_html,
        quote_html=wrap(info["patient_response"]),
        hint_html=wrap(info["hint"]),
        show_score=show_score,
        show_vitals=show_vitals,
        show_meds=show_meds,
        score=info.get("move_quality", 0),
        overview=info.get("overview", ""),
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
