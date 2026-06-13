import customtkinter as ctk
from tkinter import END

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# =========================
# Model
# =========================

model = ChatMistralAI(
    model="mistral-small-2506",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert content summarization assistant.

Your task is to summarize movies, articles, blogs, and other content in a clear and structured format.

For movies:
- Mention the movie title if identifiable.
- Mention release year, genre, director, cast, rating, and awards only if they are explicitly provided in the input.
- Do not invent or guess information.
- Write a concise plot summary.
- Mention the major themes and takeaways.

For articles and blogs:
- Identify the main topic.
- Summarize the key points and conclusions.
- Highlight important insights.

Rules:
- Keep the summary concise but informative.
- Use professional and easy-to-read language.
- Preserve the original meaning.
- Remove repetition and unnecessary details.
- If some information is unavailable, simply omit it instead of writing placeholders.
        """
    ),
    (
        "human",
        """
Extract information from this paragraph:

{paragraph}
        """
    )
])


# =========================
# Logic
# =========================

def summarize():
    text = input_box.get("1.0", END).strip()

    if not text:
        return

    result_box.delete("1.0", END)
    result_box.insert("1.0", "Generating summary...")

    app.update()

    try:
        final_prompt = prompt.invoke({
            "paragraph": text
        })

        response = model.invoke(final_prompt)

        result_box.delete("1.0", END)
        result_box.insert("1.0", response.content)

    except Exception as e:
        result_box.delete("1.0", END)
        result_box.insert("1.0", f"Error:\n\n{str(e)}")


# =========================
# UI Setup
# =========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Summarizer AI")
app.geometry("1400x850")
app.minsize(1200, 700)

# =========================
# Header
# =========================

header = ctk.CTkFrame(
    app,
    corner_radius=0,
    height=90
)
header.pack(fill="x")

title = ctk.CTkLabel(
    header,
    text="🎬 Summarizer AI",
    font=("Segoe UI", 32, "bold")
)
title.pack(pady=(15, 0))

subtitle = ctk.CTkLabel(
    header,
    text="Summarize Movies, Articles & Blogs",
    font=("Segoe UI", 14)
)
subtitle.pack(pady=(0, 10))


# =========================
# Generate Button
# =========================

button_frame = ctk.CTkFrame(
    app,
    fg_color="transparent"
)
button_frame.pack(fill="x", pady=10)

summarize_btn = ctk.CTkButton(
    button_frame,
    text="Generate Summary",
    command=summarize,
    width=220,
    height=45,
    font=("Segoe UI", 15, "bold")
)
summarize_btn.pack()


# =========================
# Main Layout
# =========================

main_frame = ctk.CTkFrame(
    app,
    corner_radius=15
)
main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=(0, 20)
)

# Configure grid
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_rowconfigure(0, weight=1)


# =========================
# Left Panel
# =========================

left_frame = ctk.CTkFrame(
    main_frame,
    corner_radius=15
)
left_frame.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(15, 8),
    pady=15
)

input_label = ctk.CTkLabel(
    left_frame,
    text="Input Content",
    font=("Segoe UI", 20, "bold")
)
input_label.pack(anchor="w", padx=15, pady=(15, 10))

input_box = ctk.CTkTextbox(
    left_frame,
    font=("Segoe UI", 15),
    corner_radius=12
)
input_box.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)


# =========================
# Right Panel
# =========================

right_frame = ctk.CTkFrame(
    main_frame,
    corner_radius=15
)
right_frame.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(8, 15),
    pady=15
)

output_label = ctk.CTkLabel(
    right_frame,
    text="Generated Summary",
    font=("Segoe UI", 20, "bold")
)
output_label.pack(anchor="w", padx=15, pady=(15, 10))

result_box = ctk.CTkTextbox(
    right_frame,
    font=("Segoe UI", 15),
    corner_radius=12
)
result_box.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)

app.mainloop()