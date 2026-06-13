from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

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

Format movie summaries like:

Movie: <title>

Release Year: <year>

Genre: <genre>

Director: <director>

Cast: <cast>

Summary:
<summary>

Key Themes:
- Theme 1
- Theme 2

Awards:
<awards if available>

Format article/blog summaries like:

Title: <title>

Summary:
<summary>

Key Takeaways:
- Point 1
- Point 2
- Point 3
        """
    ),
    (
        "human",
        """
Extract information from this paragraph:

{paragraph}
        """
    )
]) # this prompt is runnable

para = input("Tell me about a movie, and I'll create a concise summary: ")

final_prompt = prompt.invoke(
    {"paragraph": para}
)

response = model.invoke(final_prompt)
print(response.content)