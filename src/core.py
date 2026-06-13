from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
from typing import List,Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser

model = ChatMistralAI(
    model="mistral-small-2506",
)

class Movie(BaseModel):
    title: str = Field(
        description="Title of the movie"
    )

    release_year: Optional[int] = Field(
        default=None,
        description="Year the movie was released"
    )

    genre: List[str] = Field(
        default_factory=list,
        description="Movie genres"
    )

    director: Optional[str] = Field(
        default=None,
        description="Director of the movie"
    )

    cast: List[str] = Field(
        default_factory=list,
        description="Main cast members"
    )

    rating: Optional[str] = Field(
        default=None,
        description="Movie rating if mentioned"
    )

    awards: List[str] = Field(
        default_factory=list,
        description="Awards won by the movie"
    )

    summary: str = Field(
        description="Concise summary of the movie"
    )

    key_themes: List[str] = Field(
        default_factory=list,
        description="Major themes explored in the movie"
    )

parser = PydanticOutputParser(pydantic_object=Movie)


# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are an expert content summarization assistant.

# Your task is to summarize movies, articles, blogs, and other content in a clear and structured format.

# For movies:
# - Mention the movie title if identifiable.
# - Mention release year, genre, director, cast, rating, and awards only if they are explicitly provided in the input.
# - Do not invent or guess information.
# - Write a concise plot summary.
# - Mention the major themes and takeaways.

# For articles and blogs:
# - Identify the main topic.
# - Summarize the key points and conclusions.
# - Highlight important insights.

# Rules:
# - Keep the summary concise but informative.
# - Use professional and easy-to-read language.
# - Preserve the original meaning.
# - Remove repetition and unnecessary details.
# - If some information is unavailable, simply omit it instead of writing placeholders.

# Format movie summaries like:

# Movie: <title>

# Release Year: <year>

# Genre: <genre>

# Director: <director>

# Cast: <cast>

# Summary:
# <summary>

# Key Themes:
# - Theme 1
# - Theme 2

# Awards:
# <awards if available>

# Format article/blog summaries like:

# Title: <title>

# Summary:
# <summary>

# Key Takeaways:
# - Point 1
# - Point 2
# - Point 3
#         """
#     ),
#     (
#         "human",
#         """
# Extract information from this paragraph:

# {paragraph}
#         """
#     )
# ]) # this prompt is runnable

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the provided text.
Populate the movie schema accurately.
Only include information explicitly mentioned in the text.
Do not guess or invent missing details.
    {format_instructions}
        """
    ),
    (
        "human",
        """
{paragraph}
        """
    )
])

para = input("Tell me about a movie, and I'll create a concise summary: ")

final_prompt = prompt.invoke(
    {"paragraph": para,
     "format_instructions": parser.get_format_instructions()
     }
)

response = model.invoke(final_prompt)
print(response.content)