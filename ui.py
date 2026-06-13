import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Movie Summarizer AI",
    page_icon="🎬",
    layout="wide"
)

# =========================
# Model
# =========================

model = ChatMistralAI(
    model="mistral-small-2506"
)

# =========================
# Schema
# =========================

class Movie(BaseModel):
    title: str = Field(description="Title of the movie")

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


parser = PydanticOutputParser(
    pydantic_object=Movie
)

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
        "{paragraph}"
    )
])


# =========================
# Header
# =========================

st.title("🎬 Movie Summarizer AI")
st.markdown(
    "Convert movie descriptions into structured summaries using **LangChain + Mistral AI**"
)

st.divider()

# =========================
# Input
# =========================

movie_text = st.text_area(
    "Movie Description",
    height=300,
    placeholder="Paste a movie description here..."
)

# =========================
# Button
# =========================

if st.button(
    "Generate Summary",
    use_container_width=True,
    type="primary"
):

    if not movie_text.strip():
        st.warning("Please enter a movie description.")
        st.stop()

    with st.spinner("Analyzing movie..."):

        final_prompt = prompt.invoke({
            "paragraph": movie_text,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)

        movie = parser.parse(response.content)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎞️ Movie Information")

        st.write(f"**Title:** {movie.title}")

        if movie.release_year:
            st.write(f"**Release Year:** {movie.release_year}")

        if movie.director:
            st.write(f"**Director:** {movie.director}")

        if movie.rating:
            st.write(f"**Rating:** {movie.rating}")

        if movie.genre:
            st.write(
                f"**Genres:** {', '.join(movie.genre)}"
            )

    with col2:
        st.subheader("🎭 Cast")

        if movie.cast:
            for actor in movie.cast:
                st.write(f"• {actor}")
        else:
            st.write("No cast information available.")

    st.divider()

    st.subheader("📝 Summary")
    st.write(movie.summary)

    if movie.key_themes:
        st.subheader("🎯 Key Themes")

        for theme in movie.key_themes:
            st.write(f"• {theme}")

    if movie.awards:
        st.subheader("🏆 Awards")

        for award in movie.awards:
            st.write(f"• {award}")