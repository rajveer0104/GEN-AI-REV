from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(model_name="openai/gpt-oss-120b")
prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert movie and entertainment information extractor.

        Your task is to analyze the given paragraph and extract all useful
        information about the movie, series, or show mentioned in it.

        Extract the following information whenever it is available:

        - Title
        - Release Date / Year
        - Genre
        - Director
        - Writers
        - Cast / Main Actors
        - Characters played by the main actors
        - Producers
        - Production Company
        - Country / Language
        - Runtime
        - IMDb Rating or other ratings
        - Reviews / Critical Reception
        - Audience Reception
        - Awards / Nominations
        - Budget
        - Box Office
        - Main Plot / Story
        - Themes
        - Any other important information

        If a piece of information is not present in the paragraph,
        write "Not mentioned" instead of making up information.

        Finally, provide a concise summary of the movie/show

        Return the response in the following format:

        Title:
        Release:
        Genre:
        Director:
        Writers:
        Cast:
        Producers:
        Production Company:
        Country/Language:
        Runtime:
        Ratings:
        Reviews:
        Audience Reception:
        Awards:
        Budget:
        Box Office:
        Plot:
        Themes:
        Other Important Information:

        Summary:
        <concise summary>

        Do not add information that is not present in the input.
        """
    ),
    (
        "human",
        """
        Analyze the following paragraph:

        {paragraph}
        """
    )
])
inp=input("Enter your paragraph:")
final_prompt=prompt.invoke({"paragraph":inp})
response=model.invoke(final_prompt)
print(response.content)