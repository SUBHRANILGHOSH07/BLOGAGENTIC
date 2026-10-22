from src.states.blogstate import BlogState
from langchain_core.messages import HumanMessage
from src.states.blogstate import Blog

class BlogNode:
    """
    A class to represent the blog node.
    """

    def __init__(self,llm):
        self.llm = llm

    def title_creation (self,state:BlogState):
        """
        Create a title for the blog.
        """

        if "topic" in state and state["topic"]:

            prompt = """
                     You are an expert blog content writer. Use Markdown formatting. Generate a blog title for the {topic}.This title should be creative and SEO friendly.
                     """

            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)
            return {"blog":{"title":response.content}}

    def content_generation(self,state:BlogState):
        """
        Generate content for the blog.
        """

        if "topic" in state and state["topic"]:
            prompt = """
                     You are an expert blog content writer. Use Markdown formatting. Generate a  detailed blog content with the detailed breakdown for the topic {topic}. 
                     """

            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)
            return {"blog":{"title":state["blog"]["title"],"content":response.content}}

    def translation(self,state:BlogState):
        """
        Translate the blog content.
        """

        translation_prompts="""
        Translate the following blog title and content into {current_language} language.
        - Maintain the original tone, style and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.
        - Translate the entire blog without summarizing or omitting sections.
        - Return a JSON object with the translated "title" and "content".

        ORIGINAL TITLE:
        {blog_title}

        ORIGINAL CONTENT:
        {blog_content}
        """

        blog_content=state["blog"]["content"]
        message = [
            HumanMessage(translation_prompts.format(current_language=state["current_language"],blog_title=state["blog"]["title"],blog_content=blog_content))
        ]
        # Use Groq's native JSON schema output instead of forcing a tool call.
        # Disable streaming explicitly, including when invoked from Studio.
        result = self.llm.with_structured_output(
            Blog, method="json_schema", strict=True, include_raw=True,
            max_tokens=16384, stream=False,
        ).invoke(message)
        if result["raw"].response_metadata.get("finish_reason") == "length":
            raise ValueError("Translation reached the output token limit; shorten the blog before retrying")
        if result["parsing_error"] is not None:
            raise result["parsing_error"]
        translation_content = result["parsed"]
        return {"blog": translation_content.model_dump()}

    def route(self,state:BlogState):
        """
        Route the blog content based on the language.
        """

        language = (state.get("current_language") or "english").strip().lower()
        if language not in {"english", "hindi", "french"}:
            raise ValueError("current_language must be english, hindi, or french")
        return {"current_language": language}

    def route_decision(self,state:BlogState):
        """
        Decide the route based on the language.
        """

        return state.get("current_language", "english")
