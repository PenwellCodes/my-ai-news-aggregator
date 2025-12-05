import json
import os
from typing import Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class DigestOutput(BaseModel):
    title: str
    summary: str

PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence.

Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words) that captures the essence of the content
- Write a 2-3 sentence summary that highlights the main points and why they matter
- Focus on actionable insights and implications
- Use clear, accessible language while maintaining technical accuracy
- Avoid marketing fluff - focus on substance"""


class DigestAgent:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.system_prompt = PROMPT

    @staticmethod
    def _parse_json_payload(raw_text: str) -> dict:
        """
        Gemini sometimes wraps JSON in markdown fences; strip and parse safely.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = "\n".join(
                line for line in cleaned.splitlines() if not line.lower().startswith("json")
            )
        return json.loads(cleaned)

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        try:
            user_prompt = (
                f"Create a digest for this {article_type}: \n"
                f"Title: {title} \n"
                f"Content: {content[:8000]} \n\n"
                'Return JSON exactly in this shape: {"title": "...", "summary": "..."}'
            )

            response = self.model.generate_content(
                [
                    {"role": "system", "text": self.system_prompt},
                    {"role": "user", "text": user_prompt}
                ],
                generation_config=GenerationConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            parsed = self._parse_json_payload(response.text)
            return DigestOutput.model_validate(parsed)
        except Exception as e:
            print(f"Error generating digest: {e}")
            return None
