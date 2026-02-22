from typing import Any, Dict

from jinja2 import Environment


class TemplateRenderer:
    def __init__(self):
        self.env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,  # nosec B701 - renders LLM prompts, not HTML
        )

    def render_request(self, template: str, variables: Dict[str, Any]) -> str:
        template = self.env.from_string(template)
        return template.render(**variables)
