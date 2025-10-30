from jinja2 import Template

def render_template(tmpl: str, context: dict) -> str:
    return Template(tmpl).render(**context)
