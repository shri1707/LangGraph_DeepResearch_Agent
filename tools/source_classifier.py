from urllib.parse import urlparse

def classify_source_type(url: str) -> str:
    domain = urlparse(url).netloc.lower()

    if "reddit.com" in domain or "quora.com" in domain or "stackexchange.com" in domain:
        return "forum"

    if "medium.com" in domain or "substack.com" in domain:
        return "independent_blog"

    if domain.endswith(".gov") or domain.endswith(".edu"):
        return "official"

    if any(x in domain for x in [
        "blog", "wikipedia", "substack", "wordpress"
    ]):
        return "independent_blog"

    return "vendor_blog"
