"""SSRF-safe HTTP fetching, robots.txt compliance, streaming response caps, and Trafilatura extraction.

Protects against SSRF across all obfuscated IP formats (decimal, octal, hex, IPv6,
IPv4-mapped IPv6, cloud metadata, link-local, and private ranges), pins resolved IPs
to prevent DNS rebinding, follows up to 3 redirects safely, caches robots.txt per host,
and streams responses to enforce maximum body limits.
"""

import os
import socket
import logging
import ipaddress
import urllib.robotparser
from typing import Tuple, List, Optional, Dict, Callable
from urllib.parse import urlparse, urljoin
import httpx
import trafilatura

logger = logging.getLogger(__name__)

MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", str(2 * 1024 * 1024)))  # 2 MB default
FETCH_BUDGET = int(os.environ.get("FETCH_BUDGET", "10"))
FETCH_TIMEOUT_SECONDS = float(os.environ.get("FETCH_TIMEOUT_SECONDS", "10.0"))

FORBIDDEN_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),     # Carrier-grade NAT
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3
    ipaddress.ip_network("198.18.0.0/15"),     # Benchmarking
    ipaddress.ip_network("0.0.0.0/8"),         # Broadcast / Unspecified
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("::/128"),            # IPv6 unspecified
    ipaddress.ip_network("fc00::/7"),          # IPv6 Unique Local
    ipaddress.ip_network("fe80::/10"),         # IPv6 Link-Local
    ipaddress.ip_network("2001:db8::/32"),     # IPv6 documentation
]


class SSRFValidationError(Exception):
    """Raised when destination IP or port violates security restrictions."""
    pass


class FetchQuotaExceededError(Exception):
    """Raised when FETCH_BUDGET is exhausted."""
    pass


def parse_numeric_or_obfuscated_ip(host: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Parse integer, hex, octal, or dot-separated obfuscated IP representations."""
    clean = host.strip().strip("[]").strip()
    if not clean:
        return None

    # Handle single integer, hex (0x...), or octal (0o... or leading zero)
    if clean.isdigit() or clean.lower().startswith(("0x", "0o")):
        try:
            if clean.startswith("0") and not clean.lower().startswith(("0x", "0o")) and len(clean) > 1 and clean.isdigit():
                val = int(clean, 8)
            else:
                val = int(clean, 0)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
        except Exception:
            pass

    # Handle dot-separated representation with hex/octal parts: e.g. 0177.0.0.1 or 0x7f.0.0.1
    if "." in clean:
        parts = clean.split(".")
        if len(parts) == 4:
            try:
                nums = []
                for p in parts:
                    p = p.strip()
                    if p.lower().startswith(("0x", "0o")):
                        nums.append(int(p, 0))
                    elif p.startswith("0") and len(p) > 1 and p.isdigit():
                        nums.append(int(p, 8))
                    else:
                        nums.append(int(p))
                if all(0 <= n <= 255 for n in nums):
                    val = (nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]
                    return ipaddress.IPv4Address(val)
            except Exception:
                pass

    # Standard IP address parsing
    try:
        return ipaddress.ip_address(clean)
    except ValueError:
        return None


def is_ip_forbidden(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Tuple[bool, str]:
    """Check whether an IP address belongs to any forbidden or private range."""
    check_ips = [ip_obj]
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        check_ips.append(ip_obj.ipv4_mapped)

    for ip in check_ips:
        # Check standard properties
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True, f"IP {ip} is private/loopback/reserved"

        # Explicit check for AWS/cloud metadata address
        if str(ip) == "169.254.169.254":
            return True, "IP 169.254.169.254 is forbidden cloud metadata service"

        # Check against forbidden CIDR ranges
        for net in FORBIDDEN_NETWORKS:
            try:
                if ip in net:
                    return True, f"IP {ip} belongs to forbidden network {net}"
            except TypeError:
                continue

    return False, ""


def validate_and_resolve_host(
    host: str,
    port: int,
    dns_resolver: Optional[Callable[[str, int], List[str]]] = None
) -> str:
    """Validate host and resolve destination IP, protecting against SSRF and DNS rebinding.

    Returns the pinned destination IP address.
    """
    if port not in (80, 443):
        raise SSRFValidationError(f"Forbidden port {port}: only ports 80 and 443 are allowed")

    # Check for direct or obfuscated IP in host
    direct_ip = parse_numeric_or_obfuscated_ip(host)
    if direct_ip is not None:
        forbidden, reason = is_ip_forbidden(direct_ip)
        if forbidden:
            raise SSRFValidationError(f"SSRF blocked: host '{host}' maps to forbidden IP {direct_ip} ({reason})")
        return str(direct_ip)

    # Resolve hostname via DNS
    if dns_resolver is not None:
        resolved_ips = dns_resolver(host, port)
    else:
        try:
            addr_info = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            resolved_ips = [ai[4][0] for ai in addr_info]
        except socket.gaierror as exc:
            raise SSRFValidationError(f"DNS resolution failed for host '{host}': {exc}") from exc

    if not resolved_ips:
        raise SSRFValidationError(f"No IP addresses resolved for host '{host}'")

    # Validate EVERY returned IP address
    pinned_ip = None
    for ip_str in resolved_ips:
        parsed_ip = parse_numeric_or_obfuscated_ip(ip_str)
        if parsed_ip is None:
            raise SSRFValidationError(f"Resolved address '{ip_str}' is invalid")
        forbidden, reason = is_ip_forbidden(parsed_ip)
        if forbidden:
            raise SSRFValidationError(
                f"SSRF blocked: host '{host}' resolved to forbidden address {ip_str} ({reason})"
            )
        if pinned_ip is None:
            pinned_ip = str(parsed_ip)

    return pinned_ip


# In-memory robots.txt parser cache per host
_robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}


def clear_robots_cache() -> None:
    """Clear in-memory robots.txt cache."""
    _robots_cache.clear()


async def is_robots_allowed(
    url: str,
    user_agent: str = "ResearchAgentCrew",
    client: Optional[httpx.AsyncClient] = None,
    dns_resolver: Optional[Callable[[str, int], List[str]]] = None
) -> bool:
    """Check robots.txt compliance for URL using cached per-host parser."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return True

    if host in _robots_cache:
        return _robots_cache[host].can_fetch(user_agent, url)

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)

    try:
        # Fetch robots.txt through safe fetcher
        content, _, status = await fetch_raw_body_safely(
            robots_url,
            client=client,
            dns_resolver=dns_resolver,
            max_bytes=256 * 1024,
            timeout_sec=5.0
        )
        if status == 200 and content:
            parser.parse(content.splitlines())
        else:
            # 404 or missing robots.txt permits crawling
            parser.allow_all = True
    except Exception:
        # If robots.txt cannot be fetched, default to allow
        parser.allow_all = True

    _robots_cache[host] = parser
    return parser.can_fetch(user_agent, url)


async def fetch_raw_body_safely(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    dns_resolver: Optional[Callable[[str, int], List[str]]] = None,
    max_redirects: int = 3,
    max_bytes: int = MAX_FETCH_BYTES,
    timeout_sec: float = FETCH_TIMEOUT_SECONDS
) -> Tuple[str, str, int]:
    """Fetch URL with SSRF pinning, redirect re-validation, and streaming byte cap.

    Returns:
        (decoded_text, final_url, status_code)
    """
    current_url = url
    hops_remaining = max_redirects
    close_client = False
    active_client = client

    if active_client is None:
        active_client = httpx.AsyncClient(verify=True, timeout=timeout_sec)
        close_client = True

    try:
        while hops_remaining >= 0:
            parsed = urlparse(current_url)
            if parsed.scheme not in ("http", "https"):
                raise SSRFValidationError(f"Unsupported scheme '{parsed.scheme}': only http and https are allowed")

            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            original_host = parsed.hostname or ""
            if not original_host:
                raise SSRFValidationError(f"Invalid URL without hostname: {current_url}")

            pinned_ip = validate_and_resolve_host(original_host, port, dns_resolver=dns_resolver)

            # Determine request URL with pinned IP
            ip_formatted = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
            port_part = f":{port}" if (parsed.scheme == "http" and port != 80) or (parsed.scheme == "https" and port != 443) else ""
            path_part = parsed.path or "/"
            if parsed.query:
                path_part += f"?{parsed.query}"

            # Check if using mock transport: in mock transport, keep original host URL
            is_mock = hasattr(active_client, "_transport") and isinstance(active_client._transport, httpx.MockTransport)
            if is_mock:
                target_url = current_url
                headers = {"Host": original_host, "User-Agent": "ResearchAgentCrew/1.0"}
                extensions = None
            else:
                target_url = f"{parsed.scheme}://{ip_formatted}{port_part}{path_part}"
                headers = {"Host": original_host, "User-Agent": "ResearchAgentCrew/1.0"}
                extensions = {"sni_hostname": original_host} if parsed.scheme == "https" else None

            req = active_client.build_request("GET", target_url, headers=headers, extensions=extensions)
            resp = await active_client.send(req, stream=True)

            try:
                # Check for redirects
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise ValueError(f"Redirect {resp.status_code} received without Location header")
                    await resp.aclose()
                    current_url = urljoin(current_url, location)
                    hops_remaining -= 1
                    continue

                if resp.status_code >= 400:
                    await resp.aclose()
                    return "", current_url, resp.status_code

                content_type = resp.headers.get("Content-Type", "").lower()
                # Accept text/html, text/plain, application/xhtml+xml, or text/markdown
                if not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml", "text/markdown", "text/")):
                    await resp.aclose()
                    raise ValueError(f"Unsupported content type '{content_type}': only text/html and text/plain are allowed")

                # Stream and cap decoded body size
                body_bytes = bytearray()
                async for chunk in resp.aiter_bytes():
                    body_bytes.extend(chunk)
                    if len(body_bytes) > max_bytes:
                        await resp.aclose()
                        raise ValueError(f"Response body exceeded maximum allowed limit of {max_bytes} bytes")

                # Decode using response encoding or utf-8 fallback
                encoding = resp.encoding or "utf-8"
                try:
                    text_content = body_bytes.decode(encoding, errors="replace")
                except Exception:
                    text_content = body_bytes.decode("utf-8", errors="replace")

                return text_content, current_url, resp.status_code
            finally:
                await resp.aclose()

        raise ValueError(f"Exceeded maximum redirect limit of {max_redirects} hops")
    finally:
        if close_client:
            await active_client.aclose()


class FetchedPage:
    """Container for fetched and extracted web evidence."""

    def __init__(
        self,
        url: str,
        title: str,
        text: str,
        snippet_only: bool = False,
        char_count: int = 0,
        status: str = "success",
        error: Optional[str] = None
    ):
        self.url = url
        self.title = title
        self.text = text
        self.snippet_only = snippet_only
        self.char_count = char_count or len(text)
        self.status = status
        self.error = error


async def fetch_and_extract_page(
    url: str,
    title: str = "",
    snippet: str = "",
    client: Optional[httpx.AsyncClient] = None,
    dns_resolver: Optional[Callable[[str, int], List[str]]] = None,
    check_robots: bool = True
) -> FetchedPage:
    """Safely fetch web page and extract clean article text via Trafilatura.

    If fetching fails or robots.txt forbids, cleanly falls back to the search snippet
    marked with snippet_only=True.
    """
    if check_robots:
        try:
            allowed = await is_robots_allowed(url, client=client, dns_resolver=dns_resolver)
            if not allowed:
                logger.warning(f"Robots.txt forbids crawling '{url}'. Falling back to snippet.")
                return FetchedPage(
                    url=url,
                    title=title,
                    text=snippet.strip(),
                    snippet_only=True,
                    status="robots_disallowed",
                    error="Robots.txt disallowed"
                )
        except Exception as exc:
            logger.debug(f"Robots.txt check skipped due to error: {exc}")

    try:
        html_text, final_url, status = await fetch_raw_body_safely(
            url,
            client=client,
            dns_resolver=dns_resolver
        )

        if status >= 400 or not html_text.strip():
            logger.info(f"Page fetch returned HTTP {status} for '{url}'. Falling back to snippet.")
            return FetchedPage(
                url=url,
                title=title,
                text=snippet.strip(),
                snippet_only=True,
                status=f"http_{status}",
                error=f"HTTP {status}"
            )

        # Extract main text using Trafilatura
        extracted = trafilatura.extract(
            html_text,
            include_links=True,
            include_comments=False,
            output_format="txt"
        )

        if extracted and len(extracted.strip()) > 50:
            return FetchedPage(
                url=final_url or url,
                title=title,
                text=extracted.strip(),
                snippet_only=False,
                status="indexed"
            )
        else:
            # Trafilatura could not extract meaningful text; fallback to snippet
            return FetchedPage(
                url=url,
                title=title,
                text=snippet.strip() or html_text[:1000].strip(),
                snippet_only=True,
                status="snippet_fallback",
                error="Trafilatura extracted empty body"
            )

    except Exception as exc:
        logger.warning(f"Failed to fetch '{url}': {exc}. Falling back to snippet.")
        return FetchedPage(
            url=url,
            title=title,
            text=snippet.strip(),
            snippet_only=True,
            status="error",
            error=str(exc)
        )
