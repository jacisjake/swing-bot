"""
Pre-market press release scanner.

Polls free RSS feeds from major newswire services (GlobeNewswire, PR Newswire)
and FMP press release API to find overnight catalyst-driven news for stocks
that may match the momentum scanner thesis.

Covers ALL sectors — not just biotech/pharma. Major deals, contract awards,
earnings beats, and strategic partnerships move stocks in every industry.

Runs during pre-market (4:00-7:00 AM ET) to build a "catalyst watchlist"
before the trading window opens. Stocks found here get priority when the
momentum scanner runs.

Data flow:
    4:00-7:00 AM ET (every 5 min):
      1. Poll RSS feeds (GlobeNewswire all-news, PR Newswire multi-sector)
      2. Poll FMP press releases API (if key available)
      3. Extract tickers from headlines / content
      4. Quick-filter: is the stock $1-$10? Has recent volume?
      5. Store as CatalystHit with headline + source
      6. At 7:00 AM: merge catalyst watchlist with scanner results
"""

import json
import re
import time as time_mod
import xml.etree.ElementTree as ET_XML
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


# ── RSS Feed URLs (free, no API key) ────────────────────────────────────────

RSS_FEEDS = {
    # ── GlobeNewswire ──────────────────────────────────────────────────
    # Main all-news feed (covers every sector — most industry-specific
    # feeds outside biotech/pharma return empty results)
    "globenewswire_all": {
        "url": "https://www.globenewswire.com/RssFeed/feedTitle/GlobeNewswire",
        "source": "GlobeNewswire",
        "category": "All",
    },
    # Keep biotech/pharma feeds — they reliably return items and we
    # don't want to miss any if the all-news feed truncates
    "globenewswire_biotech": {
        "url": "https://www.globenewswire.com/RssFeed/industry/4573-Biotechnology/feedTitle/GlobeNewswire+-+Industry+News+on+Biotechnology",
        "source": "GlobeNewswire",
        "category": "Biotech",
    },
    "globenewswire_pharma": {
        "url": "https://www.globenewswire.com/RssFeed/industry/4577-Pharmaceuticals/feedTitle/GlobeNewswire+-+Industry+News+on+Pharmaceuticals",
        "source": "GlobeNewswire",
        "category": "Pharma",
    },
    "globenewswire_healthcare": {
        "url": "https://www.globenewswire.com/RssFeed/industry/4000-Health+Care/feedTitle/GlobeNewswire+-+Industry+News+on+Health+Care",
        "source": "GlobeNewswire",
        "category": "Healthcare",
    },
    # ── PR Newswire ────────────────────────────────────────────────────
    # All news (catch-all — every sector)
    "prnewswire_all": {
        "url": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "source": "PR Newswire",
        "category": "All",
    },
    "prnewswire_health": {
        "url": "https://www.prnewswire.com/rss/health-latest-news/health-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Health",
    },
    "prnewswire_technology": {
        "url": "https://www.prnewswire.com/rss/technology-latest-news/technology-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Technology",
    },
    "prnewswire_energy": {
        "url": "https://www.prnewswire.com/rss/energy-latest-news/energy-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Energy",
    },
    "prnewswire_consumer": {
        "url": "https://www.prnewswire.com/rss/consumer-products-latest-news/consumer-products-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Consumer",
    },
    "prnewswire_entertainment": {
        "url": "https://www.prnewswire.com/rss/entertainment-media-latest-news/entertainment-media-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Entertainment",
    },
    "prnewswire_general_business": {
        "url": "https://www.prnewswire.com/rss/general-business-latest-news/general-business-latest-news-list.rss",
        "source": "PR Newswire",
        "category": "Business",
    },
}

# Keywords that suggest a positive catalyst (deals, approvals, earnings, etc.)
POSITIVE_CATALYST_KEYWORDS = [
    # FDA / regulatory
    "fda approv", "fda grant", "fda accept", "fda clear",
    "breakthrough therapy", "fast track", "priority review",
    "rmat", "orphan drug", "accelerated approval",
    "nda accept", "bla accept", "ind clear",
    "eua", "emergency use",
    # Clinical trials
    "positive data", "positive results", "met primary endpoint",
    "statistically significant", "phase 3", "phase iii",
    "topline results", "pivotal trial", "clinical milestone",
    # Business deals / M&A / mergers (assume all mergers spike volatility)
    "contract award", "receives order", "partnership",
    "collaboration agreement", "license agreement",
    "strategic agreement", "framework agreement",
    "exclusive agent", "exclusive partner", "exclusive deal",
    "joint venture", "signs deal", "signs agreement",
    "definitive agreement", "merger agreement",
    "acquisition of", "completes acquisition", "agrees to acquire",
    "agrees to be acquired", "buyout", "tender offer",
    "million deal", "billion deal",
    "merger of", "announces merger", "announces the merger",
    "to merge with", "merge with", "merges with",
    "proposed merger", "completed merger", "pending merger",
    "to be acquired", "acquisition agreement", "acquisition by",
    # Financial / earnings (earnings announcements spike volatility — worth tracking)
    "revenue increase", "record revenue", "beats estimate",
    "earnings beat", "raises guidance", "increases guidance",
    "exceeds expectations", "record earnings", "record profit",
    "stock offering", "public offering",
    "debt reduction", "reduces debt", "pays off debt",
    "reports earnings", "earnings results", "quarterly results",
    "quarterly earnings", "annual earnings", "fiscal year results",
    "financial results", "fourth quarter", "first quarter",
    "second quarter", "third quarter", "q1 results", "q2 results",
    "q3 results", "q4 results", "earnings call",
    "reports first quarter", "reports second quarter",
    "reports third quarter", "reports fourth quarter",
    "full year results", "annual results",
    # Product launches / breakthroughs
    "announces launch", "product launch", "commercialization",
    "new product", "next-generation", "next generation",
    "first-of-its-kind", "first of its kind", "industry first",
    "breakthrough", "revolutionary", "patent grant", "patent approv",
    "receives patent", "key patent", "new platform",
    "unveils", "introduces", "debuts",
    # Growth / expansion
    "expansion", "new market", "enters market",
    "major contract", "government contract", "defense contract",
    "procurement", "supply agreement",
    "record orders", "backlog increase", "order backlog",
]

# Keywords that suggest a negative catalyst (avoid these)
NEGATIVE_CATALYST_KEYWORDS = [
    "complete response letter", "crl", "refuse to file",
    "clinical hold", "partial hold",
    "fails to meet", "did not meet", "negative results",
    "discontinued", "terminates", "withdraws",
    "delisted", "bankruptcy", "chapter 11",
    "sec investigation", "sec charges",
    "recall", "safety concern", "black box warning",
    "going concern", "reverse stock split",
]

# US exchange ticker patterns — only match stocks listed on US exchanges.
# Pattern 1: Explicit US exchange reference like "(NASDAQ: ABCD)" — guaranteed US-listed.
# Pattern 2: Generic "(ticker: ABCD)" — needs broker validation.
# Pattern 3: Company name suffix like "Inc. (ABCD)" — needs broker validation.
US_EXCHANGE_PATTERN = re.compile(
    r"\((?:NASDAQ|NYSE|NYSEAMERICAN|OTC(?:QB|QX)?|AMEX|CBOE)\s*:\s*([A-Z]{1,5})\)",
    re.IGNORECASE,
)
TICKER_PATTERNS = [
    US_EXCHANGE_PATTERN,
    re.compile(r"\((?:ticker|stock|symbol)\s*:\s*([A-Z]{1,5})\)", re.IGNORECASE),
    # Matches: "ABCD" after company name patterns
    re.compile(r"(?:Inc\.|Corp\.|Ltd\.|LLC|Company|Therapeutics|Pharmaceuticals|Biosciences|Biotech)\s*\(([A-Z]{1,5})\)"),
]

# Non-US exchanges to reject (common in GlobeNewswire all-news feed)
NON_US_EXCHANGE_PATTERNS = re.compile(
    r"\((?:TSX|TSXV|TSE|LSE|LON|ASX|HKG|HKEX|SGX|KRX|TYO|FRA|ETR|SWX"
    r"|BMV|JSE|NSE|BSE|MOEX|B3|BVMF|NZX|OSE|HEL|CPH|STO|WSE)\s*:\s*[A-Z]",
    re.IGNORECASE,
)


@dataclass
class CatalystHit:
    """A press release that may indicate a catalyst for a stock."""

    symbol: str
    headline: str
    source: str  # "GlobeNewswire", "PR Newswire", "FMP"
    category: str  # "Biotech", "Pharma", "Healthcare", "General"
    url: str
    published: Optional[datetime] = None
    sentiment: str = "neutral"  # "positive", "negative", "neutral"
    matched_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "headline": self.headline,
            "source": self.source,
            "category": self.category,
            "url": self.url,
            "published": self.published.isoformat() if self.published else None,
            "sentiment": self.sentiment,
            "matched_keywords": self.matched_keywords,
        }


class PressReleaseScanner:
    """
    Scans free newswire RSS feeds and FMP API for overnight press releases.

    Extracts stock tickers from headlines, classifies sentiment, and builds
    a catalyst watchlist for the momentum scanner.
    """

    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        rss_feeds: Optional[dict] = None,
        lookback_hours: int = 16,
        request_timeout: int = 10,
        trading_client=None,
        state_path: str = "state/press_releases.json",
        retention_days: int = 7,
    ):
        """
        Initialize press release scanner.

        Args:
            fmp_api_key: Financial Modeling Prep API key (optional, for press releases endpoint)
            rss_feeds: Custom RSS feed config (defaults to built-in feeds)
            lookback_hours: How far back to look for press releases (16h = covers overnight)
            request_timeout: HTTP request timeout in seconds
            trading_client: TastytradeClient instance for ticker validation
            state_path: Path to JSON file for rolling persistence
            retention_days: How many days to keep press releases (default 7)
        """
        self.fmp_api_key = fmp_api_key
        self.rss_feeds = rss_feeds or RSS_FEEDS
        self.lookback_hours = lookback_hours
        self.request_timeout = request_timeout
        self._state_path = Path(state_path)
        self._retention_days = retention_days

        # Results cache
        self._hits: list[CatalystHit] = []
        self._last_scan_time: Optional[datetime] = None
        self._seen_urls: set[str] = set()  # Dedup across scans within same day

        # US-exchange validation cache (avoid repeated API calls)
        # Maps ticker -> True (US-tradable) or False (not found / non-US)
        self._us_ticker_cache: dict[str, bool] = {}
        self._trading_client = trading_client

        # Rate limiting for RSS feeds
        self._last_feed_fetch: dict[str, float] = {}
        self._min_fetch_interval = 120  # 2 min between same-feed fetches

        # Load persisted state
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted press releases from JSON file."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
            for item in data.get("hits", []):
                published = None
                if item.get("published"):
                    try:
                        published = datetime.fromisoformat(item["published"])
                    except (ValueError, TypeError):
                        pass
                # Prune old entries on load
                if published and published < cutoff:
                    continue
                hit = CatalystHit(
                    symbol=item["symbol"],
                    headline=item["headline"],
                    source=item.get("source", "unknown"),
                    category=item.get("category", "General"),
                    url=item.get("url", ""),
                    published=published,
                    sentiment=item.get("sentiment", "neutral"),
                    matched_keywords=item.get("matched_keywords", []),
                )
                self._hits.append(hit)
                self._seen_urls.add(hit.url)
            if self._hits:
                logger.info(f"[PR-SCAN] Loaded {len(self._hits)} press releases from disk")
        except Exception as e:
            logger.error(f"[PR-SCAN] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist press releases to JSON file, pruning old entries."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
            hits_to_save = [
                h for h in self._hits
                if not h.published or h.published >= cutoff
            ]
            data = {
                "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
                "hits": [h.to_dict() for h in hits_to_save],
            }
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[PR-SCAN] Failed to save state: {e}")

    def scan(self) -> list[CatalystHit]:
        """
        Run a full press release scan across all sources.

        Returns:
            List of CatalystHit objects, sorted by published time (newest first)
        """
        logger.info("[PR-SCAN] Scanning press releases for catalysts...")
        new_hits: list[CatalystHit] = []

        # 1. Poll RSS feeds
        for feed_id, feed_config in self.rss_feeds.items():
            try:
                hits = self._fetch_rss_feed(feed_id, feed_config)
                new_hits.extend(hits)
            except Exception as e:
                logger.debug(f"[PR-SCAN] RSS feed {feed_id} error: {e}")

        # 2. Poll FMP press releases API
        if self.fmp_api_key:
            try:
                fmp_hits = self._fetch_fmp_press_releases()
                new_hits.extend(fmp_hits)
            except Exception as e:
                logger.debug(f"[PR-SCAN] FMP press releases error: {e}")

        # Deduplicate by URL
        unique_hits = []
        for hit in new_hits:
            if hit.url not in self._seen_urls:
                self._seen_urls.add(hit.url)
                unique_hits.append(hit)

        # Add to cumulative hits
        self._hits.extend(unique_hits)

        # Sort by published time (newest first)
        self._hits.sort(
            key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        self._last_scan_time = datetime.now()

        if unique_hits:
            logger.info(f"[PR-SCAN] Found {len(unique_hits)} new press releases with tickers")
            for hit in unique_hits[:5]:  # Log top 5
                logger.info(
                    f"  [{hit.sentiment.upper()}] {hit.symbol} ({hit.source}): "
                    f"{hit.headline[:80]}..."
                )
        else:
            logger.debug("[PR-SCAN] No new press releases with extractable tickers")

        self._save_state()
        return unique_hits

    def get_catalyst_symbols(self, positive_only: bool = True) -> list[str]:
        """
        Get unique symbols from catalyst hits.

        Args:
            positive_only: Only return symbols with positive sentiment

        Returns:
            List of unique ticker symbols
        """
        seen = set()
        result = []
        for hit in self._hits:
            if positive_only and hit.sentiment != "positive":
                continue
            if hit.symbol not in seen:
                seen.add(hit.symbol)
                result.append(hit.symbol)
        return result

    def get_hits_for_symbol(self, symbol: str) -> list[CatalystHit]:
        """Get all catalyst hits for a specific symbol."""
        return [h for h in self._hits if h.symbol == symbol]

    def reset_daily(self) -> None:
        """Reset for a new trading day (clears dedup cache, keeps persisted history)."""
        self._seen_urls.clear()
        self._us_ticker_cache.clear()
        self._last_scan_time = None
        # Reload from disk (retains rolling history, prunes old entries)
        self._hits.clear()
        self._load_state()
        logger.info(f"[PR-SCAN] Daily reset — reloaded {len(self._hits)} persisted hits")

    # ── RSS Feed Parsing ──────────────────────────────────────────────────

    def _fetch_rss_feed(
        self, feed_id: str, feed_config: dict
    ) -> list[CatalystHit]:
        """
        Fetch and parse an RSS feed for press releases with tickers.

        Args:
            feed_id: Feed identifier (for rate limiting)
            feed_config: Feed URL + metadata

        Returns:
            List of CatalystHit objects found in the feed
        """
        # Rate limit: don't hammer the same feed
        now = time_mod.time()
        last_fetch = self._last_feed_fetch.get(feed_id, 0)
        if now - last_fetch < self._min_fetch_interval:
            return []

        url = feed_config["url"]
        source = feed_config["source"]
        category = feed_config["category"]

        try:
            response = requests.get(
                url,
                timeout=self.request_timeout,
                headers={
                    "User-Agent": "MomentumTrader/1.0 (catalyst scanner)",
                    "Accept": "application/rss+xml, application/xml, text/xml",
                },
            )
            response.raise_for_status()
            self._last_feed_fetch[feed_id] = now

        except requests.RequestException as e:
            logger.debug(f"[PR-SCAN] Failed to fetch {feed_id}: {e}")
            return []

        # Parse XML
        try:
            root = ET_XML.fromstring(response.content)
        except ET_XML.ParseError as e:
            logger.debug(f"[PR-SCAN] XML parse error for {feed_id}: {e}")
            return []

        # Find all <item> elements (standard RSS 2.0)
        items = root.findall(".//item")
        if not items:
            # Try Atom format
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//atom:entry", ns)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        hits = []

        for item in items:
            hit = self._parse_rss_item(item, source, category, cutoff)
            if hit:
                hits.append(hit)

        return hits

    def _parse_rss_item(
        self,
        item: ET_XML.Element,
        source: str,
        category: str,
        cutoff: datetime,
    ) -> Optional[CatalystHit]:
        """
        Parse a single RSS item into a CatalystHit (if it has a ticker).

        Args:
            item: XML element (<item> or <entry>)
            source: Feed source name
            category: Feed category
            cutoff: Ignore items older than this

        Returns:
            CatalystHit if ticker found, None otherwise
        """
        # Extract fields (RSS 2.0 format)
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        desc_el = item.find("description")

        # Try Atom format if RSS fields missing
        if title_el is None:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            title_el = item.find("atom:title", ns)
            link_el = item.find("atom:link", ns)
            pubdate_el = item.find("atom:published", ns) or item.find("atom:updated", ns)
            desc_el = item.find("atom:summary", ns) or item.find("atom:content", ns)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        if not title:
            return None

        # Get link
        link = ""
        if link_el is not None:
            link = link_el.text.strip() if link_el.text else link_el.get("href", "")

        # Skip if already seen
        if link and link in self._seen_urls:
            return None

        # Parse publish date
        published = self._parse_date(pubdate_el)
        if published and published < cutoff:
            return None  # Too old

        # Get description text for additional context
        desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""

        # Extract ticker symbol from title + description
        combined_text = f"{title} {desc}"
        symbol = self._extract_ticker(combined_text)
        if not symbol:
            return None

        # Classify sentiment
        sentiment, matched = self._classify_sentiment(title.lower())

        return CatalystHit(
            symbol=symbol,
            headline=title,
            source=source,
            category=category,
            url=link,
            published=published,
            sentiment=sentiment,
            matched_keywords=matched,
        )

    # ── FMP Press Releases API ────────────────────────────────────────────

    def _fetch_fmp_press_releases(self) -> list[CatalystHit]:
        """
        Fetch recent press releases from Financial Modeling Prep API.

        Uses the RSS feed endpoint which returns the latest press releases
        across all companies (not symbol-specific).

        Returns:
            List of CatalystHit objects
        """
        if not self.fmp_api_key:
            return []

        # FMP press release RSS feed (latest across all symbols)
        url = (
            f"https://financialmodelingprep.com/api/v3/press-releases"
            f"?page=0&apikey={self.fmp_api_key}"
        )

        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.debug(f"[PR-SCAN] FMP API error: {e}")
            return []

        if not isinstance(data, list):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        hits = []

        for item in data:
            symbol = item.get("symbol", "")
            title = item.get("title", "")
            date_str = item.get("date", "")
            text = item.get("text", "")

            if not symbol or not title:
                continue

            # Parse date
            published = None
            if date_str:
                try:
                    published = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if published and published < cutoff:
                continue

            # Build a fake URL for dedup
            pr_url = f"fmp://press-release/{symbol}/{date_str}"
            if pr_url in self._seen_urls:
                continue

            # Classify sentiment from title
            sentiment, matched = self._classify_sentiment(title.lower())

            hits.append(CatalystHit(
                symbol=symbol,
                headline=title,
                source="FMP",
                category="Press Release",
                url=pr_url,
                published=published,
                sentiment=sentiment,
                matched_keywords=matched,
            ))

        return hits

    # ── Ticker Extraction ─────────────────────────────────────────────────

    def _extract_ticker(self, text: str) -> Optional[str]:
        """
        Extract a US-listed stock ticker from press release text.

        Only returns tickers for stocks traded on US exchanges (NASDAQ, NYSE,
        AMEX, OTC). Rejects tickers from TSX, LSE, ASX, and other non-US
        exchanges.

        Looks for patterns like:
        - "(NASDAQ: ABCD)" — guaranteed US-listed
        - "(NYSE: XYZ)" — guaranteed US-listed
        - "Company Inc. (ABCD)" — validated via broker API

        Args:
            text: Combined title + description text

        Returns:
            Ticker symbol (uppercase) or None
        """
        # Quick reject: if the text mentions a non-US exchange, skip entirely
        if NON_US_EXCHANGE_PATTERNS.search(text):
            return None

        for pattern in TICKER_PATTERNS:
            match = pattern.search(text)
            if match:
                ticker = match.group(1).upper()
                # Basic validation: 1-5 uppercase letters, not a common word
                if not self._is_valid_ticker(ticker):
                    continue

                # If matched via the explicit US exchange pattern, it's good
                if pattern is US_EXCHANGE_PATTERN:
                    return ticker

                # Otherwise validate it's actually US-tradable
                if self._is_us_tradable(ticker):
                    return ticker

        return None

    def _is_us_tradable(self, ticker: str) -> bool:
        """
        Check if a ticker is tradable on US exchanges.

        Uses a cache to avoid hammering the API. Tickers confirmed once
        stay cached for the entire session (daily reset clears cache).

        Args:
            ticker: Uppercase ticker symbol

        Returns:
            True if the symbol is active and tradable
        """
        if ticker in self._us_ticker_cache:
            return self._us_ticker_cache[ticker]

        try:
            if self._trading_client is None:
                self._us_ticker_cache[ticker] = False
                return False

            asset = self._trading_client.get_asset(ticker)
            is_tradable = bool(asset and asset.get("name"))
            self._us_ticker_cache[ticker] = is_tradable
            return is_tradable

        except Exception:
            # Symbol not found or API error — assume not US-tradable
            self._us_ticker_cache[ticker] = False
            return False

    @staticmethod
    def _is_valid_ticker(ticker: str) -> bool:
        """
        Validate a potential ticker symbol.

        Filters out common English words that match the ticker regex.
        """
        if not ticker or len(ticker) > 5:
            return False

        # Common false positives (English words that look like tickers)
        false_positives = {
            "A", "I", "AN", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
            "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR",
            "SO", "TO", "UP", "US", "WE", "AM", "AS", "HAS", "FOR",
            "ALL", "ARE", "BUT", "CAN", "HAS", "HER", "HIM", "HIS",
            "HOW", "ITS", "MAY", "NEW", "NOT", "NOW", "OLD", "OUR",
            "OUT", "OWN", "SAY", "SHE", "TOO", "USE", "WAY", "WHO",
            "THE", "AND", "INC", "FDA", "SEC", "CEO", "CFO", "COO",
            "IPO", "LLC", "LTD", "USA", "NYC", "EST", "USD", "EUR",
        }
        return ticker not in false_positives

    # ── Sentiment Classification ──────────────────────────────────────────

    @staticmethod
    def _classify_sentiment(text_lower: str) -> tuple[str, list[str]]:
        """
        Classify press release sentiment based on keyword matching.

        Args:
            text_lower: Lowercased headline/title text

        Returns:
            Tuple of (sentiment, matched_keywords)
            sentiment is "positive", "negative", or "neutral"
        """
        pos_matches = []
        neg_matches = []

        for kw in POSITIVE_CATALYST_KEYWORDS:
            if kw in text_lower:
                pos_matches.append(kw)

        for kw in NEGATIVE_CATALYST_KEYWORDS:
            if kw in text_lower:
                neg_matches.append(kw)

        # Negative keywords override positive ones
        if neg_matches:
            return "negative", neg_matches
        elif pos_matches:
            return "positive", pos_matches
        else:
            return "neutral", []

    # ── Date Parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(date_el: Optional[ET_XML.Element]) -> Optional[datetime]:
        """
        Parse a date from various RSS date formats.

        Handles:
        - RFC 2822: "Mon, 09 Feb 2026 14:30:00 -0500"
        - ISO 8601: "2026-02-09T14:30:00Z"
        - Various other common formats
        """
        if date_el is None or not date_el.text:
            return None

        date_str = date_el.text.strip()

        # Try common formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",       # RFC 2822 with tz
            "%a, %d %b %Y %H:%M:%S %Z",       # RFC 2822 with zone name
            "%Y-%m-%dT%H:%M:%S%z",             # ISO 8601 with tz
            "%Y-%m-%dT%H:%M:%SZ",              # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",               # Simple datetime
            "%Y-%m-%d",                         # Date only
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        # Try ISO format as fallback
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass

        return None

    # ── Status / Info ─────────────────────────────────────────────────────

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """When the last scan was performed."""
        return self._last_scan_time

    @property
    def hits(self) -> list[CatalystHit]:
        """All catalyst hits from current session."""
        return self._hits

    @property
    def positive_hits(self) -> list[CatalystHit]:
        """Only positive-sentiment catalyst hits."""
        return [h for h in self._hits if h.sentiment == "positive"]

    def get_status(self) -> dict:
        """Get scanner status for API/dashboard."""
        return {
            "total_hits": len(self._hits),
            "positive_hits": len(self.positive_hits),
            "negative_hits": len([h for h in self._hits if h.sentiment == "negative"]),
            "neutral_hits": len([h for h in self._hits if h.sentiment == "neutral"]),
            "unique_symbols": len(set(h.symbol for h in self._hits)),
            "positive_symbols": self.get_catalyst_symbols(positive_only=True),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "feeds_configured": len(self.rss_feeds),
            "fmp_enabled": bool(self.fmp_api_key),
        }
