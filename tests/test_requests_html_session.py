"""Tests for the HTMLSession response-wrapping fix.

Ensures that HTMLSession.get() (and all HTTP methods) always return an
HTMLResponse with a usable ``.html`` attribute, even when the requests
library silently discards the response-hook return value.

Upstream issue: https://github.com/psf/requests-html/issues/586
"""

from unittest.mock import patch, MagicMock
import requests

import pytest

# Import from the local copy of requests_html (repo root)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import requests_html  # noqa: E402
from requests_html import HTMLSession, HTMLResponse, BaseSession  # noqa: E402


def _make_plain_response(url="https://example.org/", content=b"<html><body>Hello</body></html>"):
    """Create a plain requests.Response (not HTMLResponse)."""
    resp = requests.Response()
    resp.status_code = 200
    resp._content = content
    resp.url = url
    resp.encoding = "utf-8"
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


class TestBaseSessionRequest:
    """Verify BaseSession.request always returns HTMLResponse."""

    def test_request_wraps_plain_response(self):
        """When the parent request() returns a plain Response (hook not applied),
        BaseSession.request() should still wrap it into an HTMLResponse."""
        session = HTMLSession()

        plain_resp = _make_plain_response()

        # Patch requests.Session.request to return a *plain* Response,
        # simulating the scenario where hooks are not applied.
        with patch.object(requests.Session, 'request', return_value=plain_resp):
            # Also prevent hooks from being dispatched by clearing them
            session.hooks['response'] = []
            result = session.get("https://example.org/")

        assert isinstance(result, HTMLResponse), (
            f"Expected HTMLResponse, got {type(result)}"
        )
        assert hasattr(result, 'html'), "Response should have .html attribute"
        assert result.url == "https://example.org/"

    def test_request_preserves_existing_html_response(self):
        """When the parent request() already returns HTMLResponse (hook worked),
        BaseSession.request() should not double-wrap it."""
        session = HTMLSession()

        html_resp = HTMLResponse(session=session)
        html_resp.status_code = 200
        html_resp._content = b"<html><body>Hi</body></html>"
        html_resp.url = "https://example.org/"
        html_resp.encoding = "utf-8"

        with patch.object(requests.Session, 'request', return_value=html_resp):
            result = session.get("https://example.org/")

        assert isinstance(result, HTMLResponse)
        assert result is html_resp, "Should return the same HTMLResponse object"

    def test_html_property_works(self):
        """The .html property on the returned response should produce
        an HTML object with content."""
        session = HTMLSession()
        plain_resp = _make_plain_response(
            content=b"<html><body><h1>Test</h1></body></html>"
        )

        with patch.object(requests.Session, 'request', return_value=plain_resp):
            session.hooks['response'] = []
            result = session.get("https://example.org/")

        html_obj = result.html
        assert html_obj is not None
        assert "Test" in html_obj.text

    def test_get_returns_html_response(self):
        """session.get() must return HTMLResponse."""
        session = HTMLSession()
        plain_resp = _make_plain_response()

        with patch.object(requests.Session, 'request', return_value=plain_resp):
            session.hooks['response'] = []
            result = session.get("https://example.org/")

        assert isinstance(result, HTMLResponse)

    def test_post_returns_html_response(self):
        """session.post() must return HTMLResponse."""
        session = HTMLSession()
        plain_resp = _make_plain_response()

        with patch.object(requests.Session, 'request', return_value=plain_resp):
            session.hooks['response'] = []
            result = session.post("https://example.org/")

        assert isinstance(result, HTMLResponse)

    def test_default_encoding_set_when_missing(self):
        """If the plain response has no encoding, it should be set to the default."""
        session = HTMLSession()
        plain_resp = _make_plain_response()
        plain_resp.encoding = None

        with patch.object(requests.Session, 'request', return_value=plain_resp):
            session.hooks['response'] = []
            result = session.get("https://example.org/")

        assert isinstance(result, HTMLResponse)
        assert result.encoding is not None


class TestHTMLResponseFromResponse:
    """Tests for HTMLResponse._from_response class method."""

    def test_from_response_copies_attributes(self):
        """_from_response should copy all attributes from the original response."""
        session = HTMLSession()
        plain_resp = _make_plain_response()

        html_resp = HTMLResponse._from_response(plain_resp, session)
        assert isinstance(html_resp, HTMLResponse)
        assert html_resp.status_code == 200
        assert html_resp.url == "https://example.org/"

    def test_from_response_has_html_property(self):
        """The wrapped response should have a working .html property."""
        session = HTMLSession()
        plain_resp = _make_plain_response(
            content=b"<html><body><p>paragraph</p></body></html>"
        )

        html_resp = HTMLResponse._from_response(plain_resp, session)
        assert hasattr(html_resp, 'html')
        html_obj = html_resp.html
        assert "paragraph" in html_obj.text


class TestResponseHook:
    """Tests for BaseSession.response_hook."""

    def test_response_hook_returns_html_response(self):
        """response_hook should return an HTMLResponse."""
        session = HTMLSession()
        plain_resp = _make_plain_response()

        result = session.response_hook(plain_resp)
        assert isinstance(result, HTMLResponse)

    def test_response_hook_sets_encoding(self):
        """response_hook should set default encoding if missing."""
        session = HTMLSession()
        plain_resp = _make_plain_response()
        plain_resp.encoding = None

        result = session.response_hook(plain_resp)
        assert result.encoding is not None
