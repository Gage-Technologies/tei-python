import requests

from aiohttp import ClientSession, ClientTimeout
from typing import Dict, Optional, List

from tei.types import (
    EmbedRequest, InfoResponse,
)
from tei.errors import parse_error


class Client:
    """Client to make calls to a text-embedding-inference instance

     Example:

     ```python
     >>> from tei import Client

     >>> client = Client("http://localhost:8080")
     >>> client.embed("Why is the sky blue?")
     [[0.1, 0.2, 0.3, ...]]
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-embedding-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    def info(self) -> InfoResponse:
        """
        Get the model info

        Returns:
            InfoResponse: model info
        """
        resp = requests.get(
            self.base_url + "/info",
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return InfoResponse(**payload)

    def embed(
        self,
        inputs: str,
        truncate: bool = False,
    ) -> List[List[float]]:
        """
        Given a prompt, generate the following text

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            List[List[float]]: embedding for the text
        """
        request = EmbedRequest(inputs=inputs, truncate=truncate)

        resp = requests.post(
            self.base_url + "/embed",
            json=request.model_dump(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return payload


class AsyncClient:
    """Asynchronous Client to make calls to a text-embedding-inference instance

     Example:

     ```python
     >>> from tei import AsyncClient

     >>> client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> response = await client.embed("Why is the sky blue?")
     >>> response.embedding
     [0.1, 0.2, 0.3, ...]
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-embedding-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout * 60)

    async def info(self) -> InfoResponse:
        """
        Get the model info

        Returns:
            InfoResponse: model info
        """
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.get(self.base_url + "/info") as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return InfoResponse(**payload)

    async def embed(
        self,
        inputs: str,
    ) -> List[List[float]]:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            inputs (`str`):
                Input text that will be embedded

        Returns:
            List[List[float]]: embedding for the text
        """
        request = EmbedRequest(inputs=inputs)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.model_dump()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return payload

