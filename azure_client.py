import os
import socket
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

GPT4O_DEPLOYMENT = "gpt-4o"
WHISPER_DEPLOYMENT = "whisper"


def _normalize_azure_endpoint(endpoint: str) -> str:
    if not endpoint:
        return endpoint
    endpoint = endpoint.strip().rstrip("/")
    if "/openai/deployments/" in endpoint:
        endpoint = endpoint.split("/openai/deployments/")[0]
    if endpoint.endswith("/openai"):
        endpoint = endpoint[: -len("/openai")]
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    return endpoint + "/"


def get_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")


def get_azure_endpoint() -> str:
    return _normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))


def get_client() -> AzureOpenAI:
    """GPT-4o client for frame analysis and work instructions."""
    return AzureOpenAI(
        azure_endpoint=get_azure_endpoint(),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=get_api_version(),
    )


def get_whisper_endpoint() -> str:
    whisper_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT", "").strip()
    if whisper_endpoint:
        return _normalize_azure_endpoint(whisper_endpoint)
    return get_azure_endpoint()


def get_whisper_client() -> AzureOpenAI:
    """Whisper client (separate audio transcription resource)."""
    return AzureOpenAI(
        azure_endpoint=get_whisper_endpoint(),
        api_key=os.getenv("AZURE_OPENAI_WHISPER_KEY") or os.getenv("AZURE_OPENAI_KEY"),
        api_version=get_api_version(),
    )


def get_gpt4o_model() -> str:
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", GPT4O_DEPLOYMENT)


def get_vision_model() -> str:
    return get_gpt4o_model()


def get_writer_model() -> str:
    return get_gpt4o_model()


def get_chat_model() -> str:
    return get_gpt4o_model()


def get_whisper_model() -> str:
    return os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT", WHISPER_DEPLOYMENT)


def _hostname_resolves(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        socket.getaddrinfo(hostname, 443)
        return True
    except OSError:
        return False


def check_endpoint_reachable(label: str, endpoint: str) -> None:
    """Fail fast with a clear message when an Azure hostname cannot be resolved."""
    host = urlparse(endpoint).hostname or ""
    if _hostname_resolves(host):
        return
    raise RuntimeError(
        f"\nCannot reach {label} endpoint — DNS lookup failed for:\n"
        f"  {host}\n\n"
        "This hostname does not exist or is not reachable from your network.\n"
        "Fix your .env:\n"
        "  1. Azure Portal → your Azure OpenAI resource → Keys and Endpoint\n"
        "  2. Copy the base Endpoint URL (not the full /deployments/... path)\n"
        "  3. Set AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/\n"
        "  4. Use the matching key from the same resource\n"
        "  5. Connect to company VPN if the resource is private\n"
    )


def validate_azure_config(*, check_whisper: bool = True) -> None:
    """Verify configured Azure endpoints resolve before starting a long pipeline run."""
    check_endpoint_reachable("GPT-4o", get_azure_endpoint())
    if check_whisper:
        check_endpoint_reachable("Whisper", get_whisper_endpoint())


def print_model_config() -> None:
    print("--- Model configuration ---")
    print(f"  Transcription : whisper ({get_whisper_model()}) @ {get_whisper_endpoint()}")
    print(f"  Vision (frames): gpt-4o ({get_gpt4o_model()}) @ {get_azure_endpoint()}")
    print(f"  Work instructions: gpt-4o ({get_gpt4o_model()}) @ {get_azure_endpoint()}")
    print("---------------------------")
