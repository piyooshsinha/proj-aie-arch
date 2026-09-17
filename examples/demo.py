"""Dev server with the echo provider, so the HTTP path can be exercised offline."""
from aie.api.app import create_app
from aie.observe.logging import configure_logging
from aie.config import Settings, build_platform
from aie.gateway.providers.echo import EchoProvider
from aie.store.memory import Document

configure_logging()

platform = build_platform(
    Settings(primary="local", local_model="demo"),
    documents=[
        Document("d1", "Refunds are issued within 14 days for EU orders.", "policy"),
        Document("d2", "US orders are refunded within 30 days of purchase.", "policy"),
    ],
    providers={"local": EchoProvider("Refunds take 14 days for EU orders [policy#d1].")},
)
app = create_app(platform)
