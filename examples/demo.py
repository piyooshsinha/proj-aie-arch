"""Dev server with the echo provider, so the HTTP path can be exercised offline."""
from aie.actions.write import WriteAction
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
# Write actions are registered by the application, never by the platform --
# nothing can be written that an operator did not explicitly wire up.
SENT: list[tuple[str, str]] = []

platform.write_actions.register(
    WriteAction(
        name="send_email",
        description="Send an email to a customer",
        handler=lambda *, tenant_id, to, subject: SENT.append((to, subject)) or "sent",
        describe=lambda to, subject: f"Send an email to {to} with subject {subject!r}",
    )
)

app = create_app(platform)
