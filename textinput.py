import json
import logging

import click
import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials
from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc,
)

ASSISTANT_API_ENDPOINT = "embeddedassistant.googleapis.com"
DEFAULT_GRPC_DEADLINE = 60 * 3 + 5
PLAYING = embedded_assistant_pb2.ScreenOutConfig.PLAYING
DEVICE_MODEL_ID = "telephonepi-telephonepi-01z0jk"
LANGUAGE_CODE = "en-US"


class SampleTextAssistant(object):
    def __init__(
        self, language_code, device_model_id, device_id, display, channel, deadline_sec
    ):
        self.language_code = language_code
        self.device_model_id = device_model_id
        self.device_id = device_id
        self.conversation_state = None
        # Force reset of first conversation.
        self.is_new_conversation = True
        self.display = display
        self.assistant = embedded_assistant_pb2_grpc.EmbeddedAssistantStub(channel)
        self.deadline = deadline_sec

    def __enter__(self):
        return self

    def __exit__(self, etype, e, traceback):
        if e:
            return False

    def assist(self, text_query):
        """Send a text request to the Assistant and playback the response.
        """

        def iter_assist_requests():
            config = embedded_assistant_pb2.AssistConfig(
                audio_out_config=embedded_assistant_pb2.AudioOutConfig(
                    encoding="LINEAR16", sample_rate_hertz=16000, volume_percentage=0
                ),
                dialog_state_in=embedded_assistant_pb2.DialogStateIn(
                    language_code=self.language_code,
                    conversation_state=self.conversation_state,
                    is_new_conversation=self.is_new_conversation,
                ),
                device_config=embedded_assistant_pb2.DeviceConfig(
                    device_id=self.device_id, device_model_id=self.device_model_id
                ),
                text_query=text_query,
            )
            # Continue current conversation with later requests.
            self.is_new_conversation = False
            if self.display:
                config.screen_out_config.screen_mode = PLAYING
            req = embedded_assistant_pb2.AssistRequest(config=config)
            yield req

        text_response = None
        html_response = None
        for resp in self.assistant.Assist(iter_assist_requests(), self.deadline):
            if resp.screen_out.data:
                html_response = resp.screen_out.data
            if resp.dialog_state_out.conversation_state:
                conversation_state = resp.dialog_state_out.conversation_state
                self.conversation_state = conversation_state
            if resp.dialog_state_out.supplemental_display_text:
                text_response = resp.dialog_state_out.supplemental_display_text
        return text_response, html_response


@click.command()
@click.option(
    "--device-id",
    metavar="<device id>",
    required=True,
    help="Unique registered device instance identifier.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging.")
def main(device_id, verbose, *args, **kwargs):
    # Setup logging.
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    with open("credentials.json", "r") as f:
        credentials = google.oauth2.credentials.Credentials(**json.load(f))
        http_request = google.auth.transport.requests.Request()
        credentials.refresh(http_request)

    channel = google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, ASSISTANT_API_ENDPOINT
    )
    logging.info("Connecting to %s", ASSISTANT_API_ENDPOINT)

    with SampleTextAssistant(
        LANGUAGE_CODE, DEVICE_MODEL_ID, device_id, False, channel, DEFAULT_GRPC_DEADLINE
    ) as assistant:
        while True:
            query = click.prompt("")
            click.echo("<you> %s" % query)
            response_text, response_html = assistant.assist(text_query=query)

            if response_text:
                click.echo("<@assistant> %s" % response_text)

            if response_html:
                click.echo(response_html)


if __name__ == "__main__":
    main()
