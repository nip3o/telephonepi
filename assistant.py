import concurrent.futures
import json
import logging

import click
import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials
import grpc
from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc,
)
from gpiozero import Button
from tenacity import retry, retry_if_exception, stop_after_attempt

import audio_helpers


ASSISTANT_API_ENDPOINT = "embeddedassistant.googleapis.com"
END_OF_UTTERANCE = embedded_assistant_pb2.AssistResponse.END_OF_UTTERANCE
DIALOG_FOLLOW_ON = embedded_assistant_pb2.DialogStateOut.DIALOG_FOLLOW_ON
CLOSE_MICROPHONE = embedded_assistant_pb2.DialogStateOut.CLOSE_MICROPHONE
PLAYING = embedded_assistant_pb2.ScreenOutConfig.PLAYING
DEFAULT_GRPC_DEADLINE = 60 * 3 + 5
DEVICE_MODEL_ID = "telephonepi-telephonepi-01z0jk"
LANGUAGE_CODE = "en-US"
INPUT_PIN = 4


class Assistant(object):
    def __init__(
        self,
        device_id,
        conversation_stream,
        channel,
    ):
        self.language_code = LANGUAGE_CODE
        self.device_model_id = DEVICE_MODEL_ID
        self.display = False
        self.device_id = device_id
        self.conversation_stream = conversation_stream

        # Opaque blob provided in AssistResponse that,
        # when provided in a follow-up AssistRequest,
        # gives the Assistant a context marker within the current state
        # of the multi-Assist()-RPC "conversation".
        # This value, along with MicrophoneMode, supports a more natural
        # "conversation" with the Assistant.
        self.conversation_state = None
        # Force reset of first conversation.
        self.is_new_conversation = True

        # Create Google Assistant API gRPC client.
        self.assistant = embedded_assistant_pb2_grpc.EmbeddedAssistantStub(channel)
        self.deadline = DEFAULT_GRPC_DEADLINE

    def __enter__(self):
        return self

    def __exit__(self, etype, e, traceback):
        if e:
            return False
        self.conversation_stream.close()

    def is_grpc_error_unavailable(e):
        is_grpc_error = isinstance(e, grpc.RpcError)
        if is_grpc_error and (e.code() == grpc.StatusCode.UNAVAILABLE):
            logging.error("grpc unavailable error: %s", e)
            return True
        return False

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_exception(is_grpc_error_unavailable),
    )
    def assist(self):
        """Send a voice request to the Assistant and playback the response.

        Returns: True if conversation should continue.
        """
        continue_conversation = False
        device_actions_futures = []

        self.conversation_stream.start_recording()
        logging.info("Recording audio request.")

        def iter_log_assist_requests():
            for c in self.gen_assist_requests():
                yield c
            logging.debug("Reached end of AssistRequest iteration.")

        # This generator yields AssistResponse proto messages
        # received from the gRPC Google Assistant API.
        for resp in self.assistant.Assist(iter_log_assist_requests(), self.deadline):
            if resp.event_type == END_OF_UTTERANCE:
                logging.info("End of audio request detected.")
                logging.info("Stopping recording.")
                self.conversation_stream.stop_recording()
            if resp.speech_results:
                logging.info(
                    'Transcript of user request: "%s".',
                    " ".join(r.transcript for r in resp.speech_results),
                )
            if len(resp.audio_out.audio_data) > 0:
                if not self.conversation_stream.playing:
                    self.conversation_stream.stop_recording()
                    self.conversation_stream.start_playback()
                    logging.info("Playing assistant response.")
                self.conversation_stream.write(resp.audio_out.audio_data)
            if resp.dialog_state_out.conversation_state:
                conversation_state = resp.dialog_state_out.conversation_state
                logging.debug("Updating conversation state.")
                self.conversation_state = conversation_state
            if resp.dialog_state_out.volume_percentage != 0:
                volume_percentage = resp.dialog_state_out.volume_percentage
                logging.info("Setting volume to %s%%", volume_percentage)
                self.conversation_stream.volume_percentage = volume_percentage
            if resp.dialog_state_out.microphone_mode == DIALOG_FOLLOW_ON:
                continue_conversation = True
                logging.info("Expecting follow-on query from user.")
            elif resp.dialog_state_out.microphone_mode == CLOSE_MICROPHONE:
                continue_conversation = False

        if len(device_actions_futures):
            logging.info("Waiting for device executions to complete.")
            concurrent.futures.wait(device_actions_futures)

        logging.info("Finished playing assistant response.")
        self.conversation_stream.stop_playback()
        return continue_conversation

    def gen_assist_requests(self):
        """Yields: AssistRequest messages to send to the API."""

        config = embedded_assistant_pb2.AssistConfig(
            audio_in_config=embedded_assistant_pb2.AudioInConfig(
                encoding="LINEAR16",
                sample_rate_hertz=self.conversation_stream.sample_rate,
            ),
            audio_out_config=embedded_assistant_pb2.AudioOutConfig(
                encoding="LINEAR16",
                sample_rate_hertz=self.conversation_stream.sample_rate,
                volume_percentage=self.conversation_stream.volume_percentage,
            ),
            dialog_state_in=embedded_assistant_pb2.DialogStateIn(
                language_code=self.language_code,
                conversation_state=self.conversation_state,
                is_new_conversation=self.is_new_conversation,
            ),
            device_config=embedded_assistant_pb2.DeviceConfig(
                device_id=self.device_id, device_model_id=self.device_model_id
            ),
        )
        if self.display:
            config.screen_out_config.screen_mode = PLAYING
        # Continue current conversation with later requests.
        self.is_new_conversation = False
        # The first AssistRequest must contain the AssistConfig
        # and no audio data.
        yield embedded_assistant_pb2.AssistRequest(config=config)
        for data in self.conversation_stream:
            # Subsequent requests need audio data, but not config.
            yield embedded_assistant_pb2.AssistRequest(audio_in=data)


@click.command()
@click.option(
    "--device-id",
    metavar="<device id>",
    required=True,
    help="Unique registered device instance identifier.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging.")
def main(device_id, verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    with open("credentials.json", "r") as f:
        credentials = google.oauth2.credentials.Credentials(**json.load(f))
        http_request = google.auth.transport.requests.Request()
        credentials.refresh(http_request)

    grpc_channel = google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, ASSISTANT_API_ENDPOINT
    )
    logging.info("Connecting to %s", ASSISTANT_API_ENDPOINT)

    audio_sink = audio_source = audio_helpers.SoundDeviceStream(
        sample_rate=audio_helpers.DEFAULT_AUDIO_SAMPLE_RATE,
        sample_width=audio_helpers.DEFAULT_AUDIO_SAMPLE_WIDTH,
        block_size=audio_helpers.DEFAULT_AUDIO_DEVICE_BLOCK_SIZE,
        flush_size=audio_helpers.DEFAULT_AUDIO_DEVICE_FLUSH_SIZE,
    )

    # Create conversation stream with the given audio source and sink.
    conversation_stream = audio_helpers.ConversationStream(
        source=audio_source,
        sink=audio_sink,
        iter_size=audio_helpers.DEFAULT_AUDIO_ITER_SIZE,
        sample_width=audio_helpers.DEFAULT_AUDIO_SAMPLE_WIDTH,
    )

    trigger = Button(INPUT_PIN, pull_up=False)
    with Assistant(device_id, conversation_stream, grpc_channel) as assistant:
        print('Waiting for trigger...')
        trigger.wait_for_press()
        print('Triggered!')

        while trigger.is_pressed:
            print('Running assist')
            assistant.assist()
            print('Done with assist')


if __name__ == "__main__":
    main()
