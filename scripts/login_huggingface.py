"""Interactive local login; never write tokens to experiment logs or Git."""

from getpass import getpass
import sys

from huggingface_hub import login
from requests.exceptions import RequestException


def main():
    token = getpass('Paste Hugging Face token (hidden), then press Enter: ').strip()
    if not token.startswith('hf_') or not token.isascii() or any(c.isspace() for c in token):
        print('INPUT_FORMAT_ERROR: Copy the complete hf_ token, without quotes or internal spaces.')
        return 1
    try:
        login(token=token, add_to_git_credential=False)
    except RequestException as error:
        response = getattr(error, 'response', None)
        status = response.status_code if response is not None else None
        # Avoid dumping request headers or the exception message containing user data.
        print('LOGIN_FAILED_HTTP_STATUS:', status)
        if status == 400:
            print('HTTP_400: The service rejected the request; token validity is not established.')
            print('Next diagnostic: check network/proxy route before reinstalling packages.')
        elif status == 401:
            print('HTTP_401: Verify that this is a complete, active Hugging Face token.')
        elif status == 403:
            print('HTTP_403: Check account/token permissions and service access restrictions.')
        else:
            print('NETWORK_OR_SERVICE_ERROR:', type(error).__name__)
        return 1
    except (ValueError, OSError) as error:
        print('LOCAL_LOGIN_ERROR:', type(error).__name__)
        return 1
    finally:
        token = None
    print('Login successful. Model access will be checked separately in the next step.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
