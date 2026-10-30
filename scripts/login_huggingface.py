"""Interactive local login; never write tokens to experiment logs or Git."""

from getpass import getpass
import argparse
import subprocess
import sys

from huggingface_hub import login
from requests.exceptions import RequestException


def input_checks(token):
    return {
        'INPUT_EMPTY': not bool(token),
        'PREFIX_HF_OK': token.startswith('hf_'),
        'ASCII_ONLY': token.isascii(),
        'HAS_WHITESPACE': any(c.isspace() for c in token),
        'HAS_CONTROL_CHARACTERS': any(ord(c) < 32 or ord(c) == 127 for c in token),
        'HAS_QUOTES': any(c in token for c in ['\"', "'", '\u201c', '\u201d', '\u2018', '\u2019']),
    }


def read_clipboard():
    # Invoked only by the user with --clipboard, after explicit Enter confirmation.
    # Capture all subprocess output in memory; never echo clipboard content.
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
         '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); Get-Clipboard -Raw'],
        capture_output=True, text=True, encoding='utf-8', timeout=10, check=False,
    )
    if result.returncode != 0:
        raise OSError('Clipboard read failed')
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clipboard', action='store_true',
                        help='Windows: read copied token locally after Enter; never print it')
    args = parser.parse_args()
    try:
        if args.clipboard:
            input('Copy your HF token NOW, return here, and press Enter (do not paste): ')
            token = read_clipboard()
        else:
            token = getpass('Paste Hugging Face token (hidden), then press Enter: ').strip()
    except (OSError, subprocess.TimeoutExpired, UnicodeError, EOFError) as error:
        print('INPUT_READ_FAILED:', type(error).__name__)
        return 1
    checks = input_checks(token)
    if (checks['INPUT_EMPTY'] or not checks['PREFIX_HF_OK'] or not checks['ASCII_ONLY']
            or checks['HAS_WHITESPACE'] or checks['HAS_CONTROL_CHARACTERS'] or checks['HAS_QUOTES']):
        print('INPUT_FORMAT_ERROR: No request sent. Diagnostics contain no token text:')
        for name, value in checks.items():
            print(f'{name}: {value}')
        token = None
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
