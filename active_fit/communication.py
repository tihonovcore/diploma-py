import subprocess
from enum import Enum

from configuration import Configuration


class RequestType(Enum):
    EXTRACT_PATHS = 1
    ON_PREDICT = 2


class ResponseStatus(Enum):
    PATH = 1
    SUCC = 2
    FAIL = 3
    ERROR = 4


def call_kotlin_compiler(request_type: RequestType, request_data: str) -> ResponseStatus:
    with open(Configuration.request, 'w') as communication_file:
        communication_file.write('{ request_type: "' + str(request_type) + '", request: "' + request_data + '" }')

    _ = subprocess.run(Configuration.bash_compiler, capture_output=True, shell=True)

    with open(Configuration.cooperative__take) as response_from_kotlin:
        status = response_from_kotlin.read()

    if status == 'PATH':
        return ResponseStatus.PATH
    elif status == 'SUCC':
        return ResponseStatus.SUCC
    elif status == 'FAIL':
        return ResponseStatus.FAIL
    elif status.startswith('ERROR'):
        print(status)
        return ResponseStatus.ERROR
    else:
        print('!!! Unexpected status:')
        print(status)
        return ResponseStatus.ERROR
